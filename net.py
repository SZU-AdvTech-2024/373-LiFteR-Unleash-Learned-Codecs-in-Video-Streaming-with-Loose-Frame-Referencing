import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *
from subnet.endecoder import Tree_Spynet
import torchac

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        # pretrained_dict = torch.load(f)
        pretrained_dict = torch.load(f, weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0



class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]

        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)


# distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # psnr = tf.cond(
        #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
        #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))
        

# bit per pixel

        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob


        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv
        
        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
        

class InterFrameLayer(nn.Module):
    def __init__(self, num_frames, num_channels, num_heads=8, reduction_ratio=4):
        super(InterFrameLayer, self).__init__()
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.key_proj = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.value_proj = nn.Linear(num_channels, num_channels // reduction_ratio)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_channels // reduction_ratio, num_heads=num_heads)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x):
        """
        Input: x of shape (F, C, H, W)
        Output: x of shape (F, C, H, W)
        """
        F, C, H, W = x.shape
        x = x.view(F, C, -1).permute(2, 0, 1)  # Reshape to (HW, F, C)

        # Linear projections
        query = self.query_proj(x)  # (HW, F, C//r)
        key = self.key_proj(x)      # (HW, F, C//r)
        value = self.value_proj(x)  # (HW, F, C//r)

        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)  # (HW, F, C//r)

        # Feedforward network
        output = self.feedforward(attn_output)  # (HW, F, C)

        # Reshape back to (F, C, H, W)
        output = output.permute(1, 2, 0).view(F, C, H, W)
        return output

# without IFL
# class LFRVideoCompressor(nn.Module):
#     def __init__(self):
#         super(LFRVideoCompressor, self).__init__()
#         # self.imageCompressor = ImageCompressor()
#         self.opticFlow = Tree_Spynet()
#         self.mvEncoder = Analysis_mv_net()
#         self.Q = None
#         self.mvDecoder = Synthesis_mv_net()
#         self.warpnet = Warp_net()
#         self.resEncoder = Analysis_net()
#         self.resDecoder = Synthesis_net()
#         self.respriorEncoder = Analysis_prior_net()
#         self.respriorDecoder = Synthesis_prior_net()
#         self.bitEstimator_z = BitEstimator(out_channel_N)
#         self.bitEstimator_mv = BitEstimator(out_channel_mv)
#         # self.flow_warp = Resample2d()
#         # self.bitEstimator_feature = BitEstimator(out_channel_M)
#         self.warp_weight = 0
#         self.mxrange = 150
#         self.calrealbits = False

#     def forwardFirstFrame(self, x):
#         output, bittrans = self.imageCompressor(x)
#         cost = self.bitEstimator(bittrans)
#         return output, cost

#     def motioncompensation(self, ref, mv):
#         warpframe = flow_warp(ref, mv)
#         inputfeature = torch.cat((warpframe, ref), 1)
#         prediction = self.warpnet(inputfeature) + warpframe
#         return prediction, warpframe

#     def forward(self, input_images, tree_levels, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
#         # 计算光流估计
#         estmv = self.opticFlow(input_images, tree_levels)
        
#         # 将光流特征编码 
#         mvfeature = self.mvEncoder(estmv)
#         # mvfeature torch.Size([subGOP_F, 128, 16, 16])
#         # estmv torch.Size([subGOP_F, 2, 256, 256])
        
#         # 如果模型处于训练状态，则添加量化噪声；否则，对特征进行四舍五入
#         if self.training:
#             quant_mv = mvfeature + quant_noise_mv
#         else:
#             quant_mv = torch.round(mvfeature)
        
#         # 对量化的光流特征进行上采样
#         quant_mv_upsample = self.mvDecoder(quant_mv)


#         clipped_recon_images=[0] * len(input_images)
#         warpframes = [0] * len(input_images)
#         predictions=[0] * len(input_images)
#         compressed_features_renorm=[0] * len(input_images)
#         recon_sigmas=[0] * len(input_images)
#         for level_idx, level_nodes in enumerate(tree_levels):
#             if level_idx == 0:
#                 continue
#             # 使用列表推导式创建索引列表
#             indices = [node.val - 1 for node in level_nodes]

#             # 使用Tensor的索引功能直接提取所需的元素
#             level_qmv_upsamples = quant_mv_upsample[indices]
            
#             if level_idx==1: # 依赖I帧
#                 ref_images = [input_images[node.parent.val].squeeze(0) for node in level_nodes]
#                 ref_images = torch.stack(ref_images, dim=0)
#             else:
#                 ref_images = [clipped_recon_images[node.parent.val] for node in level_nodes]
#                 ref_images = torch.stack(ref_images, dim=0)
#             # 进行运动补偿，得到预测帧和变形帧
#             level_predictions, level_warpframes = self.motioncompensation(ref_images, level_qmv_upsamples)

#             level_images=[input_images[node.val].squeeze(0) for node in level_nodes]
#             level_images = torch.stack(level_images, dim=0)
            
#             # 计算当前层输入图像与预测帧之间的残差
#             level_residuals = level_images - level_predictions

#             # 对残差特征进行编码
#             features = self.resEncoder(level_residuals)
            
#             # 获取批次大小
#             batch_size = features.size()[0]

#             # 对特征进行编码，用于响应先验
#             z = self.respriorEncoder(features)

#             # 如果模型处于训练状态，则添加量化噪声；否则，对特征进行四舍五入
#             if self.training:
#                 compressed_z = z + quant_noise_z
#             else:
#                 compressed_z = torch.round(z)

#             # 解码响应先验，得到重构的残差标准差
#             level_recon_sigma = self.respriorDecoder(compressed_z)

#             # 重命名特征变量，用于后续处理
#             features_renorm = features

#             # 如果模型处于训练状态，则添加量化噪声；否则，对特征进行四舍五入
#             if self.training:
#                 level_compressed_features_renorm = features_renorm + quant_noise_feature
#             else:
#                 level_compressed_features_renorm = torch.round(features_renorm)

#             # 对重命名的特征进行解码，得到重构的残差
#             recon_res = self.resDecoder(level_compressed_features_renorm)
            
#             # 将重构的残差与预测帧相加，得到重构图像
#             recon_images = level_predictions + recon_res

#             # 将重构图像的像素值限制在0到1之间
#             level_clip_recon_images = recon_images.clamp(0., 1.)

#             # save 
#             for idx,node in enumerate(level_nodes):
#                 clipped_recon_images[node.val]=level_clip_recon_images[idx]
#                 warpframes[node.val]=level_warpframes[idx]
#                 predictions[node.val]=level_predictions[idx]
#                 compressed_features_renorm[node.val]=level_compressed_features_renorm[idx]
#                 recon_sigmas[node.val]=level_recon_sigma[idx]

#         subGOP_images = [input_images[i].squeeze(0) for i in range(1,len(input_images))]
#         subGOP_images = torch.stack(subGOP_images, dim=0)
#         clipped_recon_images=torch.stack(clipped_recon_images[1:],dim=0)
#         warpframes=torch.stack(warpframes[1:],dim=0)
#         predictions=torch.stack(predictions[1:],dim=0)
#         compressed_features_renorm=torch.stack(compressed_features_renorm[1:],dim=0)
#         recon_sigmas=torch.stack(recon_sigmas[1:],dim=0)

#         # distortion
#         mse_loss = torch.mean((clipped_recon_images - subGOP_images).pow(2))
#         # psnr = tf.cond(
#         #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
#         #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))
#         warploss = torch.mean((warpframes - subGOP_images).pow(2))
#         interloss = torch.mean((predictions - subGOP_images).pow(2))
        

#         # bit per pixel
#         def feature_probs_based_sigma(feature, sigma):
            
#             def getrealbitsg(x, gaussian):
#                 # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
                
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

#                 return sym_out - self.mxrange, real_bits


#             mu = torch.zeros_like(sigma)
#             sigma = sigma.clamp(1e-5, 1e10)
#             gaussian = torch.distributions.laplace.Laplace(mu, sigma)
#             probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbitsg(feature, gaussian)
#                 total_bits = real_bits

#             return total_bits, probs

#         def iclr18_estrate_bits_z(z):
            
#             def getrealbits(x):
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

#                 return sym_out - self.mxrange, real_bits

#             prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbits(z)
#                 total_bits = real_bits

#             return total_bits, prob

#         def iclr18_estrate_bits_mv(mv):

#             def getrealbits(x):
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
#                 return sym_out - self.mxrange, real_bits

#             prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbits(mv)
#                 total_bits = real_bits

#             return total_bits, prob


#         total_bits_feature, _ = feature_probs_based_sigma(compressed_features_renorm, recon_sigmas)
#         # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
#         total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
#         total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

#         im_shape = subGOP_images.size()

#         bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
#         bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
#         bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
#         bpp = bpp_feature + bpp_z + bpp_mv
        
#         return clipped_recon_images, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
        

# with IFL
class LFRVideoCompressor(nn.Module):
    def __init__(self):
        super(LFRVideoCompressor, self).__init__()
        # Existing components
        self.opticFlow = Tree_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)

        # Add IFL for encoding and decoding
        self.ifl_encode = InterFrameLayer(num_frames=subGOP_F, num_channels=128)  # Adjust num_frames and num_channels as needed
        self.ifl_decode = InterFrameLayer(num_frames=subGOP_F, num_channels=128)  # IFL for decoding

    def forward(self, input_images, tree_levels, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        # Existing forward logic
        estmv = self.opticFlow(input_images, tree_levels)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = `se`lf.mvDecoder(quant_mv)

        clipped_recon_images = [0] * len(input_images)
        warpframes = [0] * len(input_images)
        predictions = [0] * len(input_images)
        compressed_features_renorm = [0] * len(input_images)
        recon_sigmas = [0] * len(input_images)

        for level_idx, level_nodes in enumerate(tree_levels):
            if level_idx == 0:
                continue
            indices = [node.val - 1 for node in level_nodes]
            level_qmv_upsamples = quant_mv_upsample[indices]

            if level_idx == 1:
                ref_images = [input_images[node.parent.val].squeeze(0) for node in level_nodes]
                ref_images = torch.stack(ref_images, dim=0)
            else:
                ref_images = [clipped_recon_images[node.parent.val] for node in level_nodes]
                ref_images = torch.stack(ref_images, dim=0)

            level_predictions, level_warpframes = self.motioncompensation(ref_images, level_qmv_upsamples)
            level_images = [input_images[node.val].squeeze(0) for node in level_nodes]
            level_images = torch.stack(level_images, dim=0)
            level_residuals = level_images - level_predictions

            # Encode residuals
            features = self.resEncoder(level_residuals)

            # Apply IFL during encoding
            features = self.ifl_encode(features)

            # Continue with existing logic
            z = self.respriorEncoder(features)
            if self.training:
                compressed_z = z + quant_noise_z
            else:
                compressed_z = torch.round(z)
            level_recon_sigma = self.respriorDecoder(compressed_z)
            features_renorm = features

            if self.training:
                level_compressed_features_renorm = features_renorm + quant_noise_feature
            else:
                level_compressed_features_renorm = torch.round(features_renorm)

            # Decode residuals
            recon_res = self.resDecoder(level_compressed_features_renorm)

            # Apply IFL during decoding
            recon_res = self.ifl_decode(recon_res)

            # Reconstruct frames
            recon_images = level_predictions + recon_res
            level_clip_recon_images = recon_images.clamp(0., 1.)

            for idx, node in enumerate(level_nodes):
                clipped_recon_images[node.val] = level_clip_recon_images[idx]
                warpframes[node.val] = level_warpframes[idx]
                predictions[node.val] = level_predictions[idx]
                compressed_features_renorm[node.val] = level_compressed_features_renorm[idx]
                recon_sigmas[node.val] = level_recon_sigma[idx]

        # Existing logic for distortion and bitrate calculation
        subGOP_images = [input_images[i].squeeze(0) for i in range(1, len(input_images))]
        subGOP_images = torch.stack(subGOP_images, dim=0)
        clipped_recon_images = torch.stack(clipped_recon_images[1:], dim=0)
        warpframes = torch.stack(warpframes[1:], dim=0)
        predictions = torch.stack(predictions[1:], dim=0)
        compressed_features_renorm = torch.stack(compressed_features_renorm[1:], dim=0)
        recon_sigmas = torch.stack(recon_sigmas[1:], dim=0)

        mse_loss = torch.mean((clipped_recon_images - subGOP_images).pow(2))
        warploss = torch.mean((warpframes - subGOP_images).pow(2))
        interloss = torch.mean((predictions - subGOP_images).pow(2))

        # bit per pixel
        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob

        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob


        # Existing logic for bitrate calculation
        total_bits_feature, _ = feature_probs_based_sigma(compressed_features_renorm, recon_sigmas)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = subGOP_images.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv

        return clipped_recon_images, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp