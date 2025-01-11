
import os
import argparse
import torch
import cv2
import logging
import numpy as np
from net import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
from dataset import DataSet, UVGDataSet, UVGDataSet_Tree,TreeDataSet
from subnet.video_dispatcher import create_tree, level_order_traversal
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt
torch.backends.cudnn.enabled = True

import concurrent.futures

# base_lr = 1e-4：基础学习率。模型训练时，学习率是一个非常重要的超参数，1e-4 是一个常见的初始值。
# cur_lr = base_lr：当前学习率，初始化时等于基础学习率。它会在训练过程中根据策略进行调整（如学习率衰减等）。
# train_lambda = 2048：这个参数用于训练中的损失计算，可能与模型的复杂度或者压缩的权重有关。它控制了训练的平衡，具体的意义需要根据具体任务来理解。
# warmup_step = 0：这是预热阶段的步数，通常用于在训练的初期逐步增加学习率，以避免过大的学习率导致训练不稳定。0 表示不进行 warm-up。
# gpu_per_batch = 4：每个 GPU 每次处理的样本数（批次大小），即每个 batch 的大小。根据 GPU 的数量和显存大小来调整这个参数。
# test_step = 10000：测试步数间隔，每训练 10000 步进行一次测试。
# tot_epoch = 1000000：总的训练周期数，即训练的总轮数。通常这个值是一个较大的数字，因为模型训练可能需要很长时间。
# tot_step = 2000000：总的训练步数，通常是用来控制训练过程的总步数。可以用来定义什么时候结束训练。
# decay_interval = 1800000：学习率衰减的步数间隔，表示每 1800000 步后，学习率会衰减。
# lr_decay = 0.1：学习率衰减系数，表示每经过 decay_interval 步后，学习率会变为原来的 10%。
# bbp: bits per pixel，每像素位数的缩写，通常用来衡量图像或视频的压缩质量。

# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 1
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)
GOP=7

parser = argparse.ArgumentParser(description='LiFteR reimplement')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testuvg_tree', action='store_true')
parser.add_argument('--training_tree', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
        help = 'hyperparameter of Reid in json format')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())

def testuvg(global_step, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        total_process_time = 0  # 总时间
        cnt = 0
        
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d"% (batch_idx, len(test_loader)))
            input_images = input[0]
            ref_image = input[1]
            ref_bpp = input[2]
            ref_psnr = input[3]
            ref_msssim = input[4]
            seqlen = input_images.size()[1]
            # print("input_images.size() ",input_images.size())

            # 记录开始时间
            start_time = time.time()

            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()
            cnt += 1

            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)
                
                # 执行网络计算（操作）
                clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(inputframe, refframe)
                # 进行日志记录
                # print(f"Shape of clipped_recon_image: {clipped_recon_image.shape}")
                # print(f"Shape of refframe: {refframe.shape}")
                # print(f"Data type of clipped_recon_image: {clipped_recon_image.dtype}")
                # print(f"Data type of refframe: {refframe.dtype}")
                # Shape of clipped_recon_image: torch.Size([1, 3, 1024, 1920])
                # Shape of refframe: torch.Size([1, 3, 1024, 1920])
                # 可以将输出作为输入
                # print("Inputframe:")
                # print("Min value:", inputframe.min().item())  # 输入帧的最小值
                # print("Max value:", inputframe.max().item())  # 输入帧的最大值
                # print("Mean value:", inputframe.mean().item())  # 输入帧的均值

                # print("\nClipped Recon Image:")
                # print("Min value:", clipped_recon_image.min().item())  # 图像的最小值
                # print("Max value:", clipped_recon_image.max().item())  # 图像的最大值
                # print("Mean value:", clipped_recon_image.mean().item())  # 图像的均值

                # 记录结束时间并计算时间
                end_time = time.time()
                process_time = end_time - start_time
                total_process_time += process_time  # 累加时间

                # 计算并统计bpp, psnr, msssim等指标
                sumbpp += torch.mean(bpp).cpu().detach().numpy()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                summsssim += ms_ssim(clipped_recon_image.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()
                cnt += 1
                ref_image = clipped_recon_image
            
            # 计算平均时间
            avg_process_time = total_process_time / cnt if cnt > 0 else 0
            print(f"Average processing time per batch: {avg_process_time:.4f} seconds")

        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt

        # 打印和记录bpp, psnr, msssim的平均值
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)
        
        # 记录速度指标
        log = "Average processing time per batch: %.4lf seconds\n" % avg_process_time
        logger.info(log)

        # 使用绘图记录结果
        uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


def check_and_tensorize(image):
    # 检查image是否已经是张量
    if isinstance(image, torch.Tensor):
        # 如果是张量，则直接返回
        return image
    else:
        # 如果不是张量，则转换为张量
        return torch.tensor(image, requires_grad=False)

def testuvg_thread(input_img_idx,input_image,ref_image):
    with torch.no_grad():
        inputframe, refframe = check_and_tensorize(input_image), check_and_tensorize(ref_image)
            # 执行网络计算（操作）
        clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(inputframe, refframe)

        # 计算并统计bpp, psnr, msssim等指标
        subbpp = torch.mean(bpp).cpu().detach().numpy()
        subpsnr = torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
        submsssim = ms_ssim(clipped_recon_image.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()

        # 清理内存
        del inputframe, refframe, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv
        del input_image,ref_image
        torch.cuda.empty_cache()
        return clipped_recon_image,subbpp,subpsnr,submsssim, input_img_idx
        

def testuvg_multi_thread(global_step, GOP=7,testfull=False):
    # 使用 ThreadPoolExecutor 创建线程池
    with torch.no_grad():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
            net.eval()
            sumbpp = 0
            sumpsnr = 0
            summsssim = 0
            cnt = 0
            
            # video dispather
            root = create_tree(GOP)
            levels = level_order_traversal(root)

            for batch_idx, input in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print("testing : %d/%d"% (batch_idx, len(test_loader)))
                
                input_images = input[0]
                ref_bpp = input[1]
                ref_psnr = input[2]
                ref_msssim = input[3]
                # print("input_images.size() ",input_images.size())
                
                sumbpp += torch.mean(ref_bpp).detach().numpy()
                sumpsnr += torch.mean(ref_psnr).detach().numpy()
                summsssim += torch.mean(ref_msssim).detach().numpy()
                cnt += 1
                
                # 记录开始时间
                start_time = time.time()
                ref_map={0:input_images[:, 0, :, :, :]} # insert I frame
                for level in levels[1:]:    # skip the I frame
                    futures=[]
                    for node in level:
                        frame_idx=node.val
                        input_img=input_images[:, frame_idx, :, :, :]
                        ref_img=ref_map[node.parent.val]
                        futures.append(executor.submit(testuvg_thread, frame_idx,input_img,ref_img)) 

                    for future in concurrent.futures.as_completed(futures):
                        output = future.result()  # 获取每个任务的返回值 计算并统计bpp, psnr, msssim等指标
                        net.eval()
                        sumbpp += output[1]
                        sumpsnr += output[2]
                        summsssim += output[3]
                        cnt += 1
                        ref_map[output[4]]=output[0]    # insert to the ref map
                    
                end_time = time.time()
                process_time = end_time - start_time
                avg_time = process_time / cnt if cnt > 0 else 0
                print(f"Average processing time per frame: {avg_time:.4f} seconds")
                

            log = "global step %d : " % (global_step) + "\n"
            logger.info(log)
            sumbpp /= cnt
            sumpsnr /= cnt
            summsssim /= cnt

            # 打印和记录bpp, psnr, msssim的平均值
            log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
            logger.info(log)
            
            # 记录速度指标
            log = "Average decoding time per batch: %.4lf seconds\n" % avg_time
            logger.info(log)

            # 使用绘图记录结果
            uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


def testuvg_tree(global_step, GOP=7, testfull=False):
    """
    使用 LFRVideoCompressor 对 UVG 数据集进行树结构测试。

    参数：
        global_step (int): 当前的全局训练步数。
        GOP (int): Group of Pictures 的长度，默认为 7。
        testfull (bool): 是否进行完整测试，默认为 False。
    """
    with torch.no_grad():
        # 初始化数据加载器
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True
        )
        net.eval()  # 设置模型为评估模式

        # 初始化累积指标
        sumbpp = 0.0
        sumpsnr = 0.0
        summsssim = 0.0
        cnt = 0
        total_process_time = 0.0  # 总处理时间

        # 构建二叉树结构
        root = create_tree(GOP)
        levels = level_order_traversal(root)

        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"testing : {batch_idx}/{len(test_loader)}")

            # 获取输入数据
            input_images = input[0]  # [batch, seq_len, C, H, W]
            ref_bpp = input[1]
            ref_psnr = input[2]
            ref_msssim = input[3]

            # 检查输入数据是否符合 GOP 要求
            if input_images.size(1) != GOP:
                print(f"Skipping batch {batch_idx} due to mismatch in GOP size")
                continue

            # 累加参考指标
            sumbpp += torch.mean(ref_bpp).cpu().detach().numpy()
            sumpsnr += torch.mean(ref_psnr).cpu().detach().numpy()
            summsssim += torch.mean(ref_msssim).cpu().detach().numpy()
            cnt += 1

            # 记录开始时间
            start_time = time.time()

            # 准备输入序列
            imglist = [input_images[:, i, :, :, :] for i in range(input_images.size(1))]

            # 模型前向传播，传入 levels
            reconstructed_sequence, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(imglist, levels)

            # 计算 bpp、psnr 和 msssim
            bpp_val = torch.mean(bpp).cpu().detach().numpy()
            psnr_val = torch.mean(10 * (torch.log(1.0 / mse_loss) / np.log(10))).cpu().detach().numpy()
            msssim_val = ms_ssim(
                reconstructed_sequence.cpu().detach(),
                input_images,
                data_range=1.0,
                size_average=True
            ).numpy()

            # 累加指标
            sumbpp += bpp_val
            sumpsnr += psnr_val
            summsssim += msssim_val
            cnt += 1

            # 记录结束时间并计算处理时间
            end_time = time.time()
            process_time = end_time - start_time
            total_process_time += process_time

            # 计算每帧的平均处理时间
            seq_len = input_images.size(1)
            avg_time_per_frame = process_time / seq_len if seq_len > 0 else 0
            print(f"Average processing time per frame: {avg_time_per_frame:.4f} seconds")

        # 计算平均指标
        avg_bpp = sumbpp / cnt
        avg_psnr = sumpsnr / cnt
        avg_msssim = summsssim / cnt
        avg_time_per_batch = total_process_time / cnt if cnt > 0 else 0

        # 日志记录
        log = f"global step {global_step} : \n"
        logger.info(log)
        log = f"UVGdataset : average bpp : {avg_bpp:.6f}, average psnr : {avg_psnr:.6f}, average msssim: {avg_msssim:.6f}\n"
        logger.info(log)
        log = f"Average decoding time per batch: {avg_time_per_batch:.4f} seconds\n"
        logger.info(log)

        # 使用绘图记录结果
        uvgdrawplt([avg_bpp], [avg_psnr], [avg_msssim], global_step, testfull=testfull)



def train(epoch, global_step):

    print ("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_per_batch, pin_memory=True)
    net.train()

    global optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    suminterpsnr = 0
    sumwarppsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_mv = 0
    sumbpp_z = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        input_image, ref_image = Var(input[0]), Var(input[1])
        quant_noise_feature, quant_noise_z, quant_noise_mv = Var(input[2]), Var(input[3]), Var(input[4])
        # ta = datetime.datetime.now()
        clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv)
        
        # tb = datetime.datetime.now()
        mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
            torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv), torch.mean(bpp)
        distribution_loss = bpp
        if global_step < 500000:
            warp_weight = 0.1
        else:
            warp_weight = 0
        distortion = mse_loss + warp_weight * (warploss + interloss)
        rd_loss = train_lambda * distortion + distribution_loss
        # tc = datetime.datetime.now()
        optimizer.zero_grad()
        rd_loss.backward()
        # tf = datetime.datetime.now()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse_loss > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if warploss > 0:
                warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
            else:
                warppsnr = 100
            if interloss > 0:
                interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()
            else:
                interpsnr = 100

            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            suminterpsnr += interpsnr
            sumwarppsnr += warppsnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_feature.cpu().detach()
            sumbpp_mv += bpp_mv.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()


        if (batch_idx % print_step)== 0 and bat_cnt > 1:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
            tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('warppsnr', sumwarppsnr / cal_cnt, global_step)
            tb_logger.add_scalar('interpsnr', suminterpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_feature', sumbpp_feature / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_z', sumbpp_z / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_mv', sumbpp_mv / cal_cnt, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), sumloss / cal_cnt, cur_lr, (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : warppsnr : {:.2f} interpsnr : {:.2f} psnr : {:.2f}'.format(sumwarppsnr / cal_cnt, suminterpsnr / cal_cnt, sumpsnr / cal_cnt)
            print(log)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_feature = sumbpp_mv = sumbpp_z = sumloss = sumpsnr = suminterpsnr = sumwarppsnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


def train_tree(epoch, global_step):

    print ("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_per_batch, pin_memory=True)
    net.train()

    global optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    suminterpsnr = 0
    sumwarppsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_mv = 0
    sumbpp_z = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()

    # tree
    root=create_tree(GOP)
    tree_levels=level_order_traversal(root)

    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        input_images= [ Variable(img.cuda()) for img in input[0]]
        quant_noise_feature, quant_noise_z, quant_noise_mv = Var(input[1]), Var(input[2]), Var(input[3])

        clipped_recon_images, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(input_images,tree_levels, quant_noise_feature, quant_noise_z, quant_noise_mv)
        
        mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
            torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv), torch.mean(bpp)
        distribution_loss = bpp
        if global_step < 500000:
            warp_weight = 0.1
        else:
            warp_weight = 0
        distortion = mse_loss + warp_weight * (warploss + interloss)
        rd_loss = train_lambda * distortion + distribution_loss

        optimizer.zero_grad()
        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse_loss > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if warploss > 0:
                warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
            else:
                warppsnr = 100
            if interloss > 0:
                interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()
            else:
                interpsnr = 100

            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            suminterpsnr += interpsnr
            sumwarppsnr += warppsnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_feature.cpu().detach()
            sumbpp_mv += bpp_mv.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()


        if (batch_idx % print_step)== 0 and bat_cnt > 1:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
            tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('warppsnr', sumwarppsnr / cal_cnt, global_step)
            tb_logger.add_scalar('interpsnr', suminterpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_feature', sumbpp_feature / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_z', sumbpp_z / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_mv', sumbpp_mv / cal_cnt, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), sumloss / cal_cnt, cur_lr, (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : warppsnr : {:.2f} interpsnr : {:.2f} psnr : {:.2f}'.format(sumwarppsnr / cal_cnt, suminterpsnr / cal_cnt, sumpsnr / cal_cnt)
            print(log)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_feature = sumbpp_mv = sumbpp_z = sumloss = sumpsnr = suminterpsnr = sumwarppsnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


if __name__ == "__main__":
    args = parser.parse_args()

    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("Training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    if args.training_tree or args.testuvg_tree:
        model = LFRVideoCompressor()
    else:
        model = VideoCompressor()

    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    # save_model(model, 0)
    global train_dataset, test_dataset
    if args.testuvg:
        test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
        print('testing UVG')
        testuvg(0, testfull=True)
        exit(0)
    
    if args.testuvg_tree:
        test_dataset = UVGDataSet_Tree(refdir=ref_i_dir, testfull=True, gop=GOP)
        print('testing UVG Tree')
        testuvg_tree(0, testfull=True)
        exit(0)

    if args.training_tree:
        train_dataset = TreeDataSet("data/vimeo_septuplet/test_tree.txt")
    else:
        train_dataset = DataSet("data/vimeo_septuplet/test.txt")

    tb_logger = SummaryWriter('./events')
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))# * gpu_num))
    for epoch in range(stepoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step)
            break
        if args.training_tree:
            global_step=train_tree(epoch,global_step)
        else:
            global_step = train(epoch, global_step)
        save_model(model, global_step)
