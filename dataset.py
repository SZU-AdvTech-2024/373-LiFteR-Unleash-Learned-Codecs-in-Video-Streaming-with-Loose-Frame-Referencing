import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels,random_crop_images,random_flip_images

class UVGDataSet(data.Dataset):
    def __init__(self, root="data/UVG/images/", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, 'im'+str(i * 12 + 1).zfill(4)+'.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1


    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [1.2929020996093752,0.6758680826822915,0.94005859375,0.6770526529947917,0.7543700358072918,0.8640651041666668,0.6924034016927084]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.7243849283854167,0.471212158203125,0.5672164713541666,0.3604554036458334,0.550234619140625,0.5805125325520833,0.5005953776041667]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = []# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = []# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


# 每个batch是一个gop，不返回依赖帧，依赖帧由首帧和重建帧构成
class UVGDataSet_Tree(data.Dataset):
    def __init__(self, root="data/UVG/images/", filelist="data/UVG/originalv.txt", refdir='H265L26', testfull=False, gop=7):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()   #对变量 folder 执行右侧去除空白字符的操作，并将去除后的结果赋值给变量 seq
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // gop
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, 'im'+str(i * gop + 1).zfill(4)+'.png')
                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * gop + j + 1).zfill(3)+'.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1


    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [1.2929020996093752,0.6758680826822915,0.94005859375,0.6770526529947917,0.7543700358072918,0.8640651041666668,0.6924034016927084]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.7243849283854167,0.471212158203125,0.5672164713541666,0.3604554036458334,0.550234619140625,0.5805125325520833,0.5005953776041667]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = [0.3411645507812501,0.3319017740885417,0.36831477864583334,0.20547257486979167,0.40615576171875,0.40312744140625,0.36251399739583334]# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = []# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
                input_images.append(input_image[:, :h, :w])
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, self.refbpp[index], refpsnr, refmsssim



class DataSet(data.Dataset):
    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)
        print("input_image ",input_image.shape, "ref_image ", ref_image.shape)
        # input_image  torch.Size([3, 256, 256]) ref_image  torch.Size([3, 256, 256])

        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv



class TreeDataSet(data.Dataset):
    def __init__(self, path="data/vimeo_septuplet/test_tree.txt", sequences_dir="data/vimeo_septuplet/sequences/",
                 im_height=256, im_width=256, gop=7):
        self.sequences_dir = sequences_dir
        self.subfolders = self.get_subfolders(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.gop = gop
        
        # 确保 out_channel_M, out_channel_N, out_channel_mv 已在上下文中定义
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("Dataset found subfolders: ", len(self.subfolders))
    
    def get_subfolders(self, filefolderlist):
        """
        读取 test_new.txt，获取所有子文件夹的路径。
        
        :param filefolderlist: test_new.txt 文件路径
        :return: 子文件夹路径列表
        """
        with open(filefolderlist) as f:
            data = f.readlines()
        
        subfolders = [line.strip() for line in data if line.strip()]
        return subfolders
    
    def __len__(self):
        return len(self.subfolders)
    
    def __getitem__(self, index):
        subfolder_rel_path = self.subfolders[index]
        subfolder_full_path = os.path.join(self.sequences_dir, subfolder_rel_path)
        # print("subfolder_full_path", subfolder_full_path)
        
        # 读取 im1.png 到 im7.png
        images = []
        for i in range(1, self.gop + 1):
            img_path = os.path.join(subfolder_full_path, f"im{i}.png")
            # print("img_path", img_path)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = imageio.imread(img_path)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC to CHW
            images.append(torch.from_numpy(img).float())
        
        # 将7张图像堆叠成一个张量，形状为 [gop, 3, H, W]
        images = torch.stack(images, dim=0)
        
        # 随机裁剪和填充
        images = random_crop_images(images, [self.im_height, self.im_width])
        images = random_flip_images(images)
        images = torch.unbind(images, dim=0)
        # print(f"Split Images: {[img.shape for img in images]}")
        
        # 生成噪声
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5)
        quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        
        return images, quant_noise_feature, quant_noise_z, quant_noise_mv