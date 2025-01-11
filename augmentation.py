import random
import torch
import torch.nn.functional as F

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])

def random_flip(images, labels):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1


    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])


    return images, labels

def random_crop_images(images, size):
    """
    随机裁剪多个视频帧到目标大小。
    
    :param images: 张量，形状为 [gop, 3, H, W]
    :param size: 目标大小 [height, width]
    :return: 裁剪后的图像张量，形状为 [gop, 3, height, width]
    """
    _, _, H, W = images.shape
    th, tw = size

    # 如果图像尺寸小于目标尺寸，进行填充
    if H < th or W < tw:
        pad_h = max(th - H, 0)
        pad_w = max(tw - W, 0)
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        images = F.pad(images, padding, "constant", 0)
        H, W = images.shape[2], images.shape[3]
    
    # 随机裁剪
    i = random.randint(0, H - th)
    j = random.randint(0, W - tw)
    images = images[:, :, i:i + th, j:j + tw]
    return images

def random_flip_images(images):
    """
    随机水平和垂直翻转多个视频帧。
    
    :param images: 张量，形状为 [gop, 3, H, W]
    :return: 翻转后的图像张量，形状为 [gop, 3, H, W]
    """
    # 随机水平翻转
    if random.random() > 0.5:
        images = torch.flip(images, dims=[3])  # 水平翻转
    
    # 随机垂直翻转
    if random.random() > 0.5:
        images = torch.flip(images, dims=[2])  # 垂直翻转
    
    return images
