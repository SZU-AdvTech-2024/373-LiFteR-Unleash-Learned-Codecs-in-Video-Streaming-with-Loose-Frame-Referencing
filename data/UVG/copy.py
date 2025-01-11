import os
import shutil

# 定义长名称目录和短名称目录
videos_crop_dir = 'videos_crop'
images_dir = 'images'

# 遍历 videos_crop 目录
for folder_name in os.listdir(videos_crop_dir):
    folder_path = os.path.join(videos_crop_dir, folder_name)
    
    # 检查该路径是否是目录
    if os.path.isdir(folder_path):
        # 根据路径名称中的前缀，提取短名称
        parts = folder_name.split('_')
        
        # 我们假设前面几个部分是相同的，例如 "Beauty", "Bosphorus" 等
        short_name = parts[0]  # 取下划线分隔后的第一个部分作为短名称
        
        # 如果 short_name 在 images 中存在对应文件夹
        target_dir = os.path.join(images_dir, short_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # 遍历该长名称文件夹中的子文件夹并拷贝到目标目录
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            
            # 只拷贝子目录
            if os.path.isdir(subfolder_path):
                target_subfolder = os.path.join(target_dir, subfolder)
                
                # 拷贝整个子文件夹到目标目录
                shutil.copytree(subfolder_path, target_subfolder,dirs_exist_ok=True)
                print(f"Copied {subfolder_path} to {target_subfolder}")
