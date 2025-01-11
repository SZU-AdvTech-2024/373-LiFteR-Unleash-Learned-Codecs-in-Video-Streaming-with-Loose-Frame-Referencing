import os

num = 7
video_name = ['Beauty_1920x1024_120fps_420_8bit_YUV.yuv', 'HoneyBee_1920x1024_120fps_420_8bit_YUV.yuv', 'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV.yuv',  'YachtRide_1920x1024_120fps_420_8bit_YUV.yuv', 'Bosphorus_1920x1024_120fps_420_8bit_YUV.yuv',  'Jockey_1920x1024_120fps_420_8bit_YUV.yuv', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV.yuv']
short = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'YachtRide', 'Bosphorus', 'Jockey', 'ShakeNDry']

for i in range(num):
    saveroot = 'images/' + short[i]
    savepath = 'images/' + short[i] + '/im%03d.png'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    print('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)
    os.system('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/Beauty_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/Beauty_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/HoneyBee_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/ReadySteadyGo_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/YachtRide_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/Bosphorus_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/Jockey_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/Jockey_1920x1024_120fps_420_8bit_YUV.yuv

# ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv -vf crop=1920:1024:0:0 ./videos_crop/ShakeNDry_1920x1024_120fps_420_8bit_YUV.yuv
