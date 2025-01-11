# OpenLiFteR

PyTorch reimplemetation for the paper:

LiFteR: Unleash Learned Codecs in Video Streaming with Loose Frame Referencing

## Reference repository

[PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression.git)
[TimeSformer](https://github.com/facebookresearch/TimeSformer)
[pytorch-spynet](https://github.com/sniklaus/pytorch-spynet)

## Requirements

- Python==3.10
- PyTorch==2.5.1

```shell
sudo apt udpate
sudo apt-get install build-essential
sudo apt-get install libstdc++6


conda install -c conda-forge libstdcxx-ng
pip install scipy imageio torch-tb-profiler torchac scikit-image -i https://mirrors.sustech.edu.cn/pypi/web/simple
```

## Env Path

```shell
sudo find / -name "libstdc++.so.6*"
# add your lib path env var to .bashrc
sudo nano ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
source ~/.bashrc
```

## Data Preparation

### Training data

1. Download [Vimeo-90k dataset](http://toflow.csail.mit.edu/): original training + test set (82GB)

[Vimeo-90k miniset](https://www.kaggle.com/datasets/wangsally/vimeo-90k-1) (9.68GB)

2. Unzip the dataset into `./data/`.
3. Remember to put the file `test.txt` in `./data/vimeo_septuplet/` to the root of your vimeo dataset if you edit the path of vimeo.

### Test data

This method only provide P-frame compression, so we first need to generate I frames by H.265. We take UVG dataset as an example.

1. Download [UVG dataset](http://ultravideo.cs.tut.fi/#testsequences_x)(1080p/8bit/YUV/RAW) to `data/UVG/videos/`.
2. Crop Videos from 1920x1080 to 1920x1024.
   ```
   cd data/UVG/
   ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/xxxx.yuv -vf crop=1920:1024:0:0 ./videos_crop/xxxx.yuv
   ```
3. Convert YUV files to images.
   ```
   python convert.py
   ```
4. Create I frames. We need to create I frames by H.265 with $crf of 20,23,26,29.
   ```
   cd CreateI
   sh h265.sh $crf 1920 1024

   # like
   sh h265.sh 20 1920 1024
   sh h265.sh 23 1920 1024
   ```

   After finished the generating of I frames of each crf, you need to use bpps of each video in `result.txt` to fill the bpps in Class UVGdataset in `dataset.py`.

## Training

    cd examples/example
    sh cp.sh
    sh run.sh
If you want models with more λ, you can edit`config.json`

If you want to use tensorboard:

    cd examples
    sh tf.sh xxxx

## Testing

Our pretrained model with λ=2048,1024,512,256 is provided on [Google Drive](https://drive.google.com/drive/folders/1M54MPrAzaA0QVySnzUu9HZWx1bfIrTZ6?usp=sharing). You can put it to `snapshot/` and run `test.sh`:

    sh test.sh
