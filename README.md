# Downsampled ImageNet Path Fixer

The official ImageNet website (https://image-net.org/) provides a downsampled ImageNet (resolution is reduced to 8x8, 16x16, 32x32, or 64x64) for faster training.<br>
However, using Downsampled ImageNet in PyTorch is not inconvenient since its file structure is different from original ImageNet.

This repo provides a script that converts the Downsampled ImageNet provided in https://image-net.org/ to the structure for `ImageFolder` in PyTorch.

### Step 1: Download Downsampled ImageNet

Download Downsampled ImageNet, and unzip `*.zip` files.<br>
After upzip the downloaded files, we can obtain files as follows:

```
train_data_batch_1   train_data_batch_2  train_data_batch_4  train_data_batch_6  train_data_batch_8  val_data
train_data_batch_10  train_data_batch_3  train_data_batch_5  train_data_batch_7  train_data_batch_9
```

### Step 2: Run Script

To run script, we have to install depencendies:

```
$ pip install -r requirements.txt
```

Then, we run `fix_pathes.py` with params.

```
$ python fix_pathes.py -d path/to/downloaded_data -o path/to/output
```

### Procedures in overall

```bash
# Download files
$ wget <imagenet32_train_download_path>
$ wget <imagenet32_val_download_path>

# Create dir and unzip files to the dir
$ mkdir tmp
$ unzip Imagenet32_train.zip -d tmp
$ unzip Imagenet32_val.zip -d tmp

# Intall depencendies and run script
$ pip install -r requirements.txt
$ python fix_pathes.py -d tmp -o imagenet_32_32

# Check results
$ ls imagenet_32_32
$ ls imagenet_32_32/val

# Remove old files
$ rm Imagenet32_train.zip Imagenet32_val.zip
$ rm -tf tmp
```

## Reference

- https://github.com/PatrykChrabaszcz/Imagenet32_Scripts
- https://arxiv.org/abs/1707.08819

