# https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/dataloader.py
import os
import traceback

from tqdm import tqdm
import netCDF4 as nc
import xarray as xr
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from PIL import Image
from numpy import asarray
import numpy as np

def build_bin_dataset(imgs, prefix):
    binary_data_dir = hparams['binary_data_dir']
    raw_data_dir = hparams['raw_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    for img in tqdm(imgs):
        try:
            # full_path = f'{raw_data_dir}/Img/target_folder/{img}'
            # full_path = f'{raw_data_dir}/Img/ssh_10_30_110_130_NorthPacific/{img}'
            full_path = f'{raw_data_dir}/Img/ssh_moxige/{img}'
            # full_path = f'{raw_data_dir}/Img/ssh_14_24_112_122_south/{img}'
            # full_path = f'{raw_data_dir}/Img/ssh_-5_20_50_90_India/{img}'
            # full_path = f'{raw_data_dir}/Img/ssh_10_25_-120_-80_Atlantic/{img}'
            # full_path = f'{raw_data_dir}/Img/ssh_-30_-20_60_70_India1/{img}'
            # full_path = f'{raw_data_dir}/Img/truth_test/{img}'
            image = Image.open(full_path).convert('RGB')
            data = asarray(image)
            builder.add_item({'item_name': img, 'path': full_path, 'img': data})
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            print("| binarize img error: ", img)
    builder.finalize()

# def build_bin_dataset(imgs, prefix):
#     # # 将数据标准化处理
#     # mean = np.mean(data)
#     # std = np.std(data)
#     # data = (data - mean) / std

#     # # 将数据转换为float32类型
#     # data = data.astype(np.float32)
    
#     binary_data_dir = hparams['binary_data_dir']
#     raw_data_dir = hparams['raw_data_dir']
#     os.makedirs(binary_data_dir, exist_ok=True)
#     builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
#     for img in tqdm(imgs):
#         try:
#             full_path = f'{raw_data_dir}/Img/data1/{img}'
#             data=xr.open_dataset(full_path)
#             # print(data)
#             # print(data.coords)
#             # data=data['mdt'].sel(longitude=slice(105,118),latitude=slice(4,21))
#             data=data['mdt']
#             data=data.where((data.longitude >= 105) & (data.longitude <= 118) & (data.latitude >= 4) & (data.latitude <= 21), drop=True)
#             # print(data.longitude)
#             # print(data)
#             ###未处理异常值
#             data = (data - data.min()) / (data.max() - data.min()) * 255#对数据进行预处理和缩放，以适应模型的输入。例如，可以使用以下代码将数据缩放到0到255的灰度值范围内
#             data = data.astype(np.uint8) # 转换数据类型为整型
#             data = np.flip(data, axis=0) # 翻转数据，使图像的y轴方向与经纬度坐标系方向一致
#             data = np.asarray(data)
#             image = Image.fromarray(data, mode='L') # 将numpy数组转换为灰度图像
#             # data = np.array(image) # 将灰度图像存储到numpy数组中
#             # image = Image.open(full_path).convert('L')
#             # data=xr.open_dataset(full_path)
#             # data = asarray(image)
#             # image = Image.open(full_path).convert('RGB')
#             data = asarray(image)
#             builder.add_item({'item_name': img, 'path': full_path, 'data': data})
#         except KeyboardInterrupt:
#             raise
#         except:
#             traceback.print_exc()
#             print("| binarize img error: ", img)
#     builder.finalize()


if __name__ == '__main__':
    set_hparams()
    raw_data_dir = hparams['raw_data_dir']
    binary_data_dir = hparams['binary_data_dir']
    eval_partition_path = f'{raw_data_dir}/Eval/ssh_moxige.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/ssh_14_24_112_122_south.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/ssh_-30_-20_60_70_India1.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/ssh_10_25_-120_-80_Atlantic.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/ssh_10_25_-120_-80_Atlantic_02.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/list_eval_partition-Copy1.txt'
    # eval_partition_path = f'{raw_data_dir}/Eval/truth_test.txt'

    train_img_list = []
    val_img_list = []
    test_img_list = []
    with open(eval_partition_path, mode='r') as f:
        while True:
            line = f.readline().split()
            if not line: break
            if line[1] == '0':
                train_img_list.append(line[0])
            elif line[1] == '1':
                val_img_list.append(line[0])
            else:
                test_img_list.append(line[0])
    build_bin_dataset(train_img_list, 'train')
    build_bin_dataset(val_img_list, 'valid')
    build_bin_dataset(test_img_list, 'test')
