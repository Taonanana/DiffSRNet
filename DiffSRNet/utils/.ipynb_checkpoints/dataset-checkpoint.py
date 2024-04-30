import os
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.matlab_resize import imresize

# SRDataSet 类用于读取并预处理数据集，其中包括高分辨率 (HR) 和低分辨率 (LR) 图像对
class SRDataSet(Dataset):
    def __init__(self, prefix='train'):
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.len = len(IndexedDataset(f'{self.data_dir}/{self.prefix}'))    # len 表示数据集的长度，即图像对的数量
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        assert hparams['data_interp'] in ['bilinear', 'bicubic']
        self.data_augmentation = hparams['data_augmentation']
        self.indexed_ds = None
        if self.prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']

    # 获取指定索引的数据项
    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    #用于从数据集中获取一个数据样本（img_hr)，将其预处理后返回，并返回img_lr、img_lr_up
    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        img_hr = item['img']
        img_hr = Image.fromarray(np.uint8(img_hr))
        img_hr = self.pre_process(img_hr)  # PIL
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / hparams['sr_scale'], method=hparams['data_interp'])  # np.uint8 [H, W, C] 将图像img_hr按照缩放因子1/hparams['sr_scale']进行降采样
        img_lr_up = imresize(img_lr / 256, hparams['sr_scale'])  # np.float [H, W, C] 首先将img_lr中的像素值除以256，将像素值的范围从0到255缩放到了0到1之间。
                                                                #接着，使用imresize函数将这个缩小后的图像按照hparams['sr_scale']的倍数进行上采样。
                                                                # Save images to a folder
        item_name = item['img']
        raw_data_dir = hparams['raw_data_dir']
        # save_dir = f'{raw_data_dir}/Img/LR_HR'  # Replace with your desired save path
        save_dir_hr = f'{raw_data_dir}/Img/LR_HR1/HR'  # Replace with your desired HR save path
        save_dir_lr = f'{raw_data_dir}/Img/LR_HR1/LR'  # Replace with your desired LR save path
        os.makedirs(save_dir_hr, exist_ok=True)
        os.makedirs(save_dir_lr, exist_ok=True)

        img_hr_path = os.path.join(save_dir_hr, f'{item_name}_hr.png')
        img_lr_path = os.path.join(save_dir_lr, f'{item_name}_lr.png')

        save_image(img_hr, img_hr_path)
        save_image(img_lr, img_lr_path)

        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        return {
            'img_hr': img_hr, 'img_lr': img_lr, 'img_lr_up': img_lr_up,
            'item_name': item['item_name']
        }

    # pre_process 方法只是简单地返回输入的图像，即没有进行任何预处理操作
    def pre_process(self, img_hr):
        return img_hr

    def __len__(self):
        return self.len
