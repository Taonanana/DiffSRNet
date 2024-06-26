import os
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.matlab_resize import imresize


class SRDataSet(Dataset):
    def __init__(self, prefix='train'):
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.len = len(IndexedDataset(f'{self.data_dir}/{self.prefix}'))
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        assert hparams['data_interp'] in ['bilinear', 'bicubic']
        self.data_augmentation = hparams['data_augmentation']
        self.indexed_ds = None
        if self.prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        img_hr = item['img']
        img_hr = Image.fromarray(np.uint8(img_hr))
        img_hr = self.pre_process(img_hr)  # PIL
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / hparams['sr_scale'], method=hparams['data_interp'])  # np.uint8 [H, W, C]
        img_lr_up = imresize(img_lr / 256, hparams['sr_scale'])  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        # # Save images to a folder
        # item_name = item['item_name']
        # raw_data_dir = hparams['raw_data_dir']
        # # save_dir = f'{raw_data_dir}/Img/LR_HR'  # Replace with your desired save path
        # save_dir_hr = f'{raw_data_dir}/Img/LR_HR/HR'  # Replace with your desired HR save path
        # save_dir_lr = f'{raw_data_dir}/Img/LR_HR/LR'  # Replace with your desired LR save path
        # os.makedirs(save_dir_hr, exist_ok=True)
        # os.makedirs(save_dir_lr, exist_ok=True)
        # img_hr_path = os.path.join(save_dir_hr, f'{item_name}')
        # img_lr_path = os.path.join(save_dir_lr, f'{item_name}')
        # save_image(img_hr, img_hr_path)
        # save_image(img_lr, img_lr_path)
        return {
            'img_hr': img_hr, 'img_lr': img_lr, 'img_lr_up': img_lr_up,
            'item_name': item['item_name']
        }

    def pre_process(self, img_hr):
        return img_hr

    def __len__(self):
        return self.len
