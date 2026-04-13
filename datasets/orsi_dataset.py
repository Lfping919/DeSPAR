import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

class BaseORSIDataset(Dataset):
    def __init__(self, data_root, prefix, cls_path=None):
        super().__init__()
        image_dir = os.path.join(data_root, f'{prefix}_Image')
        gt_dir = os.path.join(data_root, f'{prefix}_GT')
        depth_dir = os.path.join(data_root, f'{prefix}_Depth')

        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.gts = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(('.jpg', '.png'))])
        self.depths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.jpg', '.png', '.bmp'))])

        self._filter_files()
        
        self.cls_dict = self._get_cls_dict(cls_path) if cls_path else None

    def _get_cls_dict(self, npy_file):
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Label file not found: {npy_file}")
        data = np.load(npy_file, allow_pickle=True)
        return {item['filename']: {'label': item['label'], 'class_name': item['class_name']} for item in data}

    def _filter_files(self):
        assert len(self.images) == len(self.gts) == len(self.depths), "File counts mismatch!"
        valid_images, valid_gts, valid_depths = [], [], []
        for img_p, gt_p, dep_p in zip(self.images, self.gts, self.depths):
            with Image.open(img_p) as img, Image.open(gt_p) as gt, Image.open(dep_p) as dep:
                if img.size == gt.size == dep.size:
                    valid_images.append(img_p)
                    valid_gts.append(gt_p)
                    valid_depths.append(dep_p)
        self.images, self.gts, self.depths = valid_images, valid_gts, valid_depths

    def _read_image(self, path, mode='RGB'): # 'RGB' 'L'
        with open(path, 'rb') as f:
            return Image.open(f).convert(mode)

    def _get_label_info(self, file_name):
        img_name = file_name.split('.jpg')[0]
        if self.cls_dict and img_name in self.cls_dict:
            return self.cls_dict[img_name]['label'], self.cls_dict[img_name]['class_name']

        return -1, "unknown"


class ORSITrainDataset(BaseORSIDataset):
    def __init__(self, data_root, cls_path=None, img_size=352, apply_aug=True):
        super().__init__(data_root, prefix='train', cls_path=cls_path)
        self.img_size = img_size
        self.apply_aug = apply_aug
        
        self.resize = transforms.Resize((self.img_size, self.img_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images) * 8 if self.apply_aug else len(self.images)

    def _apply_deterministic_aug(self, img, gt, depth, aug_idx):
        if aug_idx == 0:
            pass
        elif aug_idx in [1, 2, 3]:
            angle = aug_idx * 90
            img, gt, depth = TF.rotate(img, angle), TF.rotate(gt, angle), TF.rotate(depth, angle)
        elif aug_idx == 4:
            img, gt, depth = TF.hflip(img), TF.hflip(gt), TF.hflip(depth)
        elif aug_idx in [5, 6, 7]:
            angle = (aug_idx - 4) * 90
            img, gt, depth = TF.rotate(TF.hflip(img), angle), TF.rotate(TF.hflip(gt), angle), TF.rotate(TF.hflip(depth), angle)
        return img, gt, depth

    def __getitem__(self, index):
        if self.apply_aug:
            real_img_idx = index // 8
            aug_idx = index % 8
        else:
            real_img_idx = index
            aug_idx = 0

        image = self._read_image(self.images[real_img_idx], 'RGB')
        gt = self._read_image(self.gts[real_img_idx], 'L')
        depth = self._read_image(self.depths[real_img_idx], 'L')

        image, gt, depth = self.resize(image), self.resize(gt), self.resize(depth)
        image, gt, depth = self._apply_deterministic_aug(image, gt, depth, aug_idx)

        image, gt, depth = self.to_tensor(image), self.to_tensor(gt), self.to_tensor(depth)
        image = self.normalize(image)

        file_name = os.path.basename(self.images[real_img_idx])
        label, cls_name = self._get_label_info(file_name)

        return {
            'image': image,
            'depth': depth,
            'gt': gt,
            'label': label,
            'cls_name': cls_name,
            'file_name': file_name
        }


class ORSITestDataset(BaseORSIDataset):
    def __init__(self, data_root, cls_path=None, img_size=352):
        super().__init__(data_root, prefix='test', cls_path=cls_path)
        self.img_size = img_size
        
        self.resize = transforms.Resize((self.img_size, self.img_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self._read_image(self.images[index], 'RGB')
        gt = self._read_image(self.gts[index], 'L')
        depth = self._read_image(self.depths[index], 'L')

        image = self.resize(image)
        depth = self.resize(depth)

        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        gt = self.to_tensor(gt)

        image = self.normalize(image)

        file_name = os.path.basename(self.images[index])
        label, cls_name = self._get_label_info(file_name)

        return {
            'image': image,
            'depth': depth,
            'gt': gt,                     
            'label': label,
            'cls_name': cls_name,
            'file_name': file_name
        }


def build_dataloader(data_root, cls_path=None, mode='train', apply_aug=True, img_size=352, batch_size=8, shuffle=True, num_workers=12, pin_memory=True):

    if mode == 'train':
        dataset = ORSITrainDataset(data_root=data_root, cls_path=cls_path, img_size=img_size, apply_aug=apply_aug)
    else:
        batch_size = 1
        shuffle = False
        dataset = ORSITestDataset(data_root=data_root, cls_path=cls_path, img_size=img_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader