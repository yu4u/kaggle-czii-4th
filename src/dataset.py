import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, cfg, x, y=None, mode="train", img_dir=None):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.x = x  # 184, 630, 630
        self.y = y  # 184, 630, 630, 5
        self.mode = mode
        self.indices = self.get_indices()
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)

    @staticmethod
    def get_stride(img_w, img_h, img_d, tile_size_x, tile_size_y, tile_size_z, stride_scale=0.5):
        tmp_stride_x = tile_size_x * stride_scale
        tmp_stride_y = tile_size_y * stride_scale
        tmp_stride_z = tile_size_z * stride_scale
        tile_num_x = max(round((img_w - tile_size_x) / tmp_stride_x + 1), 1)
        tile_num_y = max(round((img_h - tile_size_y) / tmp_stride_y + 1), 1)
        tile_num_z = max(round((img_d - tile_size_z) / tmp_stride_z + 1), 1)
        stride_x = (img_w - tile_size_x) // (tile_num_x - 1) if tile_num_x - 1 > 0 else 0
        stride_y = (img_h - tile_size_y) // (tile_num_y - 1) if tile_num_y - 1 > 0 else 0
        stride_z = (img_d - tile_size_z) // (tile_num_z - 1) if tile_num_z - 1 > 0 else 0
        return stride_x, stride_y, stride_z, tile_num_x, tile_num_y, tile_num_z

    def get_indices(self):
        img_size = self.cfg.model.img_size
        img_depth = self.cfg.model.img_depth
        tile_size_x = img_size
        tile_size_y = img_size
        tile_size_z = img_depth
        stride_scale = self.cfg.model.train_stride if self.mode == "train" else 0.5
        indices = []

        for i, x_i in enumerate(self.x):
            img_d, img_h, img_w = x_i.shape
            s_x, s_y, s_z, tile_x, tile_y, tile_z = self.get_stride(img_w, img_h, img_d, tile_size_x, tile_size_y,
                                                                    tile_size_z, stride_scale=stride_scale)
            for iz in range(tile_z):
                for iy in range(tile_y):
                    for ix in range(tile_x):
                        sx = ix * s_x
                        sy = iy * s_y
                        sz = iz * s_z
                        indices.append((i, sx, sy, sz))
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_size = self.cfg.model.img_size
        img_depth = self.cfg.model.img_depth
        class_num = self.cfg.model.class_num
        exp_id, sx, sy, sz = self.indices[idx]
        x = self.x[exp_id]  # d, h, w

        if self.mode == "train":
            # add random offset
            sx += np.random.randint(-img_size // 4, img_size // 4)
            sy += np.random.randint(-img_size // 4, img_size // 4)
            sz += np.random.randint(-img_depth // 4, img_depth // 4)
            sx = np.clip(sx, 0, x.shape[2] - img_size)
            sy = np.clip(sy, 0, x.shape[1] - img_size)
            sz = np.clip(sz, 0, x.shape[0] - img_depth)

        x = x[sz:sz + img_depth, sy:sy + img_size, sx:sx + img_size]
        x = x.transpose(1, 2, 0)  # h, w, d

        # y[exp_id]: 184, 630, 630, 5
        y = self.y[exp_id][sz:sz + img_depth, sy:sy + img_size, sx:sx + img_size] if self.y is not None else -1
        y = y.transpose(1, 2, 0, 3)  # h, w, d, c
        y = y.reshape(img_size, img_size, -1)

        sample = self.transforms(image=x, mask=y)
        x = sample["image"]
        x = x.unsqueeze(0)
        y = sample["mask"]
        y = y.reshape(img_depth, class_num, img_size, img_size)
        y = y.permute(1, 0, 2, 3)
        # y = (y / 255.0).float()

        if self.mode == "train" and self.cfg.model.depth_flip:
            if np.random.rand() > 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])

        return x, y, (exp_id, sx, sy, sz)


def get_train_transforms(cfg):
    return A.Compose(
        [
            # A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.CenterCrop(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.05, scale_limit=0.1, value=0,
                               rotate_limit=180, mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.model.img_size, min_width=cfg.model.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            # A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # normalize with imagenet statis
            # A.Normalize(p=1.0, mean=5.2577832e-08, std=7.199929e-06, max_pixel_value=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.3
            ),
            ToTensorV2(p=1.0, transpose_mask=True),
        ],
        p=1.0,
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            # A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.RandomScale(scale_limit=(1.0, 1.0), p=1),
            # A.PadIfNeeded(min_height=cfg.model.img_size, min_width=cfg.model.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Crop(y_max=self.cfg.data.val_img_h, x_max=self.cfg.data.val_img_w, p=1.0),
            # A.Normalize(p=1.0, mean=5.2577832e-08, std=7.199929e-06, max_pixel_value=1.0),
            ToTensorV2(p=1.0, transpose_mask=True),
        ],
        p=1.0,
    )
