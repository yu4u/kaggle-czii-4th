from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import MyDataset


class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.input_root = Path(__file__).parents[1].joinpath("input")
        self.output_root = Path(__file__).parents[1].joinpath("output")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or (stage == "predict" and self.cfg.test.mode == "val"):
            img_dir = self.output_root.joinpath("train_imgs")
            train_imgs = []
            train_masks = []
            val_imgs = []
            val_masks = []

            for i, npy_path in enumerate(sorted(img_dir.glob("*.npy"))):
                img = np.load(npy_path)

                if self.cfg.model.normalize_patch:
                    img = (img - img.mean()) / img.std()
                else:
                    img = (img - 5.2577832e-08) / 7.199929e-06

                if self.cfg.model.img_depth == 16:
                    img = np.pad(img, ((0, 0), (0, 10), (0, 10)), mode="constant")
                    mask = np.load(str(npy_path).replace("train_imgs", "train_masks"))
                    mask = np.pad(mask, ((0, 0), (0, 10), (0, 10), (0, 0)), mode="constant")
                elif self.cfg.model.img_depth == 32:
                    img = np.pad(img, ((4, 4), (0, 10), (0, 10)), mode="constant")
                    mask = np.load(str(npy_path).replace("train_imgs", "train_masks"))
                    mask = np.pad(mask, ((4, 4), (0, 10), (0, 10), (0, 0)), mode="constant")
                else:
                    raise ValueError(f"unknown img depth {self.cfg.model.img_depth}")

                mask = (mask / 255.0).astype(np.float32)

                if i == self.cfg.data.fold_id:
                    val_imgs.append(img)
                    val_masks.append(mask)
                else:
                    train_imgs.append(img)
                    train_masks.append(mask)

            self.train_dataset = MyDataset(self.cfg, train_imgs, train_masks, "train")
            self.val_dataset = MyDataset(self.cfg, val_imgs, val_masks, "val")
            self.test_dataset = MyDataset(self.cfg, val_imgs, val_masks, "test")
        else:
            raise ValueError(f"unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
