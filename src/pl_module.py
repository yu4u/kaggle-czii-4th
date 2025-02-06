from pathlib import Path
from typing import Any
import sklearn.metrics
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV3
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .model import get_model_from_cfg, EnsembleModel
from .loss import get_loss
from .util import mixup, get_augment_policy
from .metric import get_experiment_score, particle_types, particle_to_weights


def get_patch_weight(img_size, depth):
    s = np.linspace(0, 1, img_size)
    t = np.linspace(0, 1, depth)
    a = np.minimum(s, 1 - s).reshape(1, -1, 1)
    b = np.minimum(s, 1 - s).reshape(1, 1, -1)
    c = np.minimum(t, 1 - t).reshape(-1, 1, 1)
    patch_weight = np.minimum(a, b) * c
    patch_weight = patch_weight / patch_weight.max()
    return patch_weight


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.preds = None
        self.weights = None
        self.gts = None
        self.cfg = cfg

        img_size = self.cfg.model.img_size
        img_depth = self.cfg.model.img_depth
        self.patch_weight = get_patch_weight(img_size, img_depth)

        if mode == "test":
            self.model = EnsembleModel(cfg)
        else:
            self.model = get_model_from_cfg(cfg, cfg.model.resume_path)

        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV3(
                self.model,
                decay=cfg.model.ema_decay,
                update_after_step=cfg.model.ema_update_after_step,
            )

        self.loss = get_loss(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, (exp_id, sx, sy, sz) = batch
        augment_policy = get_augment_policy(self.cfg)

        if augment_policy == "mixup":
            x, targets1, targets2, lam = mixup(x, y)
        elif augment_policy == "nothing":
            pass
        else:
            raise ValueError(f"unknown augment policy {augment_policy}")

        output = self.model(x)

        if augment_policy == "nothing":
            loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y).items()}
        else:
            loss_dict1 = self.loss(output, targets1)
            loss_dict2 = self.loss(output, targets2)
            loss_dict = {k: lam * loss_dict1[k] + (1 - lam) * loss_dict2[k] for k in loss_dict1.keys()}
            loss_dict = {k: v if k == "loss" else v.detach() for k, v in loss_dict.items()}

        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def on_validation_epoch_start(self) -> None:
        if self.cfg.model.img_depth == 16:
            self.preds = np.zeros((self.cfg.model.class_num, 184, 640, 640), dtype=np.float32)
            self.weights = np.zeros((1, 184, 640, 640), dtype=np.float32)
        elif self.cfg.model.img_depth == 32:
            self.preds = np.zeros((self.cfg.model.class_num, 192, 640, 640), dtype=np.float32)
            self.weights = np.zeros((1, 192, 640, 640), dtype=np.float32)
        else:
            raise ValueError(f"unknown img_depth {self.cfg.model.img_depth}")

    def validation_step(self, batch, batch_idx):
        x, y, (exp_id, sx, sy, sz) = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        loss_dict = self.loss(output, y)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

        img_size = self.cfg.model.img_size
        img_depth = self.cfg.model.img_depth

        if self.cfg.model.arch == "timm3d6":
            output = output[-1]

        output = output.cpu().numpy()

        for pred, x, y, z in zip(output, sx, sy, sz):
            self.preds[:, z:z + img_depth, y:y + img_size, x:x + img_size] += pred * self.patch_weight
            self.weights[:, z:z + img_depth, y:y + img_size, x:x + img_size] += self.patch_weight

    def on_validation_epoch_end(self):
        preds = self.preds
        weights = np.maximum(self.weights, 1.0)
        preds = preds / weights

        if self.cfg.model.img_depth == 16:
            pass
        elif self.cfg.model.img_depth == 32:
            preds = preds[:, 4:-4]
        else:
            raise ValueError(f"unknown img_depth {self.cfg.model.img_depth}")

        fold_id = self.cfg.data.fold_id
        scores = get_experiment_score(preds, fold_id)
        type_to_scores = dict(zip(particle_types, scores))
        self.log_dict(type_to_scores, on_epoch=True, sync_dist=True)

        total_score = 0.0
        total_weights = 0.0

        for particle_type, score in type_to_scores.items():
            weight = particle_to_weights[particle_type]
            total_score += score * weight
            total_weights += weight

        total_score = total_score / total_weights
        self.log("total_score", total_score, on_epoch=True, sync_dist=True)

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def on_predict_start(self):
        if self.cfg.model.img_depth == 16:
            self.preds = np.zeros((self.cfg.model.class_num, 184, 640, 640), dtype=np.float32)
            self.weights = np.zeros((1, 184, 640, 640), dtype=np.float32)
        elif self.cfg.model.img_depth == 32:
            self.preds = np.zeros((self.cfg.model.class_num, 192, 640, 640), dtype=np.float32)
            self.weights = np.zeros((1, 192, 640, 640), dtype=np.float32)
        else:
            raise ValueError(f"unknown img_depth {self.cfg.model.img_depth}")


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, _, (exp_id, sx, sy, sz) = batch
        img_size = self.cfg.model.img_size
        img_depth = self.cfg.model.img_depth
        output = self.model(x)

        if self.cfg.model.arch == "timm3d6":
            output = output[-1]

        output = output.cpu().numpy()

        for pred, x, y, z in zip(output, sx, sy, sz):
            self.preds[:, z:z + img_depth, y:y + img_size, x:x + img_size] += pred * self.patch_weight
            self.weights[:, z:z + img_depth, y:y + img_size, x:x + img_size] += self.patch_weight

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)
        batch_size = self.cfg.data.batch_size
        updates_per_epoch = len(self.trainer.datamodule.train_dataset) // batch_size // self.trainer.num_devices
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
                                                    warmup_lr=0, **self.cfg.scheduler,
                                                    step_on_epochs=False, updates_per_epoch=updates_per_epoch)
        lr_dict = dict(
            scheduler=scheduler,
            interval="step",
            frequency=1,  # same as default
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)
