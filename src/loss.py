import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss

def get_loss(cfg):
    if cfg.model.arch == "timm3d6":
        return MyLossForDeepSupervision(cfg)
    else:
        return MyLoss(cfg)


class MyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.loss.name == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif cfg.loss.name == "focal":
            self.loss = FocalLoss(mode="multilabel")
        elif cfg.loss.name == "dice":
            self.loss = DiceLoss(mode="multilabel")
        else:
            raise NotImplementedError(f"loss {cfg.loss.name} not implemented")

    def forward(self, y_pred, y_true):
        return_dict = dict()
        loss = self.loss(y_pred, y_true)
        neg_weight = self.cfg.loss.neg_weight
        loss = (loss * (y_true + neg_weight)).mean()
        return_dict["loss"] = loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
