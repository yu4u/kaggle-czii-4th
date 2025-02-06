import numpy as np
import torch
import torch.nn as nn


class SlidingSlicer(nn.Module):
    def __init__(self, slice_size=3, stride=1):
        super(SlidingSlicer, self).__init__()

        # Create convolution layer to simulate the sliding slice operation
        self.conv = nn.Conv3d(1, slice_size, kernel_size=(slice_size, 1, 1), stride=(stride, 1, 1),
                              bias=False, padding=(slice_size // 2, 0, 0))

        # Set weights to simulate identity operation and bias to 0
        with torch.no_grad():
            self.conv.weight.data.fill_(0)
            for i in range(slice_size):
                self.conv.weight.data[i, 0, i] = 1

        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out


def mixup(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    return data, targets, shuffled_targets, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return data, targets, shuffled_targets, lam


def get_augment_policy(cfg):
    p_mixup = cfg.loss.mixup
    p_cutmix = cfg.loss.cutmix
    p_nothing = 1 - p_mixup - p_cutmix
    return np.random.choice(["nothing", "mixup", "cutmix"], p=[p_nothing, p_mixup, p_cutmix], size=1)[0]


def main():
    pass


if __name__ == '__main__':
    main()