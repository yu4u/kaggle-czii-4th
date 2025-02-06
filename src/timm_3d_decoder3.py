import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import FeatureListNet

from .util import SlidingSlicer


class Conv3dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super().__init__()
        if use_batchnorm:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        i,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.k = 1 if i == 0 else 2

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(self.k, 2, 2), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, i, **kwargs)
            for i, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels))
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


def channel_to_spatial_3d(input_tensor, upscale_factor=2):
    N, C, D, H, W = input_tensor.shape
    factor_cubed = upscale_factor ** 3
    new_C = C // factor_cubed
    output_tensor = input_tensor.view(
        N, new_C, upscale_factor, upscale_factor, upscale_factor, D, H, W
    )
    output_tensor = output_tensor.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    output_tensor = output_tensor.view(
        N, new_C, D * upscale_factor, H * upscale_factor, W * upscale_factor
    )

    return output_tensor


class Timm3DDecoder3(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pretrained = True if cfg.model.resume_path is None else False

        if cfg.model.in_channels > 1:
            self.stem = SlidingSlicer(slice_size=cfg.model.in_channels)

        self.out_ch = cfg.model.class_num
        model = timm.create_model(
            cfg.model.backbone,
            in_chans=cfg.model.in_channels,
            pretrained=pretrained,
            drop_path_rate=cfg.model.drop_path_rate,
        )
        out_channels = [fi["num_chs"] for fi in model.feature_info]

        try:
            self.backbone = FeatureListNet(model, out_indices=tuple(range(len(out_channels))), flatten_sequential=True)
        except AssertionError:
            self.backbone = FeatureListNet(model, out_indices=tuple(range(len(out_channels))), flatten_sequential=False)

        self.backbone.out_channels = [cfg.model.in_channels] + out_channels

        self.decoder = UnetDecoder(
            encoder_channels=self.backbone.out_channels,
            decoder_channels=(1024, 768, 256),
            n_blocks=3,
            use_batchnorm=True,
        )

        k = 3
        conv3ds = [
            torch.nn.Sequential(
                Conv3dReLU(ch, ch, k, k // 2, use_batchnorm=True),
                Conv3dReLU(ch, ch, k, k // 2, use_batchnorm=True)
            )
            for ch in self.backbone.out_channels[1:]
        ]
        self.conv3ds = torch.nn.ModuleList(conv3ds)
        self.segmentation_head = nn.Conv3d(256, 64 * self.out_ch, 1, padding=0)

    def _to2d(self, conv3d_block: torch.nn.Module, feature: torch.Tensor, b) -> torch.Tensor:
        total_batch, ch, H, W = feature.shape  # b * d, ch, H, W
        feat_3d = feature.reshape(b, total_batch // b, ch, H, W).transpose(1, 2)
        feat_3d = conv3d_block(feat_3d)  # b, ch, d, H, W
        return feat_3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape

        if self.cfg.model.in_channels > 1:
            x = self.stem(x)  # b, d, c, h, w

        x = x.reshape(b * d, self.cfg.model.in_channels, h, w)  # b * d, c, h, w
        features = [x]
        pooled_cnt = 0

        for i, (name, module) in enumerate(self.backbone.items()):
            x = module(x)

            if name in self.backbone.return_layers:
                total_batch, ch, h, w = x.shape
                x = x.reshape(b, total_batch // b, ch, h, w).transpose(1, 2)  # b, ch, d, h, w

                if pooled_cnt >= 4:
                    k = 1
                elif self.cfg.model.img_size // w > self.cfg.model.img_depth * 2 // (total_batch // b):
                    k = 4
                    pooled_cnt += 2
                else:
                    k = 2
                    pooled_cnt += 1

                x = F.avg_pool3d(x, kernel_size=(k, 1, 1), stride=(k, 1, 1), padding=0)  # b, ch, d // 2, h, w
                x = x.transpose(1, 2).reshape(total_batch // k, ch, h, w)  # b * d // 2, ch, h, w
                features.append(x)


        features[1:] = [self._to2d(conv3d, feature, b) for conv3d, feature in zip(self.conv3ds, features[1:])]
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        masks = channel_to_spatial_3d(masks, upscale_factor=4)
        return masks

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.encoder.model.set_grad_checkpointing(enable)
