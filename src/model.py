from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.timm_3d_decoder3 import Timm3DDecoder3


def get_model_from_cfg(cfg, resume_path=None):
    if cfg.model.arch == "timm3d3":
        model = Timm3DDecoder3(cfg)
    else:
        raise NotImplementedError

    if resume_path:
        print(f"loading model from {str(resume_path)}")
        checkpoint = torch.load(str(resume_path), map_location="cpu")

        if np.any([k.startswith("model_ema.") for k in checkpoint["state_dict"].keys()]):
            print(f"loading from model_ema")
            state_dict = {k[17:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model_ema.")}
        else:
            state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}

        model.load_state_dict(state_dict, strict=True)

    return model


class EnsembleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if Path(cfg.model.resume_path).is_dir():
            resume_paths = Path(cfg.model.resume_path).rglob("*.ckpt")
        else:
            resume_paths = [Path(cfg.model.resume_path)]

        self.models = nn.ModuleList()

        for resume_path in resume_paths:
            model = get_model_from_cfg(cfg, resume_path)
            self.models.append(model)

    def __call__(self, x):
        outputs = [model(x) for model in self.models]
        x = torch.mean(torch.stack(outputs), dim=0)
        return x
