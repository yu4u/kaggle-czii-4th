import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, Callback
from omegaconf import OmegaConf

from src.datamodule import MyDataModule
from src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class BaseModelFreezeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        freeze_end_epoch = pl_module.cfg.model.freeze_end_epoch
        target_model = pl_module.model.backbone

        if current_epoch < freeze_end_epoch:
            for param in target_model.parameters():
                param.requires_grad = False

            target_model.eval()
        else:
            for param in target_model.parameters():
                param.requires_grad = True

            target_model.train()


def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    seed_everything(cfg.task.seed, workers=True)
    model = MyModel(cfg, mode="train")
    print(OmegaConf.to_yaml(cfg))
    dm = MyDataModule(cfg)
    train_args = dict(cfg.trainer)

    if cfg.wandb:
        wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, log_model=False)
        train_args["logger"] = wandb_logger

    lr_monitor = LearningRateMonitor()
    dirpath = "saved_models/" + cfg.wandb.name if cfg.wandb.name else None
    finename = f"{cfg.wandb.name}"
    save_on_train_epoch_end = True if cfg.data.train_all else None
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, monitor="total_score", save_last=False, mode="max",
                                          filename=finename + "_{epoch:03d}_{total_score:.8f}", save_weights_only=True,
                                          save_on_train_epoch_end=save_on_train_epoch_end)

    # if cfg.wandb.name:
    #    checkpoint_callback.CHECKPOINT_NAME_LAST = finename

    callbacks = [lr_monitor, checkpoint_callback]

    if cfg.model.swa:
        eta_min = cfg.opt.lr_min if cfg.opt.lr_min else cfg.opt.lr / 10
        swa_callback = StochasticWeightAveraging(swa_lrs=eta_min, annealing_strategy="linear")
        callbacks = [swa_callback] + callbacks  # follow _configure_swa_callbacks

    if cfg.model.freeze_backbone:
        callbacks.append(BaseModelFreezeCallback())

    trainer = Trainer(**train_args, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
