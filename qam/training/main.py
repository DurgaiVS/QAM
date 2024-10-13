from typing import List

import hydra
import pytorch_lightning as pl
import torch
import typer

from ..modules.encoder import ConfEncoderWithClassificationHeads
from ..modules.focal_loss import FocalLoss
from ..utils import get_cfg
from .data.data_module import NCEDataModule
from .train.trainer import QAMTrainer

app = typer.Typer()


@app.command()
def pretrain(overrides: List[str] = []):

    cfg = get_cfg("train", overrides, "training")
    pl.seed_everything(cfg.experiment.seed)

    model = ConfEncoderWithClassificationHeads(**cfg.model)

    optim = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    lr_schs = [
        hydra.utils.instantiate(lr_sch, optimizer=optim) for lr_sch in cfg.lr_schedulers
    ]
    callbacks = [
        hydra.utils.instantiate(callback) for callback in cfg.pl_trainer_callbacks
    ]

    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    pl_model = QAMTrainer(
        model=model,
        optimizer=optim,
        lr_schs=lr_schs,
        loss_fn=FocalLoss(**cfg.loss),
        **cfg.pl_model
    )
    data_module = NCEDataModule(**cfg.data)

    trainer.fit(pl_model, datamodule=data_module)
