from typing import List

import hydra
import pytorch_lightning as pl
import torch
import typer

from ..modules.model.conformer_encoder import ConfEncoderWithClassificationHeads
from ..utils import get_cfg
from .data import QAMDataModule, reshard_if_needed
from .training import QAMTrainer

app = typer.Typer()


@app.command()
def pretrain(overrides: List[str] = []):

    cfg = get_cfg("train", overrides, "training")
    pl.seed_everything(cfg.experiment.seed)

    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()

    meta = reshard_if_needed(
        cfg.symbols_info, cfg.trainer.devices, cfg.data.num_workers, cfg.data.batch_size
    )
    cfg.trainer.val_check_interval = int(
        meta.train_steps_count * cfg.trainer.val_check_interval
    )
    cfg.trainer.limit_train_batches = meta.train_steps_count

    model = ConfEncoderWithClassificationHeads(**cfg.model)

    optim = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    callbacks = [
        hydra.utils.instantiate(callback) for callback in cfg.pl_trainer_callbacks
    ]

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    pl_model = QAMTrainer.from_cfg(cfg.pl_model, model, optim)
    data_module = QAMDataModule(**cfg.data)

    trainer.fit(pl_model, datamodule=data_module)
