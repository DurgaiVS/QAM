from typing import List

import hydra
import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import ModelCheckpoint

from ..constants import MAX_SEQ_LEN
from ..modules.model.conformer_encoder import ConfEncoderWithClassificationHeads
from ..utils import Classifier, get_cfg
from .data import QAMDataModule, reshard_if_needed
from .evaluation import QAMPredictor
from .training import QAMTrainer
from .utils import get_best_model_path, wrap_up_predictor, wrap_up_trainer

app = typer.Typer()


@app.command()
def ssl_pretrain(overrides: List[str] = []):

    cfg = get_cfg("train", overrides, "ssl_pretraining")
    pl.seed_everything(cfg.experiment.seed)

    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()

    model_checkpoints: List[ModelCheckpoint] = []
    meta = reshard_if_needed(
        cfg.symbols, cfg.trainer.devices, cfg.data.num_workers, cfg.data.batch_size
    )
    cfg.trainer.val_check_interval = int(
        meta.train_steps_count * cfg.trainer.val_check_interval
    )
    cfg.trainer.limit_train_batches = meta.train_steps_count

    callbacks = [
        hydra.utils.instantiate(callback) for callback in cfg.pl_trainer_callbacks
    ]
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            model_checkpoints.append(cb)

    model = ConfEncoderWithClassificationHeads(**cfg.model)

    optim = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)

    cfg.pl_model.loss.num_classes = len(Classifier)
    pl_model = QAMTrainer.from_cfg(cfg.pl_model, model, optim)
    data_module = QAMDataModule(**cfg.data)

    trainer.fit(pl_model, datamodule=data_module)

    if trainer.is_global_zero:
        wrap_up_trainer(
            pl_model, cfg, get_best_model_path(model_checkpoints), MAX_SEQ_LEN
        )


@app.command
def evaluate(overrides: List[str] = []):

    cfg = get_cfg("benchmark", overrides, "evaluation")

    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()

    meta = reshard_if_needed(
        cfg.symbols,
        cfg.trainer.devices,
        cfg.data.num_workers,
        cfg.data.batch_size,
        ["test"],
    )
    callbacks = [
        hydra.utils.instantiate(callback) for callback in cfg.experiment.callbacks
    ]
    data_module = QAMDataModule(**cfg.data)
    predictor = QAMPredictor.from_cfg(cfg)

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    trainer.predict(predictor, datamodule=data_module)

    if trainer.is_global_zero:
        wrap_up_predictor(cfg.experiment.output_dir, cfg.symbols, None, cfg)
