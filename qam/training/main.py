from typing import List

import typer

app = typer.Typer()


@app.command()
def ssl_pretrain(overrides: List[str]):
    """ """

    import hydra
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint

    from ..constants import MAX_SEQ_LEN
    from ..utils import TradeTrend, get_cfg
    from .data import QAMDataModule, reshard_if_needed
    from .training.ssl_pretrainer import SSLPreTrainer
    from .utils import get_best_model_path, wrap_up_trainer

    cfg = get_cfg("train", overrides, "ssl_pretraining")
    OmegaConf.save(cfg, f"{cfg.experiment.output_dir}/_hparams.yaml")
    pl.seed_everything(cfg.experiment.seed)

    model_checkpoints: List[ModelCheckpoint] = []
    meta = reshard_if_needed(
        cfg.data.symbols,
        cfg.trainer.devices,
        cfg.trainer.accelerator,
        cfg.data.num_workers,
        cfg.data.batch_size,
        cfg.data.src,
        cfg.data.interval,
    )
    cfg.trainer.val_check_interval = int(
        meta.train_steps_count * cfg.trainer.val_check_interval
    )
    cfg.trainer.limit_train_batches = meta.train_steps_count

    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.plt_callbacks]
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            model_checkpoints.append(cb)

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)

    cfg.pl_model.loss.num_classes = len(TradeTrend)
    data_module = QAMDataModule(**cfg.data)
    pl_model = SSLPreTrainer.from_cfg(cfg)

    trainer.fit(pl_model, datamodule=data_module)

    if trainer.is_global_zero:
        wrap_up_trainer(
            pl_model, cfg, get_best_model_path(model_checkpoints), MAX_SEQ_LEN
        )


@app.command
def ssl_evaluate(overrides: List[str]):
    """ """

    import hydra
    import pytorch_lightning as pl
    import torch

    from ..utils import get_cfg
    from .data import QAMDataModule, reshard_if_needed
    from .evaluation.ssl_predictor import SSLPredictor
    from .utils import wrap_up_predictor

    cfg = get_cfg("benchmark", overrides, "evaluation")

    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()

    reshard_if_needed(
        cfg.data.symbols,
        cfg.trainer.devices,
        cfg.data.num_workers,
        cfg.data.batch_size,
        cfg.data.src,
        cfg.data.interval,
        ["test"],
    )
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.plt_callbacks]
    data_module = QAMDataModule(**cfg.data)
    predictor = SSLPredictor.from_cfg(cfg)

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    trainer.predict(predictor, datamodule=data_module)

    if trainer.is_global_zero:
        wrap_up_predictor(cfg.experiment.output_dir, cfg.symbols, None, cfg)
