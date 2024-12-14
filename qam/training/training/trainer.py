from typing import List

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchaudio.models import Conformer
from torchmetrics import F1Score, Precision, Recall

from ...utils import Classifier, QAMDataBatch


class QAMTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
        lr_schs: List[torch.optim.lr_scheduler._LRScheduler],
        metric: str,
        f1score: F1Score,
        precision: Precision,
        recall: Recall,
        scheduler_interval: int = 1,
        scheduler_frequency: str = "epoch",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.lr_schs = lr_schs
        self.metric = metric
        self.f1score = f1score
        self.precision = precision
        self.recall = recall
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        lr_schs = []
        for each in self.lr_schs:
            lr_schs.append(
                {
                    "scheduler": each,
                    "monitor": self.metric,
                    "interval": self.scheduler_interval,
                    "frequency": self.scheduler_frequency,
                }
            )
        return [self.optimizer], lr_schs

    def training_step(self, batch: QAMDataBatch, batch_idx: int):
        frame_batch = batch.frames
        label_batch = batch.labels

        preds = self.model(frame_batch)
        loss = self.loss_fn(preds, label_batch)
        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch: QAMDataBatch, batch_idx: int):
        frame_batch = batch.frames
        label_batch = batch.labels

        preds = self.model(frame_batch)
        loss = self.loss_fn(preds, label_batch)

        logs = {
            "val/loss": loss,
            "val/f1": self.f1score(preds, label_batch),
            "val/precision": self.precision(preds, label_batch),
            "val/recall": self.recall(preds, label_batch),
        }
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: QAMDataBatch, batch_idx: int):
        frame_batch = batch.frames
        label_batch = batch.labels

        preds = self.model(frame_batch)
        loss = self.loss_fn(preds, label_batch)

        logs = {
            "val/loss": loss,
            "val/f1": self.f1score(preds, label_batch),
            "val/precision": self.precision(preds, label_batch),
            "val/recall": self.recall(preds, label_batch),
        }
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    @classmethod
    def from_cfg(
        cls, cfg: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):

        f1score = F1Score(
            task="multiclass",
            num_classes=len(Classifier.__members__),
            average="none",
        )
        precision = Precision(
            task="multiclass", num_classes=len(Classifier.__members__), average="none"
        )
        recall = Recall(
            task="multiclass", num_classes=len(Classifier.__members__), average="none"
        )

        lr_schs = [
            hydra.utils.instantiate(cfg.step_based, optimizer=optimizer),
            hydra.utils.instantiate(cfg.metric_based, optimizer=optimizer),
        ]

        return cls(
            model=model,
            optimizer=optimizer,
            loss=hydra.utils.instantiate(cfg.loss),
            lr_schs=lr_schs,
            metric=cfg.metric,
            f1score=f1score,
            precision=precision,
            recall=recall,
            scheduler_interval=cfg.scheduler_interval,
            scheduler_frequency=cfg.scheduler_frequency,
        )
