import pytorch_lightning as pl
import torch
from torchaudio.models import Conformer
from torchmetrics import FBetaScore, Precision, Recall

from ..data.data_sample import Classifier, QAMDataBatch


class QAMTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        lr_schs: list[torch.optim.lr_scheduler._LRScheduler],
        metric: str,
        f_beta: float = 1.0,
        scheduler_interval: int = 1,
        scheduler_frequency: str = "epoch",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_schs = lr_schs
        self.metric = metric
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.f1score = FBetaScore(
            task="multiclass",
            beta=f_beta,
            num_classes=len(Classifier.__members__),
            average="macro",
        )
        self.precision = Precision(
            task="multiclass", num_classes=len(Classifier.__members__), average="macro"
        )
        self.recall = Recall(
            task="multiclass", num_classes=len(Classifier.__members__), average="macro"
        )

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
        self.log("train/loss", loss, on_epoch=True, on_step=True, sync_dist=True)

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
            sync_dist=True,
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
            sync_dist=True,
        )
        return loss
