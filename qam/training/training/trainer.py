from typing import List

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from ...constants import PAD_ID
from ...utils import QAMDataBatch
from ..utils import QAMMetric


class QAMTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
        lr_schs: List[torch.optim.lr_scheduler._LRScheduler],
        metric_name: str,
        metric: QAMMetric,
        grad_acc: int = 1,
        scheduler_interval: int = 1,
        scheduler_frequency: str = "epoch",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.lr_schs = lr_schs
        self.metric = metric
        self.metric_name = metric_name

        self.grad_acc = grad_acc
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.validation_step = self.eval_step
        # self.on_validation_epoch_start = self.on_eval_epoch_start
        self.on_validation_epoch_end = self.on_eval_epoch_end
        self.on_validation_end = self.on_eval_end

        self.test_step = self.eval_step
        # self.on_test_epoch_start = self.on_eval_epoch_start
        self.on_test_epoch_end = self.on_eval_epoch_end
        self.on_test_end = self.on_eval_end

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        lr_schs = []
        for each in self.lr_schs:
            lr_schs.append(
                {
                    "scheduler": each,
                    "monitor": self.metric_name,
                    "interval": self.scheduler_interval,
                    "frequency": self.scheduler_frequency,
                }
            )
        return [self.optimizer], lr_schs

    def training_step(self, batch: QAMDataBatch, batch_idx: int) -> torch.Tensor:
        frame_batch = batch.frames
        label_batch = batch.labels

        preds, _ = self.model(frame_batch)
        loss = self.loss_fn(preds, label_batch)
        self.log(
            "train/loss", loss, on_epoch=True, on_step=True, logger=True, prog_bar=True
        )

        return loss / self.grad_acc

    def eval_step(
        self, batch: QAMDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        preds, _ = self.model(batch.frames)
        return preds

    def on_eval_batch_end(
        self,
        prefix: str,
        outputs: torch.Tensor,
        batch: QAMDataBatch,
        batch_idx: int,
        dataloader_idx=0,
    ):
        mask = batch.labels != PAD_ID
        active_labels, active_logits = batch.labels[mask], outputs[mask]

        loss = self.loss_fn(active_logits, active_labels)
        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        stats = self.metric(active_logits, active_labels)
        for s_name, s_val, is_prim in stats.walk_through():
            self.log(
                f"{prefix}/{s_name}",
                s_val,
                logger=True,
                prog_bar=is_prim,
                add_dataloader_idx=False,
            )
            # NOTE: For evaluation, seperate dataloaders for seperate symbols, so a batch will be of
            #       same symbols...
            self.log(
                f"{prefix}_{batch.symbols[0]}/{s_name}",
                s_val,
                logger=True,
                add_dataloader_idx=False,
            )

        return loss

    def on_eval_epoch_end(self):
        # Log confusion matrix to `self.logger.log_image` here...
        pass

    def on_eval_end(self):
        self.metric.reset()

    def on_validation_batch_end(
        self,
        outputs: torch.Tensor,
        batch: QAMDataBatch,
        batch_index: int,
        dataloader_idx=0,
    ) -> torch.Tensor:
        return self.on_eval_batch_end(
            "val", outputs, batch, batch_index, dataloader_idx
        )

    def on_test_batch_end(
        self,
        outputs: torch.Tensor,
        batch: QAMDataBatch,
        batch_index: int,
        dataloader_idx=0,
    ) -> torch.Tensor:
        return self.on_eval_batch_end(
            "test", outputs, batch, batch_index, dataloader_idx
        )

    @classmethod
    def from_cfg(
        cls, cfg: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):

        loss = hydra.utils.instantiate(cfg.loss)
        metric = QAMMetric(cfg.loss.num_classes)

        lr_schs = [
            hydra.utils.instantiate(cfg.step_based, optimizer=optimizer),
            hydra.utils.instantiate(cfg.metric_based, optimizer=optimizer),
        ]

        return cls(
            model=model,
            optimizer=optimizer,
            loss=loss,
            lr_schs=lr_schs,
            metric=metric,
            metric_name=cfg.metric_name,
            scheduler_interval=cfg.scheduler_interval,
            scheduler_frequency=cfg.scheduler_frequency,
        )
