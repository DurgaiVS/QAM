from typing import List

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from ...utils import QAMDataBatch, TradeTrend
from ..utils import QAMMetric


class SSLPreTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
        lr_schs: List[torch.optim.lr_scheduler._LRScheduler],
        metric_name: str,
        metrics: List[QAMMetric],
        grad_acc: int = 1,
        scheduler_interval: int = 1,
        scheduler_frequency: str = "epoch",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.lr_schs = lr_schs
        self.metrics = metrics
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

    # PTL-specific methods
    # NOTE: For different modes, add this to options dict,
    #       since we need to keeps the guards unsafe, we're going this way. If not,
    #       we could've just given the mode, which would automatically add the options.
    # {
    #   'default': {},
    #   'reduce-overhead': {'triton.cudagraphs': True},
    #   'max-autotune-no-cudagraphs': {'max_autotune': True, 'coordinate_descent_tuning': True},
    #   'max-autotune': {'max_autotune': True, 'triton.cudagraphs': True, 'coordinate_descent_tuning': True}}
    # }
    @torch.compile(
        fullgraph=False,
        dynamic=False,
        options={
            # "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            # "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe,
            "guard_filter_fn": lambda x: [False for _ in x],
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
        },
    )
    def training_step(self, batch: QAMDataBatch, batch_idx: int) -> torch.Tensor:
        preds, _ = self.model(batch.frames, batch.lengths)
        loss = self.loss_fn(preds, batch.labels)
        self.log(
            "train/loss", loss, on_epoch=True, on_step=True, logger=True, prog_bar=True
        )

        return loss / self.grad_acc

    @torch.compile(
        fullgraph=False,
        dynamic=False,
        options={
            # "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            # "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe,
            "guard_filter_fn": lambda x: [False for _ in x],
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
        },
    )
    def eval_step(
        self, batch: QAMDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        preds, _ = self.model(batch.frames, batch.lengths)
        return preds

    def on_eval_batch_end(
        self,
        prefix: str,
        outputs: torch.Tensor,
        batch: QAMDataBatch,
        batch_idx: int,
        dataloader_idx=0,
    ):
        loss = self.loss_fn(outputs, batch.labels)
        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        stats = self.metrics[dataloader_idx](outputs, batch.labels)
        for s_name, s_val, is_primary in stats.walk_through():
            self.log(
                f"{prefix}/{s_name}",
                s_val,
                logger=True,
                prog_bar=is_primary,
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

        if not hasattr(self, "__tmpval"):
            self.__tmpval = (batch.symbols[0], dataloader_idx)
        return loss

    @rank_zero_only
    def on_eval_epoch_end(self):
        # Log confusion matrix to `self.logger.log_image` here...
        ds_name, dl_id = self.__tmpval
        cm = self.metrics[dl_id].confmat.compute()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm.numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=list(TradeTrend.get_labels_name()),
            yticklabels=list(TradeTrend.get_labels_name()),
        )
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"{ds_name}'s Confusion Matrix")

        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure(
            f"{ds_name}'s Confusion Matrix", fig, self.current_epoch
        )
        plt.close(fig)
        del self.__tmpval

    def on_eval_end(self):
        for metric in self.metrics:
            metric.reset()

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
    def from_cfg(cls, cfg: DictConfig):
        model = hydra.utils.instantiate(cfg.model)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        loss = hydra.utils.instantiate(cfg.loss)

        metrics = [QAMMetric(cfg.loss.num_classes) for _ in cfg.data.symbols]
        lr_schs = [
            hydra.utils.instantiate(cfg.pl_model.step_based, optimizer=optimizer),
            hydra.utils.instantiate(cfg.pl_model.metric_based, optimizer=optimizer),
        ]

        return cls(
            model=model,
            optimizer=optimizer,
            loss=loss,
            lr_schs=lr_schs,
            metric=metrics,
            metric_name=cfg.pl_model.metric_name,
            grad_acc=cfg.pl_model.grad_acc,
            scheduler_interval=cfg.pl_model.scheduler_interval,
            scheduler_frequency=cfg.pl_model.scheduler_frequency,
        )
