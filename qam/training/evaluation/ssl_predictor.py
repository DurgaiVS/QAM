import os
import tarfile
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from ...modules.model.conformer_encoder import ConfEncoderWithClassificationHeads
from ...utils import QAMDataBatch, QAMDataSample, TradeTrend, defaultdict
from ..utils import QAMMetric, QAMStats
from .utils import QAMInferenceResultsWriter


# TODO: Update the Dataset Predictor and Results writer to up-to-date.
class SSLPredictor(pl.LightningModule):
    def __init__(
        self,
        model: ConfEncoderWithClassificationHeads,
        output_dir: str,
    ):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.labels_count = len(TradeTrend.__members__)

        self.symbolwise_metric = defaultdict(
            lambda: QAMMetric(self.labels_count).to(self.device)
        )
        self.writer = QAMInferenceResultsWriter(self.output_dir)

    def on_predict_epoch_end(self) -> None:
        self.writer.write_overall_stats(self.symbolwise_metric)

    @rank_zero_only
    def get_metrics(self) -> QAMStats:
        return QAMStats.read_from(self.writer.overall_summary_file)

    def on_predict_batch_end(
        self,
        output: torch.Tensor,
        batch: QAMDataBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        preds = torch.argmax(output, dim=1)
        for sample, pred in zip(batch, preds):
            sample: QAMDataSample
            symbol = sample.symbol

            stats = self.symbolwise_metric[symbol](pred, sample.label)
            self.writer.write_sample_wise_stats(sample, pred.cpu(), stats)

    def predict_step(
        self, batch: QAMDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        preds, _ = self.model(batch.frames)
        return preds

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "SSLPredictor":
        with TemporaryDirectory() as tmpdir:
            with tarfile.open(cfg.checkpoint.path, "r:gz") as tar:
                tar.extractall(tmpdir)

            hparams = OmegaConf.load(os.path.join(tmpdir, "hparams.yaml"))

            model = ConfEncoderWithClassificationHeads(**hparams.model)
            model.load_state_dict(os.path.join(tmpdir, "weights.pt"))
            model.eval()

            return cls(model, cfg.experiment.output_dir)
