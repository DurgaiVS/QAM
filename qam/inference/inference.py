import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import FBetaScore, Precision, Recall

from ..constants import PAD_ID
from ..modules.encoder import ConfEncoderWithClassificationHeads
from ..training.data.data_sample import QAMDataBatch, QAMDataSample
from .utils import QAMInferenceResultsWriter, defaultdict, QAMOverallDataMetric


class QAMDatasetPredictor(pl.LightningModule):
    def __init__(
        self,
        model: ConfEncoderWithClassificationHeads,
        base_dir: str,
        output_dir: str,
        f_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.f_beta = f_beta
        self.dataset_wise_score = defaultdict(self.default_fn)
        self.categories = ["punct", "capit"]
        self.metrics_name = ["f1score", "precision", "recall"]
        self.writer = QAMInferenceResultsWriter(
            self.output_dir,
            self.categories,
            self.metrics_name,
        )

    def default_fn(self) -> dict:
        val = {}
        # macro: computes score for individual classes and then return avg of them
        # none: computes score for individual classes and return as a list
        # f_beta: can give manual weightage to precision (and/or) recall
        for catg in self.categories:
            val[f"{catg}_f1score"] = FBetaScore(
                task="multiclass",
                beta=self.f_beta,
                num_classes=getattr(self.model, f"{catg}_classes_count"),
                average="none",
            ).to(self.device)

            val[f"{catg}_precision"] = Precision(
                task="multiclass",
                num_classes=getattr(self.model, f"{catg}_classes_count"),
                average="none",
            ).to(self.device)

            val[f"{catg}_recall"] = Recall(
                task="multiclass",
                num_classes=getattr(self.model, f"{catg}_classes_count"),
                average="none",
            ).to(self.device)

        return val

    def predict_step(
        self, batch: QAMDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch.tokens
        att_mask = batch.attention_mask

        punct_logits, capit_logits = self.model(input_ids, att_mask)
        return punct_logits, capit_logits

    def on_predict_batch_end(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        batch: QAMDataBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        punct_logits, capit_logits = outputs

        punct_preds = torch.argmax(punct_logits, dim=2)
        capit_preds = torch.argmax(capit_logits, dim=2)

        for (
            sample,
            punct_pred,
            capit_pred,
        ) in zip(
            batch,
            punct_preds,
            capit_preds,
        ):
            sample: QAMDataSample
            mask = sample.punct_labels != PAD_ID
            punct_label = sample.punct_labels[mask].unsqueeze(0)
            punct_pred = punct_pred[mask].unsqueeze(0)
            capit_label = sample.capit_labels[mask].unsqueeze(0)
            capit_pred = capit_pred[mask].unsqueeze(0)

            ll = locals()
            gl = globals()
            score_dict = {}
            ds_name = sample.dataset_name
            for metric in self.metrics_name:
                acc = 0
                for catg in self.categories:
                    tmp = self.dataset_wise_score[ds_name][f"{catg}_{metric}"](
                        ll[f"{catg}_pred"], ll[f"{catg}_label"]
                    )
                    cls = gl.get(f"{catg.capitalize()}Label")
                    for id, val in enumerate(tmp):
                        score_dict[f"{catg}_{cls(id).name.lower()}_{metric}"] = val.cpu().item()

                    score_dict[f"{catg}_{metric}"] = tmp.mean().cpu().item()
                    acc += tmp.mean()

                acc = torch.true_divide(acc, len(self.categories))
                score_dict[f"{metric}"] = acc.cpu().item()

            self.writer.write_sample_wise_stats(
                sample, punct_pred.squeeze(0).cpu(), capit_pred.squeeze(0).cpu(), score_dict
            )

    def on_predict_epoch_end(self) -> None:
        self.writer.write_overall_stats(self.dataset_wise_score)

    @rank_zero_only
    def get_metrics(self) -> QAMOverallDataMetric:
        return QAMOverallDataMetric.read_from(self.writer.overall_summary_file)
