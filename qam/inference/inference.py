import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import FBetaScore, Precision, Recall

from ..constants import PAD_ID
from ..modules.encoder import ConfEncoderWithClassificationHeads
from ..utils import QAMDataBatch, QAMDataSample
from .utils import QAMInferenceResultsWriter, QAMOverallDataMetric, defaultdict


# TODO: Update the Dataset Predictor and Results writer to up-to-date.
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
                        score_dict[f"{catg}_{cls(id).name.lower()}_{metric}"] = (
                            val.cpu().item()
                        )

                    score_dict[f"{catg}_{metric}"] = tmp.mean().cpu().item()
                    acc += tmp.mean()

                acc = torch.true_divide(acc, len(self.categories))
                score_dict[f"{metric}"] = acc.cpu().item()

            self.writer.write_sample_wise_stats(
                sample,
                punct_pred.squeeze(0).cpu(),
                capit_pred.squeeze(0).cpu(),
                score_dict,
            )

    def on_predict_epoch_end(self) -> None:
        self.writer.write_overall_stats(self.dataset_wise_score)

    @rank_zero_only
    def get_metrics(self) -> QAMOverallDataMetric:
        return QAMOverallDataMetric.read_from(self.writer.overall_summary_file)


####################################################################################################

from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from onnxruntime import InferenceSession
from pytorch_lightning.utilities import rank_zero_only
from registrable import Registrable
from torchmetrics import ConfusionMatrix, StatScores
from zspeech.common import defaultdict
from zspeech.itn.common_utils import (
    PAD_LABEL_ID,
    CapitLabel,
    ITNDataBatch,
    ITNDataSample,
    ITNOverallDataMetric,
    PunctLabel,
)
from zspeech.itn.models import PunctuationCapitalisationModel

from .utils import ITNInferenceResultsWriter


class ITNBasePredictor(pl.LightningModule, Registrable):
    def __init__(
        self,
        model: Union[PunctuationCapitalisationModel, str, InferenceSession],
        output_dir: str,
        punct_labels: int = len(PunctLabel.__members__),
        capit_labels: int = len(CapitLabel.__members__),
        f_beta: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        # f_beta: can give manual weightage to precision (and/or) recall
        self.f_beta = f_beta
        self.punct_labels = punct_labels
        self.capit_labels = capit_labels
        # macro: computes score for individual classes and then return avg of them
        # none: computes score for individual classes and return as a list
        self.dataset_wise_statscore = defaultdict(
            lambda: {
                f"{catg}_stats": StatScores(
                    task="multiclass",
                    num_classes=getattr(self, f"{catg}_labels"),
                    average="none",
                ).to(self.device)
                for catg in self.categories
            }
        )

        self.dataset_wise_confmat = defaultdict(
            lambda: {f"{catg}_confmat": 0 for catg in self.categories}
        )

        self.dataset_wise_labelweight = defaultdict(
            lambda: {f"{catg}_weights": defaultdict(lambda: 0) for catg in self.categories}
        )

        self.categories = ["punct", "capit"]
        self.metrics_name = ["f1score", "precision", "recall"]
        self.writer = ITNInferenceResultsWriter(
            self.output_dir,
            self.categories,
            self.metrics_name,
            [self.f1score, self.precision, self.recall],
        )

        for catg in self.categories:
            setattr(
                self,
                f"{catg}_confusion_matrix",
                ConfusionMatrix(
                    task="multiclass",
                    num_classes=getattr(self, f"{catg}_labels"),
                ),
            )

    def f1score(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        return ((1 + (self.f_beta**2)) * tp) / (
            ((1 + (self.f_beta**2)) * tp) + fp + ((self.f_beta**2) * fn)
        )

    def precision(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        return tp / (tp + fp)

    def recall(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        return tp / (tp + fn)

    def on_predict_start(self):
        for catg in self.categories:
            setattr(
                self,
                f"{catg}_confusion_matrix",
                getattr(self, f"{catg}_confusion_matrix").to(self.device),
            )

    def on_predict_epoch_end(self) -> None:
        self.writer.write_overall_stats(
            self.dataset_wise_statscore, self.dataset_wise_confmat, self.dataset_wise_labelweight
        )

    @rank_zero_only
    def get_metrics(self) -> ITNOverallDataMetric:
        return ITNOverallDataMetric.read_from(self.writer.overall_summary_file)

    def on_predict_batch_end(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor],
        batch: ITNDataBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        punct_logits, capit_logits = outputs

        gl = globals()
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
            sample: ITNDataSample
            mask = sample.punct_labels != PAD_LABEL_ID
            punct_label = sample.punct_labels[mask].unsqueeze(0)
            punct_pred = punct_pred[mask].unsqueeze(0)
            capit_label = sample.capit_labels[mask].unsqueeze(0)
            capit_pred = capit_pred[mask].unsqueeze(0)

            ll = locals()
            score_dict = {}
            ds_name = sample.dataset_name

            for catg in self.categories:
                ll[f"{catg}_score"] = self.dataset_wise_statscore[ds_name][f"{catg}_stats"](
                    ll[f"{catg}_pred"], ll[f"{catg}_label"]
                )

                score = getattr(self, f"{catg}_confusion_matrix")(
                    ll[f"{catg}_pred"], ll[f"{catg}_label"]
                )
                self.dataset_wise_confmat[ds_name][f"{catg}_confmat"] += score
                score_dict[f"{catg}_confmat"] = score.cpu().tolist()

                ll[f"{catg}_unique"], unique_count = ll[f"{catg}_label"].unique(return_counts=True)
                for id, (c_l, c_c) in enumerate(zip(ll[f"{catg}_unique"], unique_count)):
                    if c_l.item() == gl.get(f"{catg.capitalize()}Label").NoOp.value:
                        unique_count[id] = 0
                        # since the tensors are passing reference, this will make the
                        # `c_c` value `0`

                    self.dataset_wise_labelweight[ds_name][f"{catg}_weights"][c_l.item()] += c_c

                ll[f"{catg}_unique_count"] = unique_count / unique_count.sum()

            for metric in self.metrics_name:  # f1, prec, recall
                metric_vals = []
                for catg in self.categories:  # punct, capit
                    cls = gl.get(f"{catg.capitalize()}Label")
                    catg_vals = []
                    for id, (tp, fp, _, fn, _) in enumerate(ll[f"{catg}_score"]):  # labels per catg
                        val = getattr(self, metric)(tp, fp, fn)
                        score_dict[f"{catg}_{cls(id).name.lower()}_{metric}"] = val.cpu().item()
                        if not torch.isnan(val):
                            pos = torch.where(ll[f"{catg}_unique"] == id)[0]
                            if len(pos):
                                catg_vals.append(val * ll[f"{catg}_unique_count"][pos])

                    val = torch.stack(catg_vals).sum()
                    score_dict[f"{catg}_{metric}"] = val.cpu().item()
                    metric_vals.append(val)

                score_dict[f"{metric}"] = torch.stack(metric_vals).mean().cpu().item()

            self.writer.write_sample_wise_stats(
                sample, punct_pred.squeeze(0).cpu(), capit_pred.squeeze(0).cpu(), score_dict
            )

    def predict_step(
        self, batch: ITNDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Tuple["ITNBasePredictor", Dict]:
        # to return PT model name (from Huggingface), max_word_per_group, p&c labels count
        raise NotImplementedError
