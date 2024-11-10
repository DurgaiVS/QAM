import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import ConfusionMatrix, StatScores

from ..modules.encoder import ConfEncoderWithClassificationHeads
from ..utils import Classifier, QAMDataBatch, QAMDataSample, defaultdict
from .utils import QAMInferenceResultsWriter, QAMOverallDataMetric


# TODO: Update the Dataset Predictor and Results writer to up-to-date.
class QAMDatasetPredictor(pl.LightningModule):
    def __init__(
        self,
        model: ConfEncoderWithClassificationHeads,
        output_dir: str,
        f_beta: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        # f_beta: can give manual weightage to precision (and/or) recall
        self.f_beta = f_beta
        self.labels_count = len(Classifier.__members__)
        # macro: computes score for individual classes and then return avg of them
        # none: computes score for individual classes and return as a list
        self.symbolwise_statscore = defaultdict(
            lambda: StatScores(
                task="multiclass",
                num_classes=self.labels_count,
                average="none",
            ).to(self.device)
        )

        self.symbolwise_confmat = defaultdict(lambda: 0)

        self.symbolwise_labelweight = defaultdict(lambda: defaultdict(lambda: 0))

        self.metrics_name = ["f1score", "precision", "recall"]
        self.writer = QAMInferenceResultsWriter(
            self.output_dir,
            self.metrics_name,
            [self.f1score, self.precision, self.recall],
        )

    def f1score(
        self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
    ) -> torch.Tensor:
        return ((1 + (self.f_beta**2)) * tp) / (
            ((1 + (self.f_beta**2)) * tp) + fp + ((self.f_beta**2) * fn)
        )

    def precision(
        self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
    ) -> torch.Tensor:
        return tp / (tp + fp)

    def recall(
        self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
    ) -> torch.Tensor:
        return tp / (tp + fn)

    def on_predict_start(self):
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=self.labels_count,
        ).to(self.device)

    def on_predict_epoch_end(self) -> None:
        self.writer.write_overall_stats(
            self.symbolwise_statscore,
            self.symbolwise_confmat,
            self.symbolwise_labelweight,
        )

    @rank_zero_only
    def get_metrics(self) -> QAMOverallDataMetric:
        return QAMOverallDataMetric.read_from(self.writer.overall_summary_file)

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
            # mask = sample.punct_labels != PAD_ID
            # punct_label = sample.punct_labels[mask].unsqueeze(0)
            # punct_pred = punct_pred[mask].unsqueeze(0)
            # capit_label = sample.capit_labels[mask].unsqueeze(0)
            # capit_pred = capit_pred[mask].unsqueeze(0)

            ll = locals()
            score_dict = {}
            symbol = sample.symbol

            stat_score = self.symbolwise_statscore[symbol](
                pred.unsqueeze(0), sample.label.unsqueeze(0)
            )
            conf_mat = self.confusion_matrix(
                pred.unsqueeze(0), sample.label.unsqueeze(0)
            )

            self.symbolwise_confmat[symbol] += conf_mat
            score_dict["confmat"] = conf_mat.cpu().tolist()

            # ll[f"{catg}_unique"], unique_count = ll[f"{catg}_label"].unique(return_counts=True)
            # for id, (c_l, c_c) in enumerate(zip(ll[f"{catg}_unique"], unique_count)):
            #     if c_l.item() == gl.get(f"{catg.capitalize()}Label").NoOp.value:
            #         unique_count[id] = 0
            #         # since the tensors are passing reference, this will make the
            #         # `c_c` value `0`

            #     self.dataset_wise_labelweight[ds_name][f"{catg}_weights"][c_l.item()] += c_c

            # ll[f"{catg}_unique_count"] = unique_count / unique_count.sum()

            for metric in self.metrics_name:  # f1, prec, recall
                metric_vals = []
                for id, (tp, fp, _, fn, _) in enumerate(stat_score):  # labels per catg
                    val = getattr(self, metric)(tp, fp, fn)
                    score_dict[f"{Classifier(id).name.lower()}_{metric}"] = (
                        val.cpu().item()
                    )

                    if torch.isnan(val):
                        metric_vals.append(torch.zeros(1, dtype=val.dtype))
                    else:
                        metric_vals.append(val)

                score_dict[metric] = torch.stack(metric_vals).mean().cpu().item()

            self.writer.write_sample_wise_stats(sample, pred.cpu(), score_dict)

    def predict_step(
        self, batch: QAMDataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "QAMDatasetPredictor":
        # to return PT model name (from Huggingface), max_word_per_group, p&c labels count
        raise NotImplementedError
