import json
import os
import tarfile
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import MulticlassStatScores

from ..training.data.utils import QAMDataSample
from ..utils import QAMFileWriter, defaultdict, find_available_filename


@dataclass
class QAMOverallDataMetric:
    labels: str
    f1_score: float
    precision: float
    recall: float

    # when inheritting from this class, if you choose
    # some attributes of your class to not to get included
    # when computing avg, then you can add those here...
    _exceptions_list: Optional[List[str]] = None

    def __add__(self, other: "QAMOverallDataMetric") -> "QAMOverallDataMetric":
        for k in vars(self).keys():
            if (k in self._exceptions_list) or (k == "_exceptions_list"):
                continue

            setattr(self, k, getattr(self, k) + getattr(other, k))

        return self

    def __truediv__(self, value: Union[int, float]) -> "QAMOverallDataMetric":
        for k in vars(self).keys():
            if (
                (k in self._exceptions_list)
                or (k == "_exceptions_list")
                or (getattr(self, k) is None)
            ):
                continue
            setattr(self, k, (getattr(self, k) / value))

        return self

    @classmethod
    def compute(cls, samples: List["QAMOverallDataMetric"]) -> "QAMOverallDataMetric":
        v = vars(samples[0])
        if samples[0]._exceptions_list:
            for k in samples[0]._exceptions_list:
                v.pop(k)

        self = cls(**v)
        for sample in samples[1:]:
            self += sample

        return self / len(samples)

    @classmethod
    def read_from(cls, filepath: str):
        with open(filepath, "r") as f:
            return cls.from_str(f.readline())

    @rank_zero_only
    def write_from_rank_zero_only(self, filepath: str):
        self.write_to(filepath)

    def write_to(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(vars(self), f, indent=4)

    def to_str(self) -> str:
        return json.dumps(vars(self))

    @classmethod
    def from_str(cls, data: str):
        return cls(**json.loads(data))


@dataclass
class QAMSampleDataMetric(QAMOverallDataMetric):
    dataset_name: str
    prediction: str
    ground_truth: str

    def __post_init__(self):
        self._exceptions_list = ["ground_truth", "dataset_name", "prediction"]


class QAMInferenceResultsWriter:
    def __init__(
        self,
        output_dir: str,
        categories: List[str],
        metrics_name: List[str],
    ) -> None:
        self.output_dir = output_dir
        self.overall_summary_file = os.path.join(
            self.output_dir, f"overall_summary.json"
        )
        self.dataset_wise_sample_stats = defaultdict(self.default_fn, True)
        self.categories = categories
        self.metrics_name = metrics_name

    def default_fn(self, key: str) -> dict:
        ds_path = os.path.join(self.output_dir, key)
        os.makedirs(ds_path, exist_ok=True)
        return {
            "sample_wise_file": QAMFileWriter(
                full_path=find_available_filename(
                    ds_path, "sample_stats", "jsonl.gz", False
                ),
                size_per_file=float("inf"),
            ),
            "overall_summary": os.path.join(ds_path, f"overall_summary.json"),
        }

    def write_sample_wise_stats(
        self,
        sample: QAMDataSample,
        punct_preds: torch.Tensor,
        capit_preds: torch.Tensor,
        score_dict: dict[str, torch.Tensor],
        transformed_text: Optional[str] = None,
    ):
        self.dataset_wise_sample_stats[sample.symbol]["sample_wise_file"].write(
            QAMSampleDataMetric(
                ground_truth=sample.ground_truth,
                dataset_name=sample.symbol,
                prediction=transformed_text
                or generate_prediction(sample, punct_preds, capit_preds),
                **score_dict,
            )
        )

    def write_overall_stats(
        self, dataset_wise_score: dict[str, dict[str, MulticlassStatScores]]
    ):

        overall = defaultdict(lambda: [])
        gl = globals()

        for dataset, metrics_dict in dataset_wise_score.items():
            tmp = {}
            self.dataset_wise_sample_stats[dataset]["sample_wise_file"].close()

            for metric in self.metrics_name:
                acc = 0
                for catg in self.categories:
                    value = metrics_dict[f"{catg}_{metric}"].compute()
                    v = value.mean()

                    cls = gl.get(f"{catg.capitalize()}Label")
                    for id, val in enumerate(value):
                        tmp[f"{catg}_{cls(id).name.lower()}_{metric}"] = (
                            val.cpu().item()
                        )
                        overall[f"{catg}_{cls(id).name.lower()}_{metric}"].append(val)

                    tmp[f"{catg}_{metric}"] = v.cpu().item()
                    overall[f"{catg}_{metric}"].append(v)

                    acc += v

                acc /= len(self.categories)

                tmp[f"{metric}"] = acc.cpu().item()
                overall[f"{metric}"].append(acc)

            QAMOverallDataMetric(**tmp).write_from_rank_zero_only(
                self.dataset_wise_sample_stats[dataset]["overall_summary"]
            )

        tmp = {}
        for metric, value in overall.items():
            tmp[metric] = torch.stack(value).mean().cpu().item()

        QAMOverallDataMetric(**tmp).write_from_rank_zero_only(self.overall_summary_file)


def wrap_up_function(
    output_dir: str,
    dataset_names: List[str],
    model_hparams: DictConfig,
    predictor_hparams: DictConfig,
):
    hparams = OmegaConf.merge(
        {"model_hparams": model_hparams}, {"predictor_hparams": predictor_hparams}
    )
    OmegaConf.save(hparams, os.path.join(output_dir, "overall_hparams.yaml"))
    with tarfile.open(os.path.join(output_dir, "results.tar.gz"), "w:gz") as tar:
        for ds_name in dataset_names:
            tar.add(os.path.join(output_dir, ds_name), arcname=ds_name)
        tar.add(
            os.path.join(output_dir, "overall_summary.json"),
            arcname="overall_summary.json",
        )
        tar.add(
            os.path.join(output_dir, "overall_hparams.yaml"),
            arcname="overall_hparams.yaml",
        )

    # for ds_name in dataset_names:
    #     shutil.rmtree(os.path.join(output_dir, ds_name), ignore_errors=True)
    # os.remove(os.path.join(output_dir, "overall_summary.json"))

################################################################################################

import csv
import os

# import shutil
import tarfile
from typing import Callable, Dict, List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import MulticlassStatScores
from zspeech.common import defaultdict, find_available_filename
from zspeech.itn.common_utils import (
    CapitLabel,
    ITNDataSample,
    ITNFileWriter,
    ITNOverallDataMetric,
    ITNSampleDataMetric,
    PunctLabel,
    modify_capit_for_word,
    modify_punct_for_word,
)


def generate_prediction(
    sample: ITNDataSample, punct_preds: torch.Tensor, capit_preds: torch.Tensor
) -> str:
    output = []
    # removing special tokens from sample.words
    # removing except first but all other subtokens for a word using mask for punct & capit preds
    for word, punct_label, capit_label in zip(
        sample.words[1:-1], punct_preds[sample.mask], capit_preds[sample.mask]
    ):
        word = modify_punct_for_word(word, punct_label)
        word = modify_capit_for_word(word, capit_label)

        output.append(word)

    return " ".join(output)


class ITNInferenceResultsWriter:
    def __init__(
        self,
        output_dir: str,
        categories: List[str],
        metrics_name: List[str],
        metrics_fn: List[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> None:
        self.output_dir = output_dir
        self.overall_summary_file = os.path.join(self.output_dir, "overall_summary.json")
        self.dataset_wise_sample_stats = defaultdict(self.default_fn, True)
        self.categories = categories
        self.metrics_name = metrics_name
        for m_name, m_fn in zip(metrics_name, metrics_fn):
            setattr(self, m_name, m_fn)

    def default_fn(self, key: str) -> Dict:
        ds_path = os.path.join(self.output_dir, key)
        os.makedirs(ds_path, exist_ok=True)
        return {
            "sample_wise_file": ITNFileWriter(
                full_path=find_available_filename(ds_path, "sample_stats", "jsonl.gz", False),
                size_per_file=float("inf"),
            ),
            "overall_summary": os.path.join(ds_path, "overall_summary.json"),
        }

    def write_sample_wise_stats(
        self,
        sample: ITNDataSample,
        punct_preds: torch.Tensor,
        capit_preds: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        self.dataset_wise_sample_stats[sample.dataset_name]["sample_wise_file"].write(
            ITNSampleDataMetric(
                ground_truth=sample.ground_truth,
                dataset_name=sample.dataset_name,
                prediction=generate_prediction(sample, punct_preds, capit_preds),
                **score_dict,
            )
        )

    def compute_stats(
        self, scores: Dict[str, torch.Tensor], weights: Dict[str, Dict[int, torch.Tensor]]
    ) -> Dict[str, Union[float, List]]:
        stats = {}
        gl = globals()

        for catg in self.categories:
            stats[f"{catg}_confmat"] = (
                (scores[f"{catg}_confmat"] / scores[f"{catg}_confmat"].sum(dim=1)[:, None])
                .cpu()
                .tolist()
            )

            _weights = torch.zeros(
                len(gl[f"{catg.capitalize()}Label"].__members__), dtype=torch.int64
            )
            for k, v in weights[f"{catg}_weights"].items():
                if k == gl.get(f"{catg.capitalize()}Label").NoOp.value:
                    continue
                _weights[k] = v

            weights[f"{catg}_weights"] = _weights / _weights.sum()

        for metric in self.metrics_name:
            metric_vals = []
            for catg in self.categories:
                cls = gl.get(f"{catg.capitalize()}Label")
                catg_vals = []
                for id, (tp, fp, _, fn, _) in enumerate(scores[f"{catg}_score"]):
                    val = getattr(self, metric)(tp, fp, fn)
                    stats[f"{catg}_{cls(id).name.lower()}_{metric}"] = val.cpu().item()
                    if not torch.isnan(val):
                        catg_vals.append(val * weights[f"{catg}_weights"][id])

                # NOTE:
                # instead of computing sum of tp, fp, fn 's, we've computed weighted avg of labelwise metric
                # values coz, summing up is messing with the scores, causing F1, Prec, Rec all to be same...
                # like, FPs == FNs,
                val = torch.stack(catg_vals).sum()
                stats[f"{catg}_{metric}"] = val.cpu().item()
                metric_vals.append(val)

            stats[f"{metric}"] = torch.stack(metric_vals).mean().cpu().item()

        return stats

    def write_overall_stats(
        self,
        dataset_wise_statscore: Dict[str, Dict[str, MulticlassStatScores]],
        dataset_wise_confmat: Dict[str, Dict[str, torch.Tensor]],
        dataset_wise_labelweight: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    ):

        overall = defaultdict(lambda: 0)
        overall_weight = {f"{catg}_weights": defaultdict(lambda: 0) for catg in self.categories}

        for dataset, stats_metrics_dict in dataset_wise_statscore.items():
            scores = {}
            self.dataset_wise_sample_stats[dataset]["sample_wise_file"].close()

            for catg in self.categories:
                scores[f"{catg}_confmat"] = dataset_wise_confmat[dataset][f"{catg}_confmat"]
                overall[f"{catg}_confmat"] += scores[f"{catg}_confmat"]

                for c_l, c_c in dataset_wise_labelweight[dataset][f"{catg}_weights"].items():
                    overall_weight[f"{catg}_weights"][c_l] += c_c

                scores[f"{catg}_score"] = stats_metrics_dict[f"{catg}_stats"].compute()
                overall[f"{catg}_score"] += scores[f"{catg}_score"]

            ITNOverallDataMetric(
                **self.compute_stats(scores, dataset_wise_labelweight[dataset])
            ).write_from_rank_zero_only(self.dataset_wise_sample_stats[dataset]["overall_summary"])

        ITNOverallDataMetric(
            **self.compute_stats(overall, overall_weight)
        ).write_from_rank_zero_only(self.overall_summary_file)


def generate_csv(output_dir: str, dataset_names: list[str]) -> tuple[str, str]:
    stats = defaultdict(lambda: [])
    csv_path = os.path.join(output_dir, "overall_summary.csv")
    for ds_name in dataset_names:
        stats["ds_name"].append(ds_name)
        data = ITNOverallDataMetric.read_from(
            os.path.join(output_dir, ds_name, "overall_summary.json")
        )

        for k, v in vars(data).items():
            if k.startswith("_"):
                continue
            stats[k].append(v)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(stats.keys())
        w.writerows(zip(*stats.values()))

    return csv_path, "overall_summary.csv"


def wrap_up_function(
    output_dir: str,
    dataset_names: List[str],
    model_hparams: DictConfig,
    predictor_hparams: DictConfig,
):
    hparams = OmegaConf.merge(
        {"model_hparams": model_hparams}, {"predictor_hparams": predictor_hparams}
    )
    OmegaConf.save(hparams, os.path.join(output_dir, "overall_hparams.yaml"))
    csv_path, csv_name = generate_csv(output_dir, dataset_names)
    with tarfile.open(os.path.join(output_dir, "results.tar.gz"), "w:gz") as tar:
        for ds_name in dataset_names:
            tar.add(os.path.join(output_dir, ds_name), arcname=ds_name)
        tar.add(csv_path, arcname=csv_name)
        tar.add(os.path.join(output_dir, "overall_summary.json"), arcname="overall_summary.json")
        tar.add(os.path.join(output_dir, "overall_hparams.yaml"), arcname="overall_hparams.yaml")

    # for ds_name in dataset_names:
    #     shutil.rmtree(os.path.join(output_dir, ds_name), ignore_errors=True)
    # os.remove(os.path.join(output_dir, "overall_summary.json"))
