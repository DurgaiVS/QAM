import csv
import json
import os
import tarfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import MulticlassStatScores

from ...utils import Classifier, QAMFileWriter, defaultdict, find_available_filename
from ..data.utils import QAMDataSample


@dataclass
class QAMOverallDataMetric:
    confmat: List[List[float]]
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
    symbol: str
    prediction: Classifier
    ground_truth: Classifier

    def __post_init__(self):
        self._exceptions_list = ["ground_truth", "dataset_name", "prediction"]


class QAMInferenceResultsWriter:
    def __init__(
        self,
        output_dir: str,
        metrics_name: List[str],
        metrics_fn: List[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        ],
    ) -> None:
        self.output_dir = output_dir
        self.overall_summary_file = os.path.join(
            self.output_dir, "overall_summary.json"
        )
        self.symbolwise_sample_stats = defaultdict(self.default_fn, True)
        self.metrics_name = metrics_name
        for m_name, m_fn in zip(metrics_name, metrics_fn):
            setattr(self, m_name, m_fn)

    def default_fn(self, key: str) -> Dict:
        ds_path = os.path.join(self.output_dir, key)
        os.makedirs(ds_path, exist_ok=True)
        return {
            "sample_wise_file": QAMFileWriter(
                full_path=find_available_filename(
                    ds_path, "sample_stats", "jsonl.gz", False
                ),
                size_per_file=float("inf"),
            ),
            "overall_summary": os.path.join(ds_path, "overall_summary.json"),
        }

    def write_sample_wise_stats(
        self,
        sample: QAMDataSample,
        pred: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        self.symbolwise_sample_stats[sample.symbol]["sample_wise_file"].write(
            QAMSampleDataMetric(
                ground_truth=Classifier(sample.label.item()),
                symbol=sample.symbol,
                prediction=Classifier(pred.item()),
                **score_dict,
            )
        )

    def compute_stats(
        self,
        scores: Dict[str, torch.Tensor],  # weights: Dict[str, Dict[int, torch.Tensor]]
    ) -> Dict[str, Union[float, List]]:
        stats = {}
        stats["confmat"] = (
            (scores["confmat"] / scores["confmat"].sum(dim=1)[:, None]).cpu().tolist()
        )

        # _weights = torch.zeros(
        #     len(gl[f"{catg.capitalize()}Label"].__members__), dtype=torch.int64
        # )
        # for k, v in weights[f"{catg}_weights"].items():
        #     if k == gl.get(f"{catg.capitalize()}Label").NoOp.value:
        #         continue
        #     _weights[k] = v

        # weights[f"{catg}_weights"] = _weights / _weights.sum()

        for metric in self.metrics_name:
            metric_vals = []
            for id, (tp, fp, _, fn, _) in enumerate(scores["stats"]):
                val = getattr(self, metric)(tp, fp, fn)
                stats[f"{Classifier(id).name.lower()}_{metric}"] = val.cpu().item()

                # NOTE:
                # instead of computing sum of tp, fp, fn 's, we've computed weighted avg of labelwise metric
                # values coz, summing up is messing with the scores, causing F1, Prec, Rec all to be same...
                # like, FPs == FNs,
                if torch.isnan(val):
                    metric_vals.append(
                        torch.zeros(1, dtype=val.dtype)
                    )  #  * weights[f"{catg}_weights"][id]
                else:
                    metric_vals.append(val)

            stats[metric] = torch.stack(metric_vals).mean().cpu().item()

        return stats

    def write_overall_stats(
        self,
        symbolwise_statscore: Dict[str, MulticlassStatScores],
        symbolwise_confmat: Dict[str, torch.Tensor],
        symbolwise_labelweight: Dict[str, Dict[int, torch.Tensor]],
    ):

        overall = defaultdict(lambda: 0)
        # overall_weight = defaultdict(lambda: 0)

        for dataset, stats_metric in symbolwise_statscore.items():
            scores = {}
            self.symbolwise_sample_stats[dataset]["sample_wise_file"].close()

            scores["confmat"] = symbolwise_confmat[dataset]
            overall["confmat"] += scores["confmat"]

            # for c_l, c_c in symbolwise_labelweight[dataset][f"{catg}_weights"].items():
            #     overall_weight[f"{catg}_weights"][c_l] += c_c

            scores["stats"] = stats_metric.compute()
            overall["stats"] += scores["stats"]

            QAMOverallDataMetric(
                **self.compute_stats(scores)  # , symbolwise_labelweight[dataset])
            ).write_from_rank_zero_only(
                self.symbolwise_sample_stats[dataset]["overall_summary"]
            )

        QAMOverallDataMetric(
            **self.compute_stats(overall)  # , overall_weight)
        ).write_from_rank_zero_only(self.overall_summary_file)


def generate_csv(output_dir: str, symbols: list[str]) -> tuple[str, str]:
    stats = defaultdict(lambda: [])
    csv_path = os.path.join(output_dir, "overall_summary.csv")
    for symbol in symbols:
        stats["symbol"].append(symbol)
        data = QAMOverallDataMetric.read_from(
            os.path.join(output_dir, symbol, "overall_summary.json")
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
    symbols: List[str],
    model_hparams: DictConfig,
    predictor_hparams: DictConfig,
):
    hparams = OmegaConf.merge(
        {"model_hparams": model_hparams}, {"predictor_hparams": predictor_hparams}
    )
    OmegaConf.save(hparams, os.path.join(output_dir, "overall_hparams.yaml"))
    csv_path, csv_name = generate_csv(output_dir, symbols)
    with tarfile.open(os.path.join(output_dir, "results.tar.gz"), "w:gz") as tar:
        for symbol in symbols:
            tar.add(os.path.join(output_dir, symbol), arcname=symbol)
        tar.add(csv_path, arcname=csv_name)
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
