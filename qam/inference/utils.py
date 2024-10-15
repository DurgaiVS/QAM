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
