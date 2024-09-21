import json
import os
from dataclasses import dataclass

import torch
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.classification import MulticlassStatScores

from ..training.data.utils import QAMDataSample
from ..utils import QAMFileWriter


@dataclass
class QAMDataMetric:
    labels: str
    dataset_name: str
    prediction: str
    f1_score: float
    precision: float
    recall: float

    def to_json(self) -> str:
        return json.dumps(vars(self))


class QAMInferenceResultsWriter:
    def __init__(self, base_dir: str, output_dir: str) -> None:
        self.output_dir = output_dir
        self.overall_summary_file = f"{self.output_dir}/overall_summary.json"
        self.dataset_wise_sample_stats = {}
        for dataset in os.listdir(base_dir):
            os.makedirs(f"{self.output_dir}/{dataset}")

            self.dataset_wise_sample_stats[dataset] = {
                "sample_wise_stats": f"{self.output_dir}/{dataset}/sample_stats.jsonl.bz2",
                "sample_wise_file": QAMFileWriter(
                    full_path=f"{self.output_dir}/{dataset}/sample_stats.jsonl.gz",
                    size_per_file=-1,
                ),
                "overall_summary": f"{self.output_dir}/{dataset}/overall_summary.json",
            }

    def write_sample_wise_stats(
        self,
        sample: QAMDataSample,
        prediction: torch.Tensor,
        score_dict: dict[str, torch.Tensor],
    ):
        self.dataset_wise_sample_stats[sample.dataset_name]["sample_wise_file"].write(
            QAMDataMetric(
                sample.label,
                sample.dataset_name,
                prediction,
                **score_dict,
            ).to_json()
        )
        self.dataset_wise_sample_stats[sample.dataset_name]["sample_wise_file"].write(
            "\n"
        )

    @rank_zero_only
    def write_overall_stats(
        self, dataset_wise_score: dict[str, dict[str, MulticlassStatScores]]
    ):

        overall = {
            "f1": [],
            "precision": [],
            "recall": [],
        }

        for dataset, metrics in dataset_wise_score.items():
            self.dataset_wise_sample_stats[dataset]["sample_wise_file"].close()

            with open(
                self.dataset_wise_sample_stats[dataset]["overall_summary"], "w"
            ) as f:

                f.write("{\n")

                for metric_name, metric_instance in metrics.items():
                    value = metric_instance.compute()

                    overall[metric_name].append(value)
                    f.write(f'\t"{metric_name}": {value.item()}\n')

                f.write("}\n")

        with open(self.overall_summary_file, "w") as f:
            f.write("{\n")

            for metric_name, value in overall.items():
                f.write(f'\t"{metric_name}": {torch.stack(value).mean().item()}\n')

            f.write("}\n")
