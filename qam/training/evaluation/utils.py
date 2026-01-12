import os
from typing import Dict, List, Union

import torch

from ...utils import QAMFileWriter, TradeTrend, defaultdict, find_available_filename
from ..data.utils import QAMDataSample
from ..utils import METRICS_NAME_AND_FN, QAMMetric, QAMStats


class QAMInferenceResultsWriter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.overall_summary_file = os.path.join(
            self.output_dir, "overall_summary.json"
        )
        self.symbolwise_sample_stats = defaultdict(self.default_fn, True)

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
        stats: QAMStats,
    ):
        stats.label = TradeTrend(sample.label)
        stats.prediction = TradeTrend(pred)
        self.symbolwise_sample_stats[sample.symbol]["sample_wise_file"].write(stats)

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

        for metric_name, metric_fn in METRICS_NAME_AND_FN.items():
            metric_vals = []
            for id, (tp, fp, _, fn, _) in enumerate(scores["stats"]):
                val = metric_fn(tp, fp, fn)
                stats[f"{TradeTrend(id).name.lower()}_{metric_name}"] = val.cpu().item()

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

            stats[metric_name] = torch.stack(metric_vals).mean().cpu().item()

        return stats

    def write_overall_stats(self, symbolwise_metric: Dict[str, QAMMetric]):

        overall_stats = QAMStats()
        # overall_weight = defaultdict(lambda: 0)

        for symbol, metric in symbolwise_metric.items():
            self.symbolwise_sample_stats[symbol]["sample_wise_file"].close()

            stats = metric.compute()
            stats.write_from_rank_zero_only(
                self.symbolwise_sample_stats[symbol]["overall_summary"]
            )

        overall_stats.write_from_rank_zero_only(self.overall_summary_file)
