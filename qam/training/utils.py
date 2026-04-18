import csv
import json
import logging
import math
import os
import tarfile
from copy import deepcopy
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import ConfusionMatrix, Metric, StatScores

from ..utils import TradeTrend, defaultdict, find_available_filename


def f1score(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, f_beta: float = 1.0
) -> torch.Tensor:
    """
    Computes F1 score from True Positives, False Positives and False Negatives.

    Args:
            tp (torch.Tensor): True Positives.
            fp (torch.Tensor): False Positives.
            fn (torch.Tensor): False Negatives.
            f_beta (float): Weightage to Precision and Recall. Default is 1.0.
            NOTE: If f_beta is other than 1.0, then this function calculates F-beta score.

    Returns:
            torch.Tensor: F1 score.
            NOTE: If TP, FP, FN are all zero, then returns nan.
    """
    return ((1 + (f_beta**2)) * tp) / (
        ((1 + (f_beta**2)) * tp) + fp + ((f_beta**2) * fn)
    )


def precision(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """
    Computes Precision from True Positives, False Positives and False Negatives.

    Args:
            tp (torch.Tensor): True Positives.
            fp (torch.Tensor): False Positives.
            fn (torch.Tensor): False Negatives.

    Returns:
            torch.Tensor: Precision.
            NOTE: If TP, FP are zero, then returns nan.
    """
    return tp / (tp + fp)


def recall(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """
    Computes Recall from True Positives, False Positives and False Negatives.

    Args:
            tp (torch.Tensor): True Positives.
            fp (torch.Tensor): False Positives.
            fn (torch.Tensor): False Negatives.

    Returns:
            torch.Tensor: Recall.
            NOTE: If TP, FN are zero, then returns nan.
    """
    return tp / (tp + fn)


METRICS_NAME_AND_FN: Dict[
    str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "f1score": f1score,
    "precision": precision,
    "recall": recall,
}


def isnan(value: Union[float, torch.Tensor]) -> bool:
    """
    Checks if the given value is NaN.

    Args:
            value (Union[float, torch.Tensor]): The value to be checked.

    Returns:
            bool: True if the given value is NaN, False otherwise.
    """
    if isinstance(value, torch.Tensor):
        return torch.isnan(value).all()
    elif isinstance(value, float):
        return math.isnan(value)


class QAMStats:
    """
    Contains the all the stats info for QAM. It contains the confusion matrix,f1score, precision and recall
    for both labelwise and overall. The attributes can be added, subtracted with the other QAMStats and
    multiplied and divided with a scalar value.

    Attributes:
    -----------

    """

    def __init__(self, **kwargs) -> None:
        for attr, default_val in QAMStats.get_all_attributes_name_and_default_val():
            setattr(self, attr, kwargs.pop(attr, default_val))

        if len(kwargs) != 0:
            raise ValueError(f"Unknown attributes provided: `{kwargs.keys()}`")

    def copy(self) -> "QAMStats":
        """
        Copies the QAMStats object recursively.

        Returns:
        --------
        QAMStats:
                The copied QAMStats object.
        """
        return deepcopy(self)

    def walk_through(self) -> Generator[Tuple[str, torch.Tensor, bool], None, None]:
        """
        This function is used to yield the stat scores of this class's attributes,
        excluding `confmat`, coz those stats cannot be logged and attributes
        whose value is `None`.

        Returns:
        --------
        Generator[Tuple[str, torch.Tensor, bool], None, None]:
                A generator of the stat scores along with their names and boolean indicating
                whether it is primary attribute or not.
        """
        for k, v in vars(self).items():
            if (k == "confmat") or (v is None):
                continue

            yield k, v, k in METRICS_NAME_AND_FN

    @staticmethod
    def get_computable_attributes_name_and_fn() -> Generator[
        Tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        None,
        None,
    ]:
        """
        Returns a generator of all computable attributes name and their corresponding functions.

        Returns:
        --------
        Generator[Tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]], None, None]:
                A generator of all computable attributes name and their corresponding functions.
        """
        for metric, m_fn in METRICS_NAME_AND_FN.items():
            for l_name in TradeTrend.get_labels_name():
                yield f"{l_name.lower()}_{metric}", m_fn

    @staticmethod
    def get_all_attributes_name_and_default_val() -> (
        Generator[Tuple[str, Union[int, float]], None, None]
    ):
        """
        Returns a generator of all attributes name for this class and their default values.

        Returns:
        --------
        Generator[Tuple[str, Union[int, float]], None, None]:
                A generator of all attributes name for this class and their default values.
        """
        yield f"confmat", 0

        for metric, _ in METRICS_NAME_AND_FN.items():
            yield metric, math.nan

        for attr, _ in QAMStats.get_computable_attributes_name_and_fn():
            yield attr, math.nan

        yield "label", None
        yield "symbol", None
        yield "prediction", None

    def __add__(self, other: "QAMStats") -> "QAMStats":
        """
        Adds the data from the other QAMStats to the self QAMStats except the attributes
        in the exceptions list.

        Args:
                other (QAMStats): The other QAMStats from which the data is to be added.

        Returns:
                QAMStats: The QAMStats with the added data.
        """
        for k, s_v in vars(self).items():
            o_v = getattr(other, k)
            if (
                (s_v is None)
                or (o_v is None)
                or isinstance(s_v, str)
                or isinstance(o_v, str)
                or isnan(o_v)
                or ("prediction" in k)
                or ("label" in k)
            ):
                continue

            if isnan(s_v):
                setattr(self, k, o_v)
            else:
                setattr(self, k, s_v + o_v)

        if hasattr(self, "prediction"):
            setattr(self, "prediction", None)

        return self

    def __sub__(self, other: "QAMStats") -> "QAMStats":
        """
        Subtracts the data from the other QAMStats to the self QAMStats except the attributes
        in the exceptions list.

        Args:
                other (QAMStats): The other QAMStats from which the data is to be subtracted.

        Returns:
                QAMStats: The QAMStats with the subtracted data.
        """
        for k, s_v in vars(self).items():
            o_v = getattr(other, k)
            if (
                (s_v is None)
                or (o_v is None)
                or isinstance(s_v, str)
                or isinstance(o_v, str)
                or isnan(o_v)
                or ("prediction" in k)
                or ("label" in k)
            ):
                continue

            if isnan(s_v):
                setattr(self, k, -1 * o_v)
            else:
                setattr(self, k, s_v - o_v)

        if hasattr(self, "prediction"):
            setattr(self, "prediction", None)

        return self

    def __mul__(self, value: Union[int, float]) -> "QAMStats":
        """
        Multiplies the data in the self QAMStats with the given value except the attributes in the exceptions
        list.

        Args:
                value (Union[int, float]): The value to be multiplied with.

        Returns:
                QAMStats: The QAMStats with the multiplied data.
        """
        for k, v in vars(self).items():
            if (
                (v is None)
                or isinstance(v, str)
                or isnan(v)
                or ("confmat" in k)
                or ("prediction" in k)
                or ("label" in k)
            ):
                continue

            setattr(self, k, (v * value))

        return self

    def __truediv__(self, value: Union[int, float]) -> "QAMStats":
        """
        Divides the data in the self QAMStats with the given value except the attributes in the exceptions
        list.

        Args:
                value (Union[int, float]): The value to be divided with.

        Returns:
                QAMStats: The QAMStats with the divided data.
        """
        for k, v in vars(self).items():
            if (
                (v is None)
                or isinstance(v, str)
                or isnan(v)
                or ("confmat" in k)
                or ("prediction" in k)
                or ("label" in k)
            ):
                continue

            setattr(self, k, (v / value))

        return self

    @staticmethod
    def compute_avg_from_iterable(samples: List["QAMStats"]) -> "QAMStats":
        """
        Computes the average stats from the given list of QAMStats.

        Args:
                samples (List[QAMStats]): The list of QAMStats.

        Returns:
                QAMStats: The QAMStats with the average stats.
        """
        self = samples[0].copy()
        self.symbol = None
        self.label = None
        self.prediction = None

        for sample in samples[1:]:
            self += sample

        return self / len(samples)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict:
        """
        Returns the dictionary representation of the QAMStats.
        NOTE: Only the attributes with non-default values are returned.

        Returns:
                dict: The dictionary representation of the QAMStats.
        """
        s = deepcopy(vars(self))

        if isinstance(s["label"], TradeTrend):
            s["label"] = s["label"].name
        elif isinstance(s["label"], int):
            s["label"] = TradeTrend(s["label"]).name
        elif isinstance(s["label"], torch.Tensor):
            s["label"] = TradeTrend(s["label"].item()).name

        if isinstance(s["prediction"], TradeTrend):
            s["prediction"] = s["prediction"].name
        elif isinstance(s["prediction"], int):
            s["prediction"] = TradeTrend(s["prediction"]).name
        elif isinstance(s["prediction"], torch.Tensor):
            s["prediction"] = TradeTrend(s["prediction"].item()).name

        for k, v in list(s.items()):
            if isnan(v):
                s.pop(k)

            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    s[k] = v.detach().cpu().item()
                else:
                    s[k] = v.detach().cpu().tolist()

            elif v is None:
                s.pop(k)
        return s

    def to_str(self) -> str:
        """
        Returns the string representation of the QAMStats.

        Returns:
                str: The string representation of the QAMStats.
        """
        return json.dumps(self.to_dict())

    @rank_zero_only
    def write_from_rank_zero_only(self, filepath: str):
        """
        Writes the QAMStats to the given file from rank zero process. This method can be used with
        pytorch-lightning's distributed training / benchmarking.

        Args:
                filepath (str): The file path.
        """
        self.write_to(filepath)

    def write_to(self, filepath: str):
        """
        Writes the QAMStats to the given file.

        Args:
                filepath (str): The file path.
        """
        with open(filepath, "w") as f:
            f.write(self.to_str())

    @classmethod
    def from_dict(cls, data: Dict) -> "QAMStats":
        """
        Loads the QAMStats from the given dictionary.

        Args:
                data (dict): The dictionary to be loaded.

        Returns:
                QAMStats: The loaded QAMStats.
        """
        return cls(**data).reformat_after_loading()

    @classmethod
    def read_from(cls, filepath: str):
        """
        Loads the QAMStats from the given file.

        Args:
                filepath (str): The file path.

        Returns:
                QAMStats: The loaded QAMStats.
        """
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_str(cls, data: str):
        """
        Loads the QAMStats from the given string.

        Args:
                data (str): The string to be loaded.
        """
        return cls.from_dict(json.loads(data))

    def reformat_after_loading(self):
        """
        Reformats the QAMStats after loading from a file or a str or a dict.
        """
        self["label"] = TradeTrend[self["label"]]
        self["prediction"] = TradeTrend[self["prediction"]]

        for k, v in vars(self).items():
            if isinstance(v, list):
                setattr(self, k, torch.tensor(v))

        return self


class QAMMetric(Metric):
    """
    Metric calculator for QAM.

    Args:
            classes_count: int
                    Number of classes to compute the metric for.
    """

    def __init__(self, classes_count: int):
        """
        Args:
                classes_count (int): Number of classes in classifier.
        """
        super().__init__()
        self.statscore = StatScores(
            task="multiclass", num_classes=classes_count, average="none"
        )

        self.confmat = ConfusionMatrix(task="multiclass", num_classes=classes_count)

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{self.update}' is a placeholder and not meant to be used. Please use '{self.forward}' instead."
        )

    def generate_stats_scores(
        self,
        statscore: torch.Tensor,
        confmat: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> QAMStats:
        """
        Generates the QAMStats for the stats score and confusion matrix.

        Args:
                stats (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
                        The stats scores.
                confmat (torch.Tensor):
                        The confusion matrix.

        Returns:
                QAMStats: The QAMStats.
        """
        scores = {}
        for metric, m_fn in METRICS_NAME_AND_FN.items():
            avg: List[torch.Tensor] = []
            for id, (tp, fp, _, fn, _) in enumerate(statscore):
                val = m_fn(tp, fp, fn)
                scores[f"{TradeTrend(id).name.lower()}_{metric}"] = val
                if not torch.isnan(val):
                    avg.append(val)

            scores[metric] = torch.stack(avg).mean()

        return QAMStats(**scores, confmat=confmat, prediction=prediction, label=label)

    # def __call__(self, punct_preds: torch.Tensor, capit_preds: torch.Tensor, punct_labels: torch.Tensor, capit_labels: torch.Tensor) -> QAMStats:
    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> QAMStats:
        """
        Computes the stats for the given prediction and target tensors.

        Args:
                preds (torch.Tensor): The predictions from the model, without padded tokens.
                labels (torch.Tensor): The labels, without padded tokens.

        Returns:
                QAMStats: The scores of the sample as per the categories.
        """

        return self.generate_stats_scores(
            self.statscore(preds, labels), self.confmat(preds, labels), preds, labels
        )

    def compute(self) -> QAMStats:
        """
        Computes the average stats.

        Returns:
                Dict: The average stats for samples inferrenced till.
        """

        return self.generate_stats_scores(
            self.statscore.compute(), self.confmat.compute()
        )

    def reset(self):
        """
        Resets the stats.
        """
        self.statscore.reset()
        self.confmat.reset()


def wrap_up_trainer(
    exp: pl.LightningModule,
    cfg: DictConfig,
    best_model_path: str,
    model_max_length: int = 512,
):
    """
    Wrapper function to save the best model, hparams, onnx model, and torchscript model to a tarfile.

    Args:
        exp (QAM): Experiment object.
        cfg (DictConfig): Configuration object.
        best_model_path (str): Path to the best model.
        model_max_length (int, optional): Maximum length of the model. Defaults to 512.
    """

    if not os.path.exists(best_model_path):
        logging.error(f"Best model path, `{best_model_path}` doesnot exist.")
        return
    else:
        logging.info(f"Best model path, `{best_model_path}`")

    model_path = find_available_filename(cfg.experiment.output_dir, "weights", "pt")
    onnx_path = find_available_filename(cfg.experiment.output_dir, "onnx_model", "onnx")
    torchscript_path = find_available_filename(
        cfg.experiment.output_dir, "torchscript_model", "pt"
    )
    hparams_path = find_available_filename(cfg.experiment.output_dir, "hparams", "yaml")
    tar_path = find_available_filename(cfg.experiment.output_dir, "model", "tar.gz")
    tmp_cfg = os.path.join(cfg.experiment.output_dir, "_hparams.yaml")

    with open(best_model_path, "rb") as checkpoint_file:
        checkpoint = torch.load(checkpoint_file)

        exp.load_state_dict(checkpoint["state_dict"])
        model = exp.model
        model.to("cpu")
        model.eval()

        torch.save(model.state_dict(), model_path)

    try:
        script_model = model.get_torchscript_model(
            cfg.data.batch_size, model_max_length
        )
    except Exception:
        logging.warning(f"Exception occurred when converting model to torchscript.")
    else:
        script_model.save(torchscript_path)

    OmegaConf.save(cfg, hparams_path, True)
    model.export_onnx(onnx_path, cfg.data.batch_size, model_max_length)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="weights.pt")
        tar.add(hparams_path, arcname="hparams.yaml")
        tar.add(onnx_path, arcname="onnx_model.onnx")
        if os.path.exists(torchscript_path):
            tar.add(torchscript_path, arcname="torchscript_model.pt")

    if os.path.exists(tmp_cfg):
        os.remove(tmp_cfg)
    logging.info(f"Model variants and hparams were saved to tarfile in `{tar_path}`")


def init_optimizer(
    cfg: DictConfig, model: torch.nn.Module, params_layerwise: DictConfig
):
    params_overall = []
    for params_per_layer in params_layerwise:
        p = {}
        p["params"] = getattr(model, params_per_layer["params"]).parameters()
        p["lr"] = params_per_layer["lr"]

        params_overall.append(p)

    return hydra.utils.instantiate(cfg, params_overall)


def get_best_model_path(ckpt_cbs: List[ModelCheckpoint]) -> str:
    id = -1
    if ckpt_cbs[0].mode == "max":
        best = float("-inf")
        is_max = True
    else:
        best = float("inf")
        is_max = False

    for i, ckpt_cb in enumerate(ckpt_cbs):
        if getattr(ckpt_cb.best_model_score, "__gt__" if is_max else "__lt__")(best):
            best = ckpt_cb.best_model_score
            id = i

    return ckpt_cbs[id].best_model_path


def generate_csv(output_dir: str, symbols: list[str]) -> tuple[str, str]:
    stats = defaultdict(lambda: [])
    csv_path = os.path.join(output_dir, "overall_summary.csv")
    for symbol in symbols:
        stats["symbol"].append(symbol)
        data = QAMStats.read_from(
            os.path.join(output_dir, symbol, "overall_summary.json")
        )

        stats = data.to_dict()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(stats.keys())
        w.writerows(zip(*stats.values()))

    return csv_path, "overall_summary.csv"


def wrap_up_predictor(
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
