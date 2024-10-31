import asyncio
import gzip
import json
import logging
import multiprocessing as mp
import os
import threading
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import hydra
import torch
from omegaconf import DictConfig

from .constants import (
    CONFIG_PATH,
    LABEL_THRESHOLD_VALUE,
    MAX_SEQ_LEN,
    STRIDE_LENGTH,
    TREND_UPDATE_SEQ_LEN,
)

QAMTimePointTuple = namedtuple(
    "QAMTimePointTuple",
    [
        "symbol_hex",
        "open",
        "close",
        "high",
        "low",
        "n_trades",
        "volume",
        "volume_wa",
        "time_hex",
    ],
)


@runtime_checkable
class QAMJSONableClass(Protocol):
    def to_str(self) -> str: ...

    def to_dict(self) -> Dict: ...


class Classifier(Enum):
    VERY_HIGH: int = 0
    HIGH: int = auto()
    LOW: int = auto()
    VERY_LOW: int = auto()
    NO_IMP: int = auto()


@dataclass
class QAMTimePoint:
    symbol: str
    open: float
    close: float
    high: float
    low: float
    n_trades: int
    volume: int
    volume_wa: float
    time: str

    def __lt__(self, other: "QAMTimePoint") -> bool:
        return self.low < other.low

    def __le__(self, other: "QAMTimePoint") -> bool:
        return self.low <= other.low

    def __gt__(self, other: "QAMTimePoint") -> bool:
        return self.low > other.low

    def __ge__(self, other: "QAMTimePoint") -> bool:
        return self.low >= other.low

    def __eq__(self, other: "QAMTimePoint") -> bool:
        return (self.symbol == other.symbol) and (self.time == other.time)

    def __ne__(self, other: "QAMTimePoint") -> bool:
        return (self.symbol != other.symbol) or (self.time != other.time)

    def __repr__(self):
        return json.dumps(vars(self))

    def to_str(self):
        return json.dumps(vars(self))

    def to_dict(self):
        return vars(self)

    def as_timepoint_tuple(self) -> QAMTimePointTuple:
        return QAMTimePointTuple(
            hash(self.symbol),
            self.open,
            self.close,
            self.high,
            self.low,
            self.n_trades,
            self.volume,
            self.volume_wa,
            hash(self.time),
        )

    @classmethod
    def from_str(cls, s: str) -> "QAMTimePoint":
        return cls(**json.loads(s))

    @classmethod
    def yield_from_file(
        cls, file_path: Union[str, Path]
    ) -> Generator["QAMTimePoint", None, None]:
        with gzip.open(file_path, "rt") as f:
            for line in f:
                yield cls.from_str(line)


# TODO: Finish this
@dataclass
class DatasetMeta:

    def update_stats_for_group(self, group_type: str, samples_count: int):
        pass

    def compute_max_safe_batches_count(
        self, batch_size: int, gpus_count: int
    ) -> "DatasetMeta":
        pass


@dataclass
class QAMDataSample:
    symbol: str
    frame: Union[List[QAMTimePointTuple], torch.Tensor]
    label: Union[int, torch.Tensor]

    def tensorize(self) -> "QAMDataSample":
        if not isinstance(self.frame, torch.Tensor):
            self.frame = torch.tensor(self.frame).to(torch.float32)
            self.label = torch.tensor(self.label).to(torch.long)
        return self

    def to(self, *args, **kwargs) -> "QAMDataSample":
        self.frame = self.frame.to(*args, **kwargs)
        self.label = self.label.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs) -> "QAMDataSample":
        self.frame = self.frame.cuda(*args, **kwargs)
        self.label = self.label.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs) -> "QAMDataSample":
        self.frame = self.frame.cpu(*args, **kwargs)
        self.label = self.label.cpu(*args, **kwargs)
        return self

    def __len__(self) -> int:
        return len(self.frame)

    @classmethod
    def from_list(
        cls, samples: List[QAMTimePoint], label: Classifier
    ) -> "QAMDataSample":
        qam_data_tuples = []
        symbol = samples[0].symbol

        for sample in samples:
            assert (
                symbol == sample.symbol
            ), f"Unexpected symbol {sample.symbol} encountered, expected {symbol}. Cannot create a sample out of different sample's timepoint."

            qam_data_tuples.append(sample.as_timepoint_tuple())

        return cls(symbol, qam_data_tuples, label.value)

    @classmethod
    def get_label(
        entry_point: QAMTimePoint, trend_samples: List[QAMTimePoint]
    ) -> Classifier:
        max_val, min_val = entry_point, entry_point

        for trend_sample in trend_samples:
            if trend_sample > max_val:
                max_val = trend_sample
            if trend_sample < min_val:
                min_val = trend_sample

        if max_val != entry_point:
            graph_diff = max_val.low - entry_point.low
        elif min_val != entry_point:
            graph_diff = min_val.low - entry_point.low
        else:
            logging.warning(
                "Encountered a case where the trend is neither moving up nor moving down."
            )
            return Classifier.NO_IMP

        if graph_diff > LABEL_THRESHOLD_VALUE:
            return Classifier.VERY_HIGH

        if graph_diff > 0 and graph_diff < LABEL_THRESHOLD_VALUE:
            return Classifier.HIGH

        if graph_diff < 0 and graph_diff > -LABEL_THRESHOLD_VALUE:
            return Classifier.LOW

        if graph_diff < -LABEL_THRESHOLD_VALUE:
            return Classifier.VERY_LOW

    @classmethod
    def init_sample(
        cls, data_samples: List[QAMTimePoint], trend_samples: List[QAMTimePoint]
    ) -> "QAMDataSample":
        label = cls.get_label(data_samples[-1], trend_samples)
        return cls.from_list(trend_samples, label)


def yield_sample_from_file(self, path: Path) -> Generator[QAMDataSample, None, None]:
    samples: List[QAMTimePoint] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            sample = QAMTimePoint.from_str(line)
            samples.append(sample)

            if len(samples) < (MAX_SEQ_LEN + TREND_UPDATE_SEQ_LEN):
                continue

            yield QAMDataSample.init_sample(
                samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:]
            )
            samples = samples[STRIDE_LENGTH:]

    if len(samples) > MAX_SEQ_LEN:
        yield QAMDataSample.init_sample(samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:])


@dataclass
class QAMDataBatch:
    symbols: List[str] = field(default_factory=list)
    frames: Union[torch.Tensor, List[torch.Tensor]] = field(default_factory=list)
    labels: Union[torch.Tensor, List[torch.Tensor]] = field(default_factory=list)

    def tensorize(self) -> "QAMDataBatch":
        if not isinstance(self.frames, torch.Tensor):
            self.frames = torch.stack(self.frames).to(torch.float32)
            self.labels = torch.stack(self.labels).to(torch.long)
        return self

    def to(self, device) -> "QAMDataBatch":
        self.frames = self.frames.to(device)
        self.labels = self.labels.to(device)
        return self

    def cuda(self, device) -> "QAMDataBatch":
        self.frames = self.frames.cuda(device)
        self.labels = self.labels.cuda(device)
        return self

    def cpu(self) -> "QAMDataBatch":
        self.frames = self.frames.cpu()
        self.labels = self.labels.cpu()
        return self

    def __len__(self) -> int:
        return len(self.symbols)

    def extend(self, samples: List[QAMDataSample]) -> "QAMDataBatch":
        for sample in samples:
            self.frames.append(sample.frame)
            self.labels.append(sample.label)
        return self

    def __getitem__(self, index: int) -> QAMDataSample:
        return QAMDataSample(
            self.symbols[index],
            self.frames[index],
            self.labels[index],
        )

    @classmethod
    def from_list(cls, samples: List[QAMDataSample]) -> "QAMDataBatch":
        self = cls()

        for sample in samples:
            self.symbols.append(sample.symbol)
            self.frames.append(sample.frame)
            self.labels.append(sample.label)

        return self.tensorize()


class QAMFileWriter:
    def __init__(
        self,
        full_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        filename_stem: Optional[str] = None,
        extension: Optional[str] = None,
        size_per_file: Optional[float] = None,
        count_per_file: Optional[int] = None,
        extra_one_rounder: Optional[int] = None,
    ):
        if not (full_path or (base_dir and filename_stem and extension)):
            raise AttributeError(
                "Specify either `full_path` or `base_dir`, `filename_stem`, `extension` when initialising"
            )

        if (not (size_per_file or count_per_file)) and (full_path is None):
            raise AttributeError(
                "Specify either `size_per_file` in bytes or `count_per_file` when initialising"
            )

        self.full_path = full_path
        self.base_dir = base_dir
        self.filename_stem = filename_stem
        self.extension = extension
        self.size_per_file = size_per_file
        self.count_per_file = count_per_file
        self.extra_one_rounder = extra_one_rounder
        self._files_counter = 0
        self._count = 0

        if count_per_file:
            self._should_wrap = self._should_wrap_count
        elif size_per_file:
            self._should_wrap = self._should_wrap_size
        else:
            self._should_wrap = lambda: False

        self._open()

    def _open(self):
        self._file = gzip.open(
            (
                self.full_path
                if self.full_path
                else f"{self.base_dir}/{self.filename_stem}-{self._files_counter:04d}.{self.extension}"
            ),
            "wt",
        )

        if self.count_per_file and self.extra_one_rounder:
            self._count = -1
            self.extra_one_rounder -= 1

    def _wrap_up_and_open_new(self):
        self._file.close()
        self._files_counter += 1
        self._count = 0
        self._open()

    def write(self, s: Union[str, QAMJSONableClass, List[QAMJSONableClass], List[str]]):
        if isinstance(s, list):
            for i in s:
                if getattr(i, "to_str", None):
                    self._file.write(i.to_str())
                else:
                    self._file.write(i)
                self._file.write("\n")

        else:
            if getattr(s, "to_str", None):
                self._file.write(s.to_str())
            else:
                self._file.write(s)
            self._file.write("\n")

        if self._should_wrap():
            self._wrap_up_and_open_new()

    def close(self):
        _size = self._file.tell()
        self._file.close()
        if _size == 0:
            os.remove(self._file.name)
        else:
            self._count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _should_wrap_count(self) -> bool:
        self._count += 1
        return self._count == self.count_per_file if self.count_per_file != 0 else False

    def _should_wrap_size(self) -> bool:
        return self._file.tell() >= self.size_per_file

    def __del__(self):
        if not self._file.closed:
            self._file.close()


class defaultdict(dict):
    def __init__(
        self,
        default_factory: Optional[Callable] = None,
        default_factory_key_argument: bool = False,
        /,
        *args,
        **kwargs,
    ):
        self.default_factory = default_factory
        self.default_factory_key_argument = default_factory_key_argument

        super().__init__(*args, **kwargs)
        ...

    def __missing__(self, key: str):
        if self.default_factory:
            self[key] = (
                self.default_factory(key)
                if self.default_factory_key_argument
                else self.default_factory()
            )
            return self[key]
        else:
            super().__missing__(key)


def find_available_filename(
    base_path: str, filename_stem: str, extension: str, add_v: bool = True
) -> str:
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    i = 0
    while True:
        new_path = os.path.join(
            base_path, f"{filename_stem}-{'v' if add_v else ''}{i}.{extension}"
        )
        if not os.path.isfile(new_path):
            return new_path

        i += 1


def get_cfg(config_name: str, overrides: List[str], job_name: str) -> DictConfig:
    with hydra.initialize(
        config_path=CONFIG_PATH, job_name=job_name, version_base=None
    ):
        return hydra.compose(config_name=config_name, overrides=overrides)


class WorkerPool:
    def __init__(
        self,
        fn: Callable[..., None],
        mappable: Union[List[Any], Dict[str, List[Any]]],
        worker_count: int,
        args: List[Any] = [],
        kwargs: Dict[str, Any] = {},
        backend: str = "thread",
        start_method: str = "fork",
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.mappable = mappable
        self.worker_count = worker_count
        self.start_method = start_method

        self.mapper = (
            self.map_dict_iterable
            if isinstance(mappable, dict)
            else self.map_list_iterable
        )
        if backend == "thread":
            self.backend = self.backend_thread
        elif backend == "process":
            self.backend = self.backend_process
        self._w = []

    def start(self):
        self.backend()

    def join(self):
        for w in self._w:
            w.join()

        self._w = []

    def map_list_iterable(self):
        for arg in self.mappable:
            yield [arg], {}

    def map_dict_iterable(self):
        k_s = list(self.mappable.keys())
        v_s = list(self.mappable.values())

        for _v_s in zip(*v_s):
            kwarg = {}
            for k, v in zip(k_s, _v_s):
                kwarg[k] = v

            yield [], kwarg

    def wait_and_pop(self):
        while True:
            for i, w in enumerate(self._w):
                w.join(0.5)
                if w.is_alive():
                    continue

                self._w.pop(i)
                return

    def backend_asyncio(self):
        for arg, kwarg in self.mapper():
            asyncio.run(self.fn(*(self.args + arg), **(self.kwargs | kwarg)))

    def backend_thread(self):
        w_count = 0

        for arg, kwarg in self.mapper():
            t = threading.Thread(
                target=self.fn, args=(self.args + arg), kwargs=(self.kwargs | kwarg)
            )
            t.start()
            self._w.append(t)

            if len(self._w) > self.worker_count:
                self.wait_and_pop()

    def backend_process(self):
        ctx = mp.get_context(self.start_method)

        for arg, kwarg in self.mapper():
            p = ctx.Process(
                target=self.fn, args=(self.args + arg), kwargs=(self.kwargs | kwarg)
            )
            p.start()
            self._w.append(p)

            if len(self._w) > self.worker_count:
                self.wait_and_pop()
