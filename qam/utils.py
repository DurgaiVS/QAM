import asyncio
import gzip
import json
import logging
import math
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
    Tuple,
    Union,
    overload,
    runtime_checkable,
)

import hydra
import torch
from omegaconf import DictConfig

from .constants import (
    CONFIG_PATH,
    LABEL_MAX_DIFF_PERCENT,
    LABEL_MIN_INCREMENT_PERCENT,
    MAX_SEQ_LEN,
    STRIDE_LENGTH,
    SUB_SPLITS,
    TREND_UPDATE_SEQ_LEN,
)


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


TimePointTuple = Tuple[int, float, float, float, float, int, int, float, int]
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


@overload
def get_samples_and_extrarounder_count_per_shard(
    samples_count: int, shards_count: int
) -> int: ...


def get_samples_and_extrarounder_count_per_shard(
    samples_count: int, gpus_count: int, workers_per_gpu_count: int = 1
) -> Tuple[int, int]:
    return (
        samples_count // (gpus_count * workers_per_gpu_count),
        samples_count % (gpus_count * workers_per_gpu_count),
    )


def find_available_filename(
    base_path: str, filename_stem: str, extension: str, add_v: bool = False
) -> str:
    """
    Returns a new filename that is not already present in the specified directory.
    First check for the availability of the basic name, if it is not available, then
    try with version numbers.

    Parameters
    -----------
    base_path: `str`
        The `base_path` parameter is a string representing the directory in which the new file
        will be created.
    filename_stem: `str`
        The `filename_stem` parameter is a string representing the stem of the new filename.
    extension: `str`
        The `extension` parameter is a string representing the extension of the new filename.
    add_v: `bool`
        The `add_v` parameter is a boolean that specifies whether to add a 'v' prefix to the
        filename.

    Returns
    -------
    new_path: `str`
        The `new_path` parameter is a string representing the path to the new file.
    """
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    new_path = os.path.join(base_path, f"{filename_stem}.{extension}")
    if not os.path.isfile(new_path):
        return new_path

    i = 1
    while True:
        new_path = os.path.join(
            base_path, f"{filename_stem}-{'v' if add_v else ''}{i}.{extension}"
        )
        if not os.path.isfile(new_path):
            return new_path

        i += 1


@runtime_checkable
class QAMJSONableClass(Protocol):
    def to_str(self) -> str: ...

    def to_dict(self) -> Dict: ...


class TradeTrend(Enum):
    VERY_HIGH: int = 0
    HIGH: int = auto()
    NO_IMP: int = auto()
    LOW: int = auto()
    VERY_LOW: int = auto()

    @classmethod
    def get_labels_name(cls) -> Generator[str, None, None]:
        for label in cls:
            yield label.name


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

    def __repr__(self) -> str:
        return self.to_str()

    def to_bytes(self) -> bytes:
        return self.to_str().encode()

    def to_str(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict:
        return vars(self)

    def as_tuple(self) -> TimePointTuple:
        return (
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


@dataclass
class DatasetMeta:
    symbols: List[str]
    batch_size: int
    gpus_count: int
    workers_per_gpu_count: int
    train_samples_count: int = 0
    dev_samples_count: int = 0
    test_samples_count: int = 0
    train_steps_count: int = None

    def is_aligning_with(self, other: "DatasetMeta") -> bool:
        return (
            (self.gpus_count % other.gpus_count == 0)
            and (self.workers_per_gpu_count % other.workers_per_gpu_count == 0)
            and set(self.symbols) == set(other.symbols)
        )

    def import_sample_counts(self, other: "DatasetMeta") -> "DatasetMeta":
        for split in SUB_SPLITS:
            if getattr(self, f"{split}_samples_count") or (
                getattr(other, f"{split}_samples_count") is None
            ):
                continue
            setattr(
                self, f"{split}_samples_count", getattr(other, f"{split}_samples_count")
            )

    def get_sample_count(self, split: str) -> int:
        return getattr(self, f"{split}_samples_count")

    def __repr__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict:
        return vars(self)

    def write_to(self, filepath):
        with open(filepath, "w") as f:
            json.dump(vars(self), f)

    def update_samplescount_for_split(
        self, split: str, samples_count: int
    ) -> "DatasetMeta":
        if split not in SUB_SPLITS:
            raise RuntimeError(f"Only `{SUB_SPLITS}` splits are valid.")

        setattr(self, f"{split}_samples_count", samples_count)
        return self

    def compute_max_safe_batches_count(self) -> "DatasetMeta":
        self.train_steps_count = int(
            math.floor(self.train_samples_count / (self.gpus_count * self.batch_size))
        )

        return self

    @classmethod
    def from_str(cls, s) -> "DatasetMeta":
        return cls(**json.loads(s))

    @classmethod
    def from_file(cls, filepath) -> "DatasetMeta":
        with open(filepath) as f:
            return cls(**json.load(f))


@dataclass
class QAMDataSample:
    symbol: str
    frame: Union[List[TimePointTuple], torch.Tensor]
    label: Union[TradeTrend, torch.Tensor]

    def to_str(self) -> str:
        return json.dumps(vars(self))

    # TODO: Check whether named tuples are serializable...
    def to_bytes(self) -> bytes:
        return self.to_str().encode()

    @classmethod
    def from_str(cls, s) -> "QAMDataSample":
        return cls(**json.loads(s))

    def tolist(self) -> "QAMDataSample":
        if isinstance(self.frame, torch.Tensor):
            self.frame = self.frame.cpu().tolist()
            self.label = TradeTrend(self.label.cpu().item())

    def tensorize(self) -> "QAMDataSample":
        if not isinstance(self.frame, torch.Tensor):
            self.frame = torch.tensor(self.frame).to(torch.float32)
            self.label = torch.tensor(self.label.value).to(torch.long)
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
    def from_timepoint_list(
        cls, samples: List[QAMTimePoint], label: TradeTrend
    ) -> "QAMDataSample":
        qam_data_tuples = []
        symbol = samples[0].symbol

        for sample in samples:
            assert (
                symbol == sample.symbol
            ), f"Unexpected symbol {sample.symbol} encountered, expected {symbol}. Cannot create a sample out of different symbol's timepoint."

            qam_data_tuples.append(sample.as_tuple())

        return cls(symbol, qam_data_tuples, label)

    @staticmethod
    def get_label(
        entry_point: QAMTimePoint, trend_samples: List[QAMTimePoint]
    ) -> TradeTrend:
        """ """
        # NOTE: If the trend moves up beyond this absolute difference percentage,
        #       we will consider using `High` or `VeryHigh` label...

        #       This is due to the brokerage and extra charges we'll get from our
        #       profit, like, if we get 2% profit, after paying all the charges
        #       when selling we might make a loss. So having a buffer region, and
        #       only if the trend goes above that, we'll buy/sell.
        max_val_entry = entry_point.close + (
            entry_point.close * LABEL_MIN_INCREMENT_PERCENT
        )

        # NOTE: Minimum fluctuation absolute percentage from the last entry value, to consider
        #       for the `Very...` label.
        max_val_border_for_very = entry_point.close + (
            entry_point.close * LABEL_MAX_DIFF_PERCENT
        )
        min_val_border_for_very = entry_point.close - (
            entry_point.close * LABEL_MAX_DIFF_PERCENT
        )

        max_val, min_val = entry_point.close, entry_point.close

        for trend_sample in trend_samples:
            if trend_sample.high > max_val:
                max_val = trend_sample.high
            if trend_sample.low < min_val:
                min_val = trend_sample.low

        if max_val >= max_val_entry:
            if max_val > max_val_border_for_very:
                return TradeTrend.VERY_HIGH
            else:
                return TradeTrend.HIGH

        elif min_val < entry_point.close:
            if min_val < min_val_border_for_very:
                return TradeTrend.VERY_LOW
            else:
                return TradeTrend.LOW

        else:
            logging.warning(
                "Encountered a case where the trend is neither moving up nor moving down."
            )
            return TradeTrend.NO_IMP

    @classmethod
    def init_from_history_and_future_timepoints(
        cls, data_samples: List[QAMTimePoint], trend_samples: List[QAMTimePoint]
    ) -> "QAMDataSample":
        label = cls.get_label(data_samples[-1], trend_samples)
        return cls.from_timepoint_list(trend_samples, label)


# NOTE: The reason for not using `yield_sample_from_file` in the below fn is
#       because, we have to generate a sample from all possible timepoints
#       serially available, like, the last `K`samples...
def yield_sample_from_serially_sorted_files(
    paths: List[Path],
) -> Generator[QAMDataSample, None, None]:
    samples: List[QAMTimePoint] = []
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                sample = QAMTimePoint.from_str(line)
                samples.append(sample)

                if len(samples) < (MAX_SEQ_LEN + TREND_UPDATE_SEQ_LEN):
                    continue

                yield QAMDataSample.init_from_history_and_future_timepoints(
                    samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:]
                )
                samples = samples[STRIDE_LENGTH:]

    if len(samples) > MAX_SEQ_LEN:
        yield QAMDataSample.init_from_history_and_future_timepoints(
            samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:]
        )


def yield_sample_from_file(path: Path) -> Generator[QAMDataSample, None, None]:
    samples: List[QAMTimePoint] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            sample = QAMTimePoint.from_str(line)
            samples.append(sample)

            if len(samples) < (MAX_SEQ_LEN + TREND_UPDATE_SEQ_LEN):
                continue

            yield QAMDataSample.init_from_history_and_future_timepoints(
                samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:]
            )
            samples = samples[STRIDE_LENGTH:]

    if len(samples) > MAX_SEQ_LEN:
        yield QAMDataSample.init_from_history_and_future_timepoints(
            samples[:MAX_SEQ_LEN], samples[MAX_SEQ_LEN:]
        )


@dataclass
class QAMDataBatch:
    symbols: List[str] = field(default_factory=list)
    frames: Union[torch.Tensor, List[torch.Tensor]] = field(default_factory=list)
    labels: Union[torch.Tensor, List[torch.Tensor]] = field(default_factory=list)
    lengths: Union[torch.Tensor, List[int]] = field(default_factory=list)

    def tensorize(self) -> "QAMDataBatch":
        if not isinstance(self.frames, torch.Tensor):
            self.pad_samples_if_needed()

            self.frames = torch.stack(self.frames).to(torch.float32)
            self.labels = torch.stack(self.labels).to(torch.long)
            self.lengths = torch.tensor(self.lengths).to(torch.long)
        return self

    def pad_samples_if_needed(self):
        s_lens = set(self.lengths)
        if len(s_lens) == 1:
            return

        max_s_len = max(s_lens)
        for sample_id, (length, frame) in enumerate(zip(self.lengths, self.frames)):
            if length == max_s_len:
                continue

            assert frame.shape[-1] == len(TimePointTuple)

            pad_len = max_s_len - length
            frame = torch.nn.functional.pad(
                frame, (0, 0, 0, pad_len), "constant", 0
            )  # pad right on 'dim=-2'
            self.frames[sample_id] = frame

            assert frame.shape[-1] == len(TimePointTuple)

    def to(self, *args, **kwargs) -> "QAMDataBatch":
        self.frames = self.frames.to(*args, **kwargs)
        self.labels = self.labels.to(*args, **kwargs)
        self.lengths = self.lengths.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs) -> "QAMDataBatch":
        self.frames = self.frames.cuda(*args, **kwargs)
        self.labels = self.labels.cuda(*args, **kwargs)
        self.lengths = self.lengths.cuda(*args, **kwargs)
        return self

    def cpu(self) -> "QAMDataBatch":
        self.frames = self.frames.cpu()
        self.labels = self.labels.cpu()
        self.lengths = self.lengths.cpu()
        return self

    def __len__(self) -> int:
        return len(self.symbols)

    def extend(self, samples: List[QAMDataSample]) -> "QAMDataBatch":
        for sample in samples:
            self.symbols.append(sample.symbol)
            self.frames.append(sample.frame)
            self.labels.append(sample.label)
            self.lengths.append(len(sample))
        return self

    def __getitem__(self, index: int) -> QAMDataSample:
        return QAMDataSample(
            self.symbols[index],
            self.frames[index],
            self.labels[index],
            self.lengths[index],
        )

    @classmethod
    def from_list(cls, samples: List[QAMDataSample]) -> "QAMDataBatch":
        self = cls()

        for sample in samples:
            self.symbols.append(sample.symbol)
            self.frames.append(sample.frame)
            self.labels.append(sample.label)
            self.lengths.append(len(sample))
        return self.tensorize()


class QAMFileWriter:
    """
    ITNFileWriter is used to write the data into the files in the ITN format. It can write the data to many shards
    based on the size per shard or the count per shard. Either full path or base directory, filename stem and extension
    should be provided. If the size per file is provided, then the data will be written to the file until the size
    reaches the provided size. If the count per file is provided, then the data will be written to the file until the
    count reaches the provided count. If the extra one rounder is provided, then the count will be decremented by one
    and the extra one rounder will be added to the count. The file will be written in gzip format. The file will be
    closed when the object is deleted or when the close method is called. The object can be used as a context manager.
    Eg:
    ```python
    with ITNFileWriter(
        base_dir="data",
        filename_stem="sample",
        extension="json",
        count_per_file=24,
        extra_one_rounder=4,
    ) as writer:
        ...
    ```
    The first 4 files will have 25 samples(writes) and the remaining files will have 24 samples(writes).

    Args:
        full_path (Optional[str], optional): The full path of the file to be written.
        base_dir (Optional[str], optional): The base directory where the files will be stored.
        filename_stem (Optional[str], optional): The stem of the filename.
        extension (Optional[str], optional): The extension of the file.
        size_per_file (Optional[float], optional): The size of the file in bytes.
        count_per_file (Optional[int], optional): Number of lines(or writes in case of data groups. Each write will insert N sub samples) per file.
        extra_one_rounder (Optional[int], optional): The extra one rounder to be added. This will be used only when the count per file is provided.
        writer_mode (str, optional): The mode in which the file should be opened. Either 't' or 'b'. Defaults to 't', where,
            't': text mode
            'b': binary mode

    Raises:
        AttributeError: If the size_per_file or count_per_file or full_path is not provided.
        AttributeError: If the full_path or base_dir, filename_stem, extension is not provided.
    """

    def __init__(
        self,
        full_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        filename_stem: Optional[str] = None,
        extension: Optional[str] = None,
        size_per_file: Optional[float] = None,
        count_per_file: Optional[int] = None,
        extra_one_rounder: Optional[int] = None,
        writer_mode: str = "t",
    ):
        if not (full_path or (base_dir and filename_stem and extension)):
            raise AttributeError(
                "Specify either `full_path` or `base_dir`, `filename_stem`, `extension` when initialising"
            )

        if (not (size_per_file or count_per_file)) and (full_path is None):
            raise AttributeError(
                "Specify either `size_per_file` in bytes or `count_per_file` when initialising"
            )

        if writer_mode not in ("t", "b"):
            raise AttributeError("Only 't' or 'b' are supported for the writer_mode")

        self.full_path = full_path
        self.base_dir = base_dir
        self.filename_stem = filename_stem
        self.extension = extension
        self.size_per_file = size_per_file
        self.count_per_file = count_per_file
        self.extra_one_rounder = (
            extra_one_rounder
            if isinstance(extra_one_rounder, int) and (extra_one_rounder > 0)
            else None
        )
        self._files_counter = 0
        self._count = 0
        self._mode = writer_mode

        if count_per_file:
            self._should_wrap = self._should_wrap_count
        elif size_per_file:
            self._should_wrap = self._should_wrap_size
        else:
            self._should_wrap = lambda: False

        if writer_mode == "b":
            self._new_line_char = b"\n"
        else:
            self._new_line_char = "\n"

        self._open()

    def _open(self):
        self._file = gzip.open(
            (
                self.full_path
                if self.full_path
                else f"{self.base_dir}/{self.filename_stem}-{self._files_counter:04d}.{self.extension}"
            ),
            f"w{self._mode}",
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
        if not isinstance(s, list):
            s = [s]

        for i in s:
            if getattr(i, "to_str", None):
                self._file.write(i.to_str())
            else:
                self._file.write(i)
            self._file.write(self._new_line_char)

        if self._should_wrap():
            self._wrap_up_and_open_new()

    def close(self):
        if self._file.closed:
            return

        _size = self._file.tell()
        self._file.close()
        if _size == 0:
            os.remove(self._file.name)

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
        self.close()


def get_cfg(config_name: str, overrides: List[str], job_name: str) -> DictConfig:
    with hydra.initialize(
        config_path=CONFIG_PATH, job_name=job_name, version_base=None
    ):
        return hydra.compose(config_name=config_name, overrides=overrides)


class WorkerPool:
    def __init__(
        self,
        fn: Callable[..., None],
        worker_count: int,
        args: List[Any] = [],
        kwargs: Dict[str, Any] = {},
        mappable: Optional[Union[List[Any], Dict[str, List[Any]]]] = None,
        backend: str = "thread",
        start_method: str = "fork",
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.mappable = mappable
        self.worker_count = worker_count
        self.start_method = start_method

        if mappable == None:
            self.mapper = lambda: [], {}
        elif isinstance(mappable, dict):
            self.mapper = self.map_dict_iterable
        else:
            self.mapper = self.map_list_iterable

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
        for arg, kwarg in self.mapper():
            t = threading.Thread(
                target=self.fn, args=(self.args + arg), kwargs=(self.kwargs | kwarg)
            )
            t.start()
            self._w.append(t)

            if len(self._w) == self.worker_count:
                self.wait_and_pop()

    def backend_process(self):
        ctx = mp.get_context(self.start_method)

        for arg, kwarg in self.mapper():
            p = ctx.Process(
                target=self.fn, args=(self.args + arg), kwargs=(self.kwargs | kwarg)
            )
            p.start()
            self._w.append(p)

            if len(self._w) == self.worker_count:
                self.wait_and_pop()
