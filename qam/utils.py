import gzip, os
from typing import Optional, Protocol, runtime_checkable, Union, Callable

import hydra
from omegaconf import DictConfig

from .constants import CONFIG_PATH


@runtime_checkable
class QAMJSONableClass(Protocol):
    def to_str(self) -> str:
        pass

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
        if not (size_per_file or count_per_file):
            raise AttributeError(
                "Specify either `size_per_file` in bytes or `count_per_file` when initialising"
            )

        if not (full_path or (base_dir and filename_stem and extension)):
            raise AttributeError(
                "Specify either `full_path` or `base_dir`, `filename_stem`, `extension` when initialising"
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

        self._open()

    def _open(self):
        self.file = gzip.open(
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
        self.file.close()
        self._files_counter += 1
        self._count = 0
        self._open()

    def write(self, s: Union[str, QAMJSONableClass, list[QAMJSONableClass], list[str]]):
        if isinstance(s, list):
            for i in s:
                if getattr(i, "to_str", None):
                    self.file.write(i.to_str())
                else:
                    self.file.write(i)
                self.file.write("\n")

        else:
            if getattr(s, "to_str", None):
                self.file.write(s.to_str())
            else:
                self.file.write(s)
            self.file.write("\n")

        if self._should_wrap():
            self._wrap_up_and_open_new()

    def close(self):
        _size = self.file.tell()
        self.file.close()
        if _size == 0:
            os.remove(self.file.name)
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
        return self.file.tell() >= self.size_per_file

    def __del__(self):
        if not self.file.closed:
            self.file.close()


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
        new_path = os.path.join(base_path, f"{filename_stem}-{'v' if add_v else ''}{i}.{extension}")
        if not os.path.isfile(new_path):
            return new_path

        i += 1


def get_cfg(config_name: str, overrides: list[str], job_name: str) -> DictConfig:
    with hydra.initialize(
        config_path=CONFIG_PATH, job_name=job_name, version_base=None
    ):
        return hydra.compose(config_name=config_name, overrides=overrides)
