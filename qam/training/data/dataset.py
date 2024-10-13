import gzip
import random
from pathlib import Path
from typing import Generator, List

from infinibatch.datasets import chunked_dataset_iterator
from infinibatch.iterators import BucketedReadaheadBatchIterator
from torch.utils.data import IterableDataset

from ...constants import (
    DATA_DIR,
    SAMPLE_COUNT_PER_LABEL,
    STRIDE_LENGTH,
    SUBSAMPLING_FACTOR,
    WINDOW_SIZE,
)
from ...utils import QAMDataSample
from .utils import find_label


class NCEDataset(IterableDataset):
    def __init__(
        self,
        split_name: str,
        batch_size: str,
        buffer_factor: int,
        shuffle: bool = False,
        seed: int = 7,
    ):
        self.base_dir = Path(f"{DATA_DIR}/{split_name}").resolve()
        self.batch_size = batch_size
        self.buffer_factor = buffer_factor
        self.seed = seed
        self.yielder = self.shuffled_yielder if shuffle else self.serial_yielder

        self.dist_world_size = 1
        self.dist_rank = 0
        self.num_workers = 1
        self.worker_id = 0

    def read_chunk_fn(self, path: Path) -> Generator[QAMDataSample, None, None]:
        samples: List[List] = []
        labels: List[int] = []
        sample_length = WINDOW_SIZE * SUBSAMPLING_FACTOR
        stride_length = STRIDE_LENGTH * SUBSAMPLING_FACTOR

        with gzip.open(path, "rt") as f:
            for line in f:
                sample = []
                for val in line.rstrip().split(","):
                    sample.append(float(val))

                samples.append(sample)

                if (len(samples) - SAMPLE_COUNT_PER_LABEL > 0) and (
                    (len(samples) - SAMPLE_COUNT_PER_LABEL) % SUBSAMPLING_FACTOR == 0
                ):
                    labels.append(
                        find_label(
                            samples[-(SAMPLE_COUNT_PER_LABEL + SUBSAMPLING_FACTOR) :]
                        )
                    )
                if len(labels) == WINDOW_SIZE:
                    yield QAMDataSample(
                        samples[:sample_length],
                        labels[:WINDOW_SIZE],
                    ).tensorize()

                    samples = samples[stride_length:]
                    labels = labels[STRIDE_LENGTH:]

    def shuffled_yielder(
        self, chunk_refs: List[Path]
    ) -> Generator[QAMDataSample, None, None]:
        ds_i = chunked_dataset_iterator(
            chunk_refs,
            self.read_chunk_fn,
            (self.buffer_factor * self.batch_size),
            train=False,
            shuffle=False,
            seed=self.seed,
        )
        ds_it = BucketedReadaheadBatchIterator(
            ds_i,
            batch_size=self.batch_size,
            read_ahead=(self.buffer_factor * self.batch_size),
            shuffle=True,
            seed=self.seed,
            key=lambda sample: random.randint(0, WINDOW_SIZE),
        )

        for batch in ds_it:
            yield from batch

    def serial_yielder(
        self, chunk_refs: List[Path]
    ) -> Generator[QAMDataSample, None, None]:
        for chunk_ref in chunk_refs:
            yield from self.read_chunk_fn(chunk_ref)

    def __iter__(self) -> Generator[QAMDataSample, None, None]:

        files = list(self.base_dir.glob("**/*tgz"))[self.worker_id :: self.num_workers]
        random.shuffle(files)

        self.yielder(files)
