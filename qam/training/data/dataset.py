import gzip
import random
from pathlib import Path
from typing import Generator, List

from infinibatch.datasets import chunked_dataset_iterator
from infinibatch.iterators import BucketedReadaheadBatchIterator
from torch.utils.data import IterableDataset

from ...constants import MAX_SEQ_LEN
from ...utils import QAMDataSample, yield_sample_from_file


class QAMDataset(IterableDataset):
    def __init__(
        self,
        base_dir: str,
        split_name: str,
        batch_size: str,
        buffer_factor: int,
        shuffle: bool = False,
        seed: int = 7,
    ):
        super().__init__()
        self.base_dir = Path(base_dir).resolve()
        self.split = split_name
        self.batch_size = batch_size
        self.buffer_factor = buffer_factor
        self.seed = seed
        self.yielder = self.shuffled_yielder if shuffle else self.serial_yielder

        self.dist_world_size = 1
        self.dist_rank = 0
        self.num_workers = 1
        self.worker_id = 0

    def read_chunk_fn(self, filepath: Path) -> Generator[QAMDataSample, None, None]:
        with gzip.open(filepath, "rt") as f:
            for line in f:
                yield QAMDataSample.from_str(line)

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
            key=lambda sample: random.randint(0, MAX_SEQ_LEN),
        )

        for batch in ds_it:
            yield from batch

    def serial_yielder(
        self, chunk_refs: List[Path]
    ) -> Generator[QAMDataSample, None, None]:
        for chunk_ref in chunk_refs:
            yield from self.read_chunk_fn(chunk_ref)

    def __iter__(self) -> Generator[QAMDataSample, None, None]:
        # NOTE: Serially written Timepoint shards is rewritten as
        # QAMDataSample shards shuffled. So any worker can take any
        # shards and read in any random order...

        files = list(self.base_dir.rglob(f"**/{self.split}*jsonl.gz"))[
            self.worker_id :: self.num_workers
        ]
        random.shuffle(files)

        self.yielder(files)
