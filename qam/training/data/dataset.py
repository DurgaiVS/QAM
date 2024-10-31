import gzip
import random
from pathlib import Path
from typing import Generator, List

from infinibatch.datasets import chunked_dataset_iterator
from infinibatch.iterators import BucketedReadaheadBatchIterator
from torch.utils.data import IterableDataset

from ...constants import DATA_DIR, MAX_SEQ_LEN, STRIDE_LENGTH, TREND_UPDATE_SEQ_LEN
from ...utils import QAMDataSample, yield_sample_from_file


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

    def shuffled_yielder(
        self, chunk_refs: List[Path]
    ) -> Generator[QAMDataSample, None, None]:
        ds_i = chunked_dataset_iterator(
            chunk_refs,
            yield_sample_from_file,
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
            yield from yield_sample_from_file(chunk_ref)

    def __iter__(self) -> Generator[QAMDataSample, None, None]:
        files = list(self.base_dir.rglob("**/*jsonl.gz"))[
            self.worker_id :: self.num_workers
        ]
        random.shuffle(files)

        self.yielder(files)
