import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ...constants import DATA_DIR
from .dataset import NCEDataset
from .utils import collate_fn, worker_init_fn


class NCEDataModule(pl.LightningDataModule):
    def __init__(self, num_workers: int, batch_size: int, buffer_factor: int):
        super().__init__()
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._buffer_factor = buffer_factor

    def setup(self, stage: str) -> None:
        self.train_dataset = NCEDataset(DATA_DIR, "train", self._batch_size, True)
        self.val_dataset = NCEDataset(DATA_DIR, "val", self._batch_size)
        self.test_dataset = NCEDataset(DATA_DIR, "test", self._batch_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
