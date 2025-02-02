import os
from typing import List

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ...constants import DATA_DIR, RESHARD_DIR_NAME
from .dataset import QAMDataset
from .utils import collate_fn


class QAMDataModule(pl.LightningDataModule):
    def __init__(
        self, symbols: DictConfig, num_workers: int, batch_size: int, buffer_factor: int
    ):
        super().__init__()
        self.symbols = symbols
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._buffer_factor = buffer_factor

    def setup(self, stage: str) -> None:

        if stage == "fit":
            self.train_dataset = QAMDataset(
                os.path.join(DATA_DIR, RESHARD_DIR_NAME),
                "train",
                self._batch_size,
                self._buffer_factor,
                True,
            )

        if (stage == "validate") or (stage == "fit"):
            self.val_dataset = [
                QAMDataset(
                    os.path.join(DATA_DIR, s_name, RESHARD_DIR_NAME),
                    "dev",
                    self._batch_size,
                    self._buffer_factor,
                )
                for s_name, s_info in self.symbols.items()
                if (s_info is None) or ("dev" in s_info)
            ]

        if (stage == "test") or (stage == "fit") or (stage == "predict"):
            self.test_dataset = [
                QAMDataset(
                    os.path.join(DATA_DIR, s_name, RESHARD_DIR_NAME),
                    "test",
                    self._batch_size,
                    self._buffer_factor,
                )
                for s_name, s_info in self.symbols.items()
                if (s_info is None) or ("test" in s_info)
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset=ds,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            for ds in self.val_dataset
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset=ds,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            for ds in self.test_dataset
        ]

    def predict_dataloader(self) -> List[DataLoader]:
        return self.test_dataloader()

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
