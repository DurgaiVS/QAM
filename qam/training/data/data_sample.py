from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union

import torch


@dataclass
class QAMDataSample:
    dataset_name: str
    frame: Union[list[list], torch.Tensor]
    label: Union[list[list], torch.Tensor]

    def tensorize(self) -> "QAMDataSample":
        if not isinstance(self.frame, torch.Tensor):
            self.frame = torch.stack(self.frame).to(torch.float32)
            self.label = torch.tensor(self.label).to(torch.long)
        return self

    def to(self, *args, **kwargs) -> "QAMDataSample":
        self.frame.to(*args, **kwargs)
        self.label.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs) -> "QAMDataSample":
        self.frame.cuda(*args, **kwargs)
        self.label.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs) -> "QAMDataSample":
        self.frame.cpu(*args, **kwargs)
        self.label.cpu(*args, **kwargs)
        return self

    def __len__(self) -> int:
        return len(self.frame)


@dataclass
class QAMDataBatch:
    dataset_names: list[str] = field(default_factory=list)
    frames: Union[torch.Tensor, list[torch.Tensor]] = field(default_factory=list)
    labels: Union[torch.Tensor, list[torch.Tensor]] = field(default_factory=list)

    def tensorize(self) -> "QAMDataBatch":
        if not isinstance(self.frames, torch.Tensor):
            self.frames = torch.stack(self.frames).to(torch.float32)
            self.labels = torch.stack(self.labels).to(torch.long)
        return self

    def to(self, device) -> "QAMDataBatch":
        self.frames.to(device)
        self.labels.to(device)
        return self

    def cuda(self, device) -> "QAMDataBatch":
        self.frames.cuda(device)
        self.labels.cuda(device)
        return self

    def cpu(self) -> "QAMDataBatch":
        self.frames.cpu()
        self.labels.cpu()
        return self

    def __len__(self) -> int:
        return len(self.frames)

    def extend(self, samples: list[QAMDataSample]) -> "QAMDataBatch":
        for sample in samples:
            self.frames.append(sample.frame)
            self.labels.append(sample.label)
        return self

    def __getitem__(self, index: int) -> QAMDataSample:
        return QAMDataSample(
            self.dataset_names[index],
            self.frames[index],
            self.labels[index],
        )

    @classmethod
    def from_list(cls, samples: list[QAMDataSample]) -> "QAMDataBatch":
        self = cls()

        for sample in samples:
            self.dataset_names.append(sample.dataset_name)
            self.frames.append(sample.frame)
            self.labels.append(sample.label)

        return self.tensorize()


class Classifier(Enum):
    VERY_HIGH: int = 0
    HIGH: int = auto()
    LOW: int = auto()
    VERY_LOW: int = auto()
