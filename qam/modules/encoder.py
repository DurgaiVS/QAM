import math

import torch
from omegaconf import DictConfig

from ..constants import SAMPLE_DIM, SUBSAMPLING_FACTOR, WINDOW_SIZE
from ..utils import Classifier
from .conformer import ConformerLayer
from .position_encoder import PositionalEncoding


class ConfEncoderWithClassificationHeads(torch.nn.Module):
    def __init__(
        self,
        encoder: DictConfig,
        classification_layer: DictConfig,
        input_dim: int = SAMPLE_DIM,
        max_seq_len: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoder.input_dim),
            PositionalEncoding(encoder.input_dim, max_seq_len),
        )

        down_samplers_count = int(math.log2(SUBSAMPLING_FACTOR))
        layers = []
        encoder_params = dict(encoder)
        num_layers = encoder_params.pop("num_layers") - down_samplers_count

        for _ in range(down_samplers_count):
            layers.append(ConformerLayer(**encoder_params, downsampling_factor=2))
        for _ in range(num_layers):
            layers.append(ConformerLayer(**encoder_params))

        self.conformer_layers = torch.nn.ModuleList(layers)

        self.classification_layer = torch.nn.Sequential(
            *(
                [torch.nn.Dropout(classification_layer.dropout)]
                if classification_layer.dropout > 0
                else []
            ),
            *(
                [getattr(torch.nn.functional, classification_layer.activation)]
                if classification_layer.activation
                else []
            ),
            torch.nn.Linear(encoder.input_dim, len(Classifier.__members__)),
        )
        self.classification_head_count: int = len(Classifier.__members__)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.embedding(inputs)
        for layer in self.conformer_layers:
            result, lengths = layer(result, lengths)

        return self.classification_layer(result), lengths
