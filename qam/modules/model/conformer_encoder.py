import math
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from ...constants import MAX_SEQ_LEN, SAMPLE_DIM, SUBSAMPLING_FACTOR
from ...utils import Classifier
from ..nn.conformer import ConformerLayer
from ..pos_enc.auto_pos_enc import AutoPosEncoder


class ConfEncoderWithClassificationHeads(torch.nn.Module):
    def __init__(
        self,
        encoder: DictConfig,
        classification_layer: DictConfig,
        input_dim: int = SAMPLE_DIM,
        seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()

        self.embedding = AutoPosEncoder(input_dim, encoder.input_dim)
        self.seq_len = seq_len

        down_samplers_count = int(math.log2(SUBSAMPLING_FACTOR))
        layers = []
        encoder_params = dict(encoder)
        num_layers = encoder_params.pop("num_layers") - down_samplers_count

        for _ in range(down_samplers_count):
            layers.append(ConformerLayer(**encoder_params, downsampling_factor=2))
        for _ in range(num_layers):
            layers.append(ConformerLayer(**encoder_params))

        # pooler layer, which pools from (B, S, D) -> (B, D)
        # which will be applicable for the single output layer
        layers.append(ConformerLayer(**encoder_params, output_dim=1))

        self.conformer_layers = torch.nn.ModuleList(layers)

        self.classification_layer = torch.nn.Sequential(
            *([hydra.utils.instantiate(pre_s) for pre_s in classification_layer]),
            torch.nn.Linear(encoder.input_dim, len(Classifier.__members__)),
        )
        self.classification_head_count: int = len(Classifier.__members__)

    def forward(
        self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len, _ = inputs.shape
        if seq_len != self.seq_len:
            raise RuntimeError(
                f"Expected sequence length is {self.seq_len}, but got an input with seq len {seq_len}. Try padding the input."
            )

        result = self.embedding(inputs)
        for layer in self.conformer_layers:
            result, lengths = layer(result, lengths)

        return self.classification_layer(result.squeeze(1)), lengths
