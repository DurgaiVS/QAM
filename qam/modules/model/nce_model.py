from typing import Union

import torch

from ...utils import QAMDataBatch
from ..nn.transf_encoder import TransformerEncoderModule


class NCEModel(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_stacks: int,
        expansion_factor: int,
        device: Union[int, torch.device, str] = torch.device("cpu"),
    ):
        self.encoder = TransformerEncoderModule(
            embed_dim, num_heads, 2, expansion_factor, device=device
        )
        self.decoder = TransformerEncoderModule(
            embed_dim,
            num_heads,
            num_stacks,
            expansion_factor,
            device=device,
            with_pos_enc=False,
        )

    def forward(self, input_batch: QAMDataBatch) -> torch.Tensor:
        pass
