import torch
from .position_encoder import PositionalEncoding
from .transformer import MHSelfAModule
from typing import Union


class TransformerEncoderModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_seq_length: int,
        num_heads: int,
        num_stacks: int,
        expansion_factor: int,
        device: Union[int, torch.device],
        with_pos_enc: bool = True,
    ) -> None:
        super().__init__()
        if with_pos_enc:
            self.transformer_stack = torch.nn.Sequential(
                PositionalEncoding(embed_dim, max_seq_length, device=device),
                *[
                    MHSelfAModule(embed_dim, num_heads, expansion_factor, device=device)
                    for _ in range(num_stacks)
                ]
            )
        else:
            self.transformer_stack = torch.nn.Sequential(
                *[
                    MHSelfAModule(embed_dim, num_heads, expansion_factor, device=device)
                    for _ in range(num_stacks)
                ]
            )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.transformer_stack(input_seq)
