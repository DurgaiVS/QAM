import torch
from .transformer import MHCrossAModule
from typing import Union


class TransformerDecoderModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_stacks: int,
        expansion_factor: int,
        device: Union[int, torch.device],
    ) -> None:
        super().__init__()
        # self.embedding = PositionalEncoding(embed_dim)
        self.transformer_stack = torch.nn.ModuleList(
            [
                MHCrossAModule(embed_dim, num_heads, expansion_factor, device=device)
                for _ in range(num_stacks)
            ]
        )

    def forward(self, input_seq: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        # result = self.embedding(input_seq)
        result = input_seq
        for layer in self.transformer_stack:
            result, attention = layer(result, attention)
        return result
