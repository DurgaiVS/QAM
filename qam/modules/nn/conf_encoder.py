from typing import Union

import torch

from .conformer import _ConformerModule


class ConformerEncoderModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_stacks: int,
        kernel_size: int,
        device: Union[int, torch.device],
    ) -> None:
        super().__init__()
        pad_size = kernel_size // 2

        self.embedding = torch.nn.Conv1d(1, embed_dim, kernel_size, device=device)
        """
        Since the convolution module itself encode positional information withing the embedding,
        neglecting the use of positional_encoding

        >>> import torch
        >>> cc = torch.nn.Conv1d(1, 4, 5)
        >>> val = torch.randn(1, 14)
        >>> val_sort, _ = val.sort()
        >>> cc(val) == cc(val_sort)
        tensor([[False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False]])
        """

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim, device=device),
            *[
                _ConformerModule(
                    embed_dim, num_heads, kernel_size, pad_size, device=device
                )
                for _ in range(num_stacks)
            ],
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        result = self.embedding(input_seq).transpose(1, 2)
        return self.encoder(result)
