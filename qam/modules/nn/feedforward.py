from typing import Union

import torch


class FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


# own implementation ...
class _FeedForwardModule(torch.nn.Module):
    def __init__(
        self, embed_dim: int, expansion_factor: int, device: Union[int, torch.device]
    ) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(
            embed_dim, embed_dim * expansion_factor, device=device
        )
        self.linear2 = torch.nn.Linear(
            embed_dim * expansion_factor, embed_dim, device=device
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        result = self.linear1(input_seq)
        result = torch.nn.functional.gelu(result)
        return self.linear2(result)
