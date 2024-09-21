from typing import Union

import torch


class ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
        downsampling_factor: int = 1,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=downsampling_factor,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            (
                torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
                if use_group_norm
                else torch.nn.BatchNorm1d(num_channels)
            ),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


# own implementation ...
class _ConvolutionModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        pad_size: int,
        device: Union[int, torch.device],
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(embed_dim, device=device)
        self.linear = torch.nn.Linear(embed_dim, embed_dim * 2, device=device)
        self.conv_mod = torch.nn.Sequential(
            torch.nn.Conv1d(embed_dim * 2, embed_dim * 2, 1, device=device),
            torch.nn.GLU(1),
            torch.nn.Conv1d(
                embed_dim, embed_dim, kernel_size, padding=pad_size, device=device
            ),
            torch.nn.BatchNorm1d(embed_dim, device=device),
            torch.nn.SiLU(),
            torch.nn.Conv1d(embed_dim, embed_dim, 1, device=device),
            torch.nn.Dropout1d(0.3),
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        res = self.norm(input_seq)
        res = self.linear(res)
        return self.conv_mod(res.transpose(1, 2)).transpose(1, 2)
