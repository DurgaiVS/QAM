from typing import Optional, Union

import torch

from .convolution import ConvolutionModule, _ConvolutionModule
from .feedforward import FeedForwardModule, _FeedForwardModule
from .transformer import MHSelfAModule


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
        output_dim (int): output dimension.
        NOTE: if output_dim is provided, then ff2 and layernorm won't be used. Assuming it was a pooler layer at the end of the model...

    NOTE:
        Modified this class from torchaudio.models.conformer to downsample
        input sequence as specified amount. Tried creating a FastConformer
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_heads: int,
        depthwise_conv_kernel_size: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        downsampling_factor: int = 1,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        # bool added intentionally for downsampling, as residual cannot be applied
        self.downsampler = False if downsampling_factor == 1 else True
        self.pooler = True if output_dim is not None else False
        self.conv_module = ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            output_dim=output_dim,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
            downsampling_factor=downsampling_factor,
        )

        self.convolution_first = convolution_first
        if not output_dim:
            self.ffn2 = FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
            self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        # input = input.transpose(0, 1)
        input = self.conv_module(input)
        # input = input.transpose(0, 1)
        if not (self.downsampler or self.pooler):
            input = residual + input
        return input

    # TODO: change the key_padding_mask to attention_mask, which is most needed...
    def forward(
        self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(B, T, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        if self.pooler:
            return x, (
                torch.ceil(key_padding_mask / 2)
                if self.downsampler and (key_padding_mask is not None)
                else key_padding_mask
            )

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, (
            torch.ceil(key_padding_mask / 2)
            if self.downsampler and (key_padding_mask is not None)
            else key_padding_mask
        )


# own implementation ...
class _ConformerModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int,
        pad_size: int,
        device: Union[int, torch.device],
    ) -> None:
        super().__init__()

        self.conf = torch.nn.Sequential(
            _FeedForwardModule(embed_dim, device=device),
            MHSelfAModule(embed_dim, num_heads, device=device),
            _ConvolutionModule(embed_dim, kernel_size, pad_size, device=device),
            _FeedForwardModule(embed_dim, device=device),
            torch.nn.LayerNorm(embed_dim, device=device),
        )

    def forward(self, input_seq) -> torch.Tensor:
        return self.conf(input_seq)
