from typing import Union

import torch

from .feedforward import _FeedForwardModule


class MHSelfAModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion_factor: int,
        device: Union[int, torch.device],
    ) -> None:
        super().__init__()

        self.query = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.key = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.value = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.mha = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            add_bias_kv=True,
            kdim=embed_dim,
            vdim=embed_dim,
            device=device,
        )
        self.ff = _FeedForwardModule(embed_dim, expansion_factor, device=device)
        self.norm = torch.nn.LayerNorm(embed_dim, device=device)

        # cos_pos = repeat_elements(sinusoidal_pos[..., None, 1::2], rep=2, axis=-1)
        # sin_pos = repeat_elements(sinusoidal_pos[..., None, ::2], rep=2, axis=-1)
        # qw2 = stack([-qw[..., 1::2], qw[..., ::2]], 4)
        # qw2 = reshape(qw2, shape(qw))
        # qw = qw * cos_pos + qw2 * sin_pos
        # kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
        # kw2 = K.reshape(kw2, K.shape(kw))
        # kw = kw * cos_pos + kw2 * sin_pos

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        q = self.query(input_seq)
        k = self.key(input_seq)
        v = self.value(input_seq)
        tmp, _ = self.mha(q, k, v)
        tmp = self.norm(tmp + input_seq)

        tmp2 = self.ff(tmp)
        tmp2 = self.norm(tmp + tmp2)

        return tmp2


class MHCrossAModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion_factor: int,
        device: Union[int, torch.device],
    ) -> None:
        super().__init__()

        self.query = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.key = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.value = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.mha = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            add_bias_kv=True,
            kdim=embed_dim,
            vdim=embed_dim,
            device=device,
        )
        self.ff = _FeedForwardModule(embed_dim, expansion_factor, device=device)
        self.norm = torch.nn.LayerNorm(embed_dim, device=device)

    def forward(
        self, input_seq: torch.Tensor, encoded_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(input_seq)
        k = self.key(encoded_seq)
        v = self.value(encoded_seq)
        tmp, _ = self.mha(q, k, v)
        tmp = self.norm(tmp + input_seq)

        tmp2 = self.ff(tmp)
        tmp2 = self.norm(tmp + tmp2)

        return tmp2, encoded_seq
