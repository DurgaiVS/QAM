import torch


class AutoPosEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()

        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        self.pos_embd = torch.nn.Linear(1, embed_dim)

    def forward(self, ip: torch.Tensor) -> torch.Tensor:
        # ip shape: B, SeqLen, IpDim
        _, seq_len, _ = ip.shape

        embd = self.embedding(ip)
        pos_embd = self.pos_embd(
            torch.arange(2, seq_len + 2).unsqueeze(-1).to(embd.dtype)
        )

        return embd + pos_embd
