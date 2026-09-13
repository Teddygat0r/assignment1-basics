import torch
from torch import nn
from einops import einsum
import math


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = math.sqrt(2.0 / (d_model + d_ff))
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.silu = SiLU()

        nn.init.trunc_normal_(self.w2, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(x, self.w1, "... i, o i -> ... o")
        silu = self.silu(w1x)
        glu = silu * einsum(x, self.w3, "... i, o i -> ... o")

        return einsum(glu, self.w2, "... i, o i -> ... o")
