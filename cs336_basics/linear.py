import torch
from torch import nn
from torch import einsum
import math


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.w = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.w, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum("... i, o i -> ... o", x, self.w)
