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
        self.w = torch.nn.Parameter(
            (
                torch.randn(out_features, in_features, device=device, dtype=dtype) * std
            ).clip(-3 * std, 3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum("... d_in, d_out d_in -> ... d_out", x, self.w)
