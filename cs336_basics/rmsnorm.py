import torch
from torch import nn
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1) + self.eps)
        x = x / rearrange(rms, "... -> ... 1") * self.g

        return x.to(dtype=in_dtype)
