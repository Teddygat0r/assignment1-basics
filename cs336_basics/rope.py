import torch
from torch import nn
from einops import einsum, rearrange
import math


class RotaryPositionalEmbedding(nn.Module):
    cache: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        thetas = torch.arange(max_seq_len)
        k_deltas = theta ** (-2 * torch.arange(0, d_k // 2) / d_k)
        thetas = einsum(thetas, k_deltas, "k, d -> k d")
        cos, sin = torch.cos(thetas), torch.sin(thetas)
        dim_1 = torch.stack([cos, -sin], dim=-1)
        dim_2 = torch.stack([sin, cos], dim=-1)
        self.register_buffer(
            "cache",
            torch.stack([dim_1, dim_2], dim=-1).to(device=device),
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... (a two) -> ... a two", two=2)
        cache_subset = self.cache[token_positions]
        x = einsum(x, cache_subset, "... k a b, ... k a b y -> ... k a y")
        x = rearrange(x, "... a two -> ... (a two)")

        return x
