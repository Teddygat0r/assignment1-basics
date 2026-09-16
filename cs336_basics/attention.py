import torch
from einops import einsum
import math


def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.size()[-1]
    x = einsum(q, k, "... a b, ... c b -> ... a c") / math.sqrt(d_k)
    if mask is not None:
        x = torch.where(mask, x, -torch.inf)

    x = softmax(x)
    return einsum(x, v, "... a b, ... b c -> ... a c")
