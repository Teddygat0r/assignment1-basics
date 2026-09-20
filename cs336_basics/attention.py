import torch
from einops import einsum, rearrange
import math
from torch import nn
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding


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


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(q, "... c (a b) -> ... a c b", a=self.num_heads)
        k = rearrange(k, "... c (a b) -> ... a c b", a=self.num_heads)
        v = rearrange(v, "... c (a b) -> ... a c b", a=self.num_heads)

        dims = q.shape[:-1]

        mask = torch.tril(
            torch.ones((dims[-1], dims[-1]), dtype=torch.bool, device=self.device)
        )
        x = scaled_dot_product_attention(q, k, v, mask)
        x = rearrange(x, "... a c b -> ... c (a b)")

        x = self.w_o(x)
        return x


class MultiheadSelfAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
        )

        self.device = device
        self.dtype = dtype

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(q, "... c (a b) -> ... a c b", a=self.num_heads)
        k = rearrange(k, "... c (a b) -> ... a c b", a=self.num_heads)
        v = rearrange(v, "... c (a b) -> ... a c b", a=self.num_heads)
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, device=self.device)
            token_positions = token_positions.expand(
                *q.shape[:-2], *token_positions.shape
            )

        q, k = self.rope(q, token_positions), self.rope(k, token_positions)

        dims = q.shape[:-1]

        mask = torch.tril(
            torch.ones((dims[-1], dims[-1]), dtype=torch.bool, device=self.device)
        )
        x = scaled_dot_product_attention(q, k, v, mask)
        x = rearrange(x, "... a c b -> ... c (a b)")

        x = self.w_o(x)
        return x
