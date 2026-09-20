import importlib.metadata
from .tokenizer import train_bpe
from .tokenizer import Tokenizer
from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU, SiLU
from .rope import RotaryPositionalEmbedding
from .attention import (
    softmax,
    scaled_dot_product_attention,
    MultiheadSelfAttention,
    MultiheadSelfAttentionWithRope,
)
from .transformer_lm import TransformerBlock, Transformer

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
