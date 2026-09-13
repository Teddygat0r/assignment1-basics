import importlib.metadata
from .tokenizer import train_bpe
from .tokenizer import Tokenizer
from .linear import Linear

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
