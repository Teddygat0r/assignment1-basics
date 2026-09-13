import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
PILE_BYTES = 825_000_000_000  # 825 GB, decimal

DATASETS = {
    "tinystories": {
        "train": Path("data/TinyStoriesV2-GPT4-train.txt"),
        "dev": Path("data/TinyStoriesV2-GPT4-valid.txt"),
        "vocab": Path("artifacts_tiny/vocab.pkl"),
        "merges": Path("artifacts_tiny/merges.pkl"),
    },
    "openwebtext": {
        "train": Path("data/owt_train.txt"),
        "dev": Path("data/owt_valid.txt"),
        "vocab": Path("artifacts_full/vocab.pkl"),
        "merges": Path("artifacts_full/merges.pkl"),
    },
}


def iter_documents(
    path: str | Path,
    delimiter: str = EOT,
    chunk_size: int = 1 << 20,
):
    """Stream documents without loading the entire dataset into memory."""
    remainder = ""

    with open(path, encoding="utf-8") as file:
        while chunk := file.read(chunk_size):
            remainder += chunk
            pieces = remainder.split(delimiter)
            remainder = pieces.pop()

            yield from pieces

    if remainder:
        yield remainder


def sample_documents(
    path: str | Path,
    n: int = 10,
    seed: int = 0,
) -> list[str]:
    """Uniform reservoir sample of documents."""
    rng = random.Random(seed)
    sample: list[str] = []

    for index, document in enumerate(iter_documents(path)):
        if index < n:
            sample.append(document)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < n:
                sample[replacement] = document

    if len(sample) < n:
        raise ValueError(f"{path} contained only {len(sample)} documents")

    return sample


def benchmark(
    tokenizer: Tokenizer,
    documents: list[str],
) -> tuple[float, float]:
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)

    start = time.perf_counter()
    token_ids = [
        token_id for document in documents for token_id in tokenizer.encode(document)
    ]
    elapsed = time.perf_counter() - start

    bytes_per_token = total_bytes / len(token_ids)
    bytes_per_second = total_bytes / elapsed
    return bytes_per_token, bytes_per_second


def serialize_dataset(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Write raw uint16 token IDs incrementally.

    Load later using:
        tokens = np.fromfile(output_path, dtype=np.uint16)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as output_file:
        for document in iter_documents(input_path):
            # Restore the separator removed by iter_documents().
            ids = tokenizer.encode(document + EOT)
            np.asarray(ids, dtype=np.uint16).tofile(output_file)


def main() -> None:
    ts = DATASETS["tinystories"]
    owt = DATASETS["openwebtext"]

    ts_tokenizer = Tokenizer.from_files(ts["vocab"], ts["merges"], special_tokens=[EOT])
    owt_tokenizer = Tokenizer.from_files(
        owt["vocab"], owt["merges"], special_tokens=[EOT]
    )

    ts_sample = sample_documents(ts["train"], seed=0)
    owt_sample = sample_documents(owt["train"], seed=0)

    ts_ratio, ts_speed = benchmark(ts_tokenizer, ts_sample)
    owt_ratio, owt_speed = benchmark(owt_tokenizer, owt_sample)
    cross_ratio, _ = benchmark(ts_tokenizer, owt_sample)

    print(f"TinyStories tokenizer: {ts_ratio:.3f} bytes/token")
    print(f"OpenWebText tokenizer: {owt_ratio:.3f} bytes/token")
    print("TinyStories tokenizer on OpenWebText: " f"{cross_ratio:.3f} bytes/token")

    average_speed = (ts_speed + owt_speed) / 2
    pile_days = PILE_BYTES / average_speed / 86_400

    print(f"TinyStories throughput: {ts_speed:,.0f} bytes/s")
    print(f"OpenWebText throughput: {owt_speed:,.0f} bytes/s")
    print(f"Estimated Pile tokenization time: {pile_days:,.1f} days")

    serialize_dataset(ts_tokenizer, ts["train"], "tokenized/tinystories_train.bin")
    serialize_dataset(ts_tokenizer, ts["dev"], "tokenized/tinystories_dev.bin")
    serialize_dataset(owt_tokenizer, owt["train"], "tokenized/owt_train.bin")
    serialize_dataset(owt_tokenizer, owt["dev"], "tokenized/owt_dev.bin")


if __name__ == "__main__":
    main()
