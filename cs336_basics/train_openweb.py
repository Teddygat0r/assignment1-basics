import pickle
import resource
import sys
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe

if __name__ == "__main__":
    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak_rss / (1024**2 if sys.platform == "darwin" else 1024)

    output_dir = Path("artifacts_full")
    output_dir.mkdir(exist_ok=True)

    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Peak process memory: {peak_mib:.2f} MiB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Longest token ID: {longest_id}")
    print(f"Longest token ({len(longest_token)} bytes): {longest_token!r}")
    print(f"Decoded: {longest_token.decode('utf-8', errors='replace')!r}")
