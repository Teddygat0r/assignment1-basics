from .pretokenization_example import find_chunk_boundaries
from typing import Iterable
import regex as re
import itertools
import collections
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# Should be stripped of special tokens already?
def pretokenize(chunk: str, special_tokens: list[str]) -> Iterable[str]:
    pattern = "|".join([re.escape(token) for token in special_tokens])
    split_chunks = re.split(pattern, chunk)

    return (
        match.group(0) for chunk in split_chunks for match in re.finditer(PAT, chunk)
    )


def most_frequent_item(
    cached_state: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], int]:
    most_frequent_item = (-1, (0, 0))

    for key, value in cached_state.items():
        if value > 0 and (value, key) > most_frequent_item:
            most_frequent_item = (value, key)

    return most_frequent_item[1], most_frequent_item[0]


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        words_original = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            words_original.append(pretokenize(chunk, special_tokens))

        words_original = itertools.chain.from_iterable(words_original)
        token_count = 256
        vocabulary = [bytes([i]) for i in range(256)]
        merges: list[tuple[bytes, bytes]] = []

        words_original, words = itertools.tee(words_original)
        words_tokenized = []
        cached_state = collections.defaultdict(int)
        for word in words:
            word_bytes = word.encode("utf-8")
            words_tokenized.append(list(word_bytes))
            for i in range(len(word_bytes) - 1):
                cached_state[(word_bytes[i], word_bytes[i + 1])] += 1

        while token_count < vocab_size - len(special_tokens):
            token_count += 1
            tokens_to_merge, _ = most_frequent_item(cached_state)

            vocabulary.append(
                vocabulary[tokens_to_merge[0]] + vocabulary[tokens_to_merge[1]]
            )
            merges.append(
                (vocabulary[tokens_to_merge[0]], vocabulary[tokens_to_merge[1]])
            )
            new_token_index = len(vocabulary) - 1

            del cached_state[tokens_to_merge]
            for tokens in words_tokenized:
                for i in range(len(tokens) - len(tokens_to_merge)):
                    if tokens[i : i + len(tokens_to_merge)] != tokens_to_merge:
                        continue
                    if i > 0:
                        cached_state[(tokens[i - 1], tokens_to_merge[0])] -= 1
                        cached_state[(tokens[i - 1], new_token_index)] += 1
                    if i + len(tokens_to_merge) < len(tokens):
                        cached_state[
                            (tokens_to_merge[1], tokens[i + len(tokens_to_merge)])
                        ] -= 1
                        cached_state[
                            (new_token_index, tokens[i + len(tokens_to_merge)])
                        ] += 1

        vocabulary = {i: x for i, x in enumerate(vocabulary)}
        return vocabulary, merges


# class Tokenizer:
#     def __init__(self, vocab, merges, special_tokens=None):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens

#     @classmethod
#     def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#         pass

#     def encode(self, text: str) -> list[int]:
#         pass

#     def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
#         pass

#     def decode(self, ids: list[int]):
#         pass
