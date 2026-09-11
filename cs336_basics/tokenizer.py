from .pretokenization_example import find_chunk_boundaries
from typing import Iterable
import regex as re
import itertools
import collections
import os
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(chunk: str, special_tokens: list[str]) -> Iterable[str]:
    pattern = "|".join([re.escape(token) for token in special_tokens])
    split_chunks = re.split(pattern, chunk) if special_tokens else [chunk]

    return (
        match.group(0) for chunk in split_chunks for match in re.finditer(PAT, chunk)
    )


def pretokenize_include_special_tokens(
    chunk: str, special_tokens: list[str]
) -> list[str]:
    if not special_tokens:
        return [chunk]

    pattern = (
        "("
        + "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
        + ")"
    )

    return [part for part in re.split(pattern, chunk) if part]


def most_frequent_item(
    cached_state: dict[tuple[int, int], int], vocabulary: list
) -> tuple[tuple[int, int], int]:
    most_frequent_item = (-1, (b"\x00", b"\x00"), (0, 0))

    for key, value in cached_state.items():
        item = (value, tuple([vocabulary[i] for i in key]), key)
        if value > 0 and item > most_frequent_item:
            most_frequent_item = item

    if most_frequent_item[0] < 0:
        raise ValueError("Cached state wrong?", cached_state)

    return most_frequent_item[2], most_frequent_item[0]


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        words = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            words.append(pretokenize(chunk, special_tokens))

        words = itertools.chain.from_iterable(words)
        token_count = 256
        vocabulary = [bytes([i]) for i in range(256)]
        merges: list[tuple[bytes, bytes]] = []

        words_tokenized = []
        word_counts = collections.Counter(words)
        cached_state = collections.defaultdict(int)
        for word, count in word_counts.items():
            word_bytes = word.encode("utf-8")
            words_tokenized.append((list(word_bytes), count))
            for i in range(len(word_bytes) - 1):
                cached_state[(int(word_bytes[i]), int(word_bytes[i + 1]))] += count

        while token_count < vocab_size - len(special_tokens):
            token_count += 1
            tokens_to_merge, _ = most_frequent_item(cached_state, vocabulary)
            n = len(tokens_to_merge)
            vocabulary.append(
                vocabulary[tokens_to_merge[0]] + vocabulary[tokens_to_merge[1]]
            )
            merges.append(
                (vocabulary[tokens_to_merge[0]], vocabulary[tokens_to_merge[1]])
            )
            new_token_index = len(vocabulary) - 1

            for tokens, count in words_tokenized:
                i = 0
                while i <= len(tokens) - n:
                    if tuple(tokens[i : i + n]) != tokens_to_merge:
                        i += 1
                        continue

                    if i > 0:
                        cached_state[(tokens[i - 1], tokens_to_merge[0])] -= count
                        cached_state[(tokens[i - 1], new_token_index)] += count

                    if i + n < len(tokens):
                        cached_state[(tokens_to_merge[-1], tokens[i + n])] -= count
                        cached_state[(new_token_index, tokens[i + n])] += count

                    tokens[i : i + n] = [new_token_index]
                    i += 1
            del cached_state[tokens_to_merge]
        vocabulary = vocabulary + [x.encode("utf-8") for x in special_tokens]
        vocabulary = {i: x for i, x in enumerate(vocabulary)}
        return vocabulary, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        self.vocab = vocab
        self.bytes_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens: list[str] = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        words = pretokenize_include_special_tokens(text, self.special_tokens)
        word_tokens: list[list[int]] = []
        for word in words:
            word_bytes = word.encode("utf-8")
            if word in self.special_tokens:
                word_tokens.append([self.bytes_to_token[word_bytes]])
                continue
            word_bytes = [bytes([i]) for i in word_bytes]
            for s1, s2 in self.merges:
                i = 0
                while i < len(word_bytes) - 1:
                    if (word_bytes[i], word_bytes[i + 1]) == (s1, s2):
                        word_bytes[i : i + 2] = [s1 + s2]
                    else:
                        i += 1

            tokens = [self.bytes_to_token[i] for i in word_bytes]
            word_tokens.append(tokens)
        return [x for xs in word_tokens for x in xs]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8")
