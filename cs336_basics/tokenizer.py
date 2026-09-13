from functools import cache, lru_cache

from .pretokenization_example import find_chunk_boundaries
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor
import regex as re
import itertools
import collections
import os
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(chunk: str, special_tokens: list[str]) -> Iterable[str]:
    pattern = "|".join([re.escape(token) for token in special_tokens])
    split_chunks = re.split(pattern, chunk) if special_tokens else [chunk]

    for c in split_chunks:
        for match in re.finditer(PAT, c):
            yield match.group(0)


def pretokenize_include_special_tokens(
    chunk: str, special_tokens: list[str]
) -> Iterable[str]:
    if not special_tokens:
        yield chunk
        return

    special_pattern = "|".join(
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    )

    last_end = 0

    for match in re.finditer(special_pattern, chunk):
        if match.start() > last_end:
            yield chunk[last_end : match.start()]

        yield match.group(0)
        last_end = match.end()

    if last_end < len(chunk):
        yield chunk[last_end:]


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


def _process_chunk(
    file_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> collections.Counter[str]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = collections.Counter()
    pattern = "|".join([re.escape(token) for token in special_tokens])
    split_chunks = re.split(pattern, chunk) if special_tokens else [chunk]

    for c in split_chunks:
        for match in re.finditer(PAT, c):
            counts[match.group(0)] += 1

    return counts


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        # Use first special token if present, fallback to default delimiter
        delimiter = (
            special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
        )
        boundaries = find_chunk_boundaries(f, num_processes, delimiter)

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #     words.append(pretokenize(chunk, special_tokens))
    word_counts = collections.Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(_process_chunk, *args) for args in chunk_args]
        for future in futures:
            word_counts.update(future.result())

    vocabulary = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []

    words_tokenized: list[tuple[list[int], int]] = []
    cached_state = collections.defaultdict(int)

    # Inverted index: (pair) -> set of word indices that contain it
    pair_to_words = collections.defaultdict(set)

    for word_idx, (word, count) in enumerate(word_counts.items()):
        token_ids = list(word.encode("utf-8"))
        words_tokenized.append((token_ids, count))
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            cached_state[pair] += count
            pair_to_words[pair].add(word_idx)

    target_vocab_size = vocab_size - len(special_tokens)

    while len(vocabulary) < target_vocab_size:
        if not cached_state:
            break

        best_item = (-1, (b"\x00", b"\x00"), (0, 0))
        for key, count in cached_state.items():
            if count > 0:
                cand = (count, (vocabulary[key[0]], vocabulary[key[1]]), key)
                if cand > best_item:
                    best_item = cand

        if best_item[0] <= 0:
            break

        pair_to_merge = best_item[2]
        p0, p1 = pair_to_merge
        new_token_id = len(vocabulary)

        vocabulary.append(vocabulary[p0] + vocabulary[p1])
        merges.append((vocabulary[p0], vocabulary[p1]))

        # Grab only the words containing this pair
        affected_word_indices = pair_to_words.pop(pair_to_merge, set())
        del cached_state[pair_to_merge]

        for word_idx in affected_word_indices:
            tokens, count = words_tokenized[word_idx]

            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                cached_state[p] -= count
                if p in pair_to_words:
                    pair_to_words[p].discard(word_idx)

            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                cached_state[p] += count
                pair_to_words[p].add(word_idx)

            words_tokenized[word_idx] = (new_tokens, count)

    vocabulary.extend(x.encode("utf-8") for x in special_tokens)
    vocab_dict = {i: b for i, b in enumerate(vocabulary)}
    return vocab_dict, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        self.vocab = vocab
        self.bytes_to_token = {v: k for k, v in vocab.items()}
        self.merges = {x: i for i, x in enumerate(merges)}
        self.merge_count = len(merges)
        self.special_tokens: list[str] = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    def _bpe(self, input: str) -> list[bytes]:
        parts = [bytes([i]) for i in input.encode("utf-8")]
        while len(parts) >= 2:
            merge_ind = None
            earliest_ind = self.merge_count
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merges and earliest_ind > self.merges[pair]:
                    merge_ind = i
                    earliest_ind = self.merges[pair]

            if merge_ind is None:
                break

            parts[merge_ind : merge_ind + 2] = [parts[merge_ind] + parts[merge_ind + 1]]

        return parts

    def encode(self, text: str) -> list[int]:
        words = pretokenize_include_special_tokens(text, self.special_tokens)
        word_tokens: list[int] = []
        for word in words:
            if word in self.special_tokens:
                word_bytes = word.encode("utf-8")
                word_tokens.append(self.bytes_to_token[word_bytes])
                continue
            for match in re.finditer(PAT, word):
                word_found = match.group(0)
                word_bytes = self._bpe(word_found)

                tokens = [self.bytes_to_token[i] for i in word_bytes]
                word_tokens.extend(tokens)
        return word_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode(
            "utf-8", errors="replace"
        )
