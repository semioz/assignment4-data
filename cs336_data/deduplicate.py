from collections import Counter, defaultdict
from hashlib import blake2b
from itertools import combinations
from pathlib import Path
import random
import unicodedata

import mmh3
from xopen import xopen


def _line_hash(line: str) -> bytes:
    return blake2b(line.encode(), digest_size=8).digest()


def exact_line_deduplication(input_files: list[Path], output_directory: Path) -> None:
    counts = Counter()
    for input_file in input_files:
        with xopen(input_file, "rt") as f:
            counts.update(_line_hash(line) for line in f)

    output_directory.mkdir(parents=True, exist_ok=True)
    for input_file in input_files:
        output_path = output_directory / Path(input_file).name
        with xopen(input_file, "rt") as source, xopen(output_path, "wt") as output:
            for line in source:
                if counts[_line_hash(line)] == 1:
                    output.write(line)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text).lower()
    return " ".join(
        "".join(char if char.isalnum() or char.isspace() else " " for char in text).split()
    )


def _ngrams(text: str, ngrams: int) -> set[str]:
    words = _normalize(text).split()
    if len(words) < ngrams:
        return {" ".join(words)}
    return {" ".join(words[index : index + ngrams]) for index in range(len(words) - ngrams + 1)}


def _minhash_signature(ngrams: set[str], num_hashes: int) -> tuple[int, ...]:
    return tuple(
        min(mmh3.hash64(ngram, seed=seed, signed=False)[0] for ngram in ngrams)
        for seed in range(num_hashes)
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def minhash_deduplication(
    input_files: list[Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Path,
) -> None:
    if num_hashes % num_bands:
        raise ValueError("num_hashes must be divisible by num_bands")

    texts = []
    shingles = []
    for input_file in input_files:
        with xopen(input_file, "rt") as f:
            text = f.read()
        texts.append(text)
        shingles.append(_ngrams(text, ngrams))

    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(list)
    for index, shingle_set in enumerate(shingles):
        signature = _minhash_signature(shingle_set, num_hashes)
        for band in range(num_bands):
            start = band * rows_per_band
            buckets[band, signature[start : start + rows_per_band]].append(index)

    parent = list(range(len(input_files)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for bucket in buckets.values():
        for left, right in combinations(bucket, 2):
            if _jaccard(shingles[left], shingles[right]) >= jaccard_threshold:
                union(left, right)

    clusters = defaultdict(list)
    for index in range(len(input_files)):
        clusters[find(index)].append(index)
    rng = random.Random(336)
    kept = {rng.choice(cluster) for cluster in clusters.values()}

    output_directory.mkdir(parents=True, exist_ok=True)
    for index in kept:
        output_path = output_directory / Path(input_files[index]).name
        with xopen(output_path, "wt") as f:
            f.write(texts[index])
