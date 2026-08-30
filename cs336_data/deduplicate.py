from collections import Counter
from hashlib import blake2b
from pathlib import Path

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
        with xopen(input_file, "rt") as source, xopen(output_directory / Path(input_file).name, "wt") as output:
            for line in source:
                if counts[_line_hash(line)] == 1:
                    output.write(line)
