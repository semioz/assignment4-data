import argparse
import gzip
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER = None


def initialize_tokenizer() -> None:
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2", model_max_length=2**31)


def tokenize_document(text: str) -> list[int]:
    return TOKENIZER.encode(text) + [TOKENIZER.eos_token_id]


def read_documents(paths: list[Path]):
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            yield from (line.strip() for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    documents = list(read_documents(args.input_paths))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with multiprocessing.Pool(args.workers, initializer=initialize_tokenizer) as pool:
        with args.output_path.open("wb") as output:
            for ids in tqdm(pool.imap(tokenize_document, documents, chunksize=100), total=len(documents)):
                ids_array = np.asarray(ids, dtype=np.uint16)
                ids_array.tofile(output)
                token_count += len(ids_array)

    print(f"Tokenized {len(documents)} documents into {token_count} tokens")


if __name__ == "__main__":
    main()
