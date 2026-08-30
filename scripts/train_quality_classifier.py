import gzip
import random
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.language_id import identify_language
from cs336_data.quality import passes_gopher_quality_filters

POSITIVE_WARC = Path("local-shared-data/wiki/subsampled_positive_urls.warc.gz")
NEGATIVE_WARC = Path("example.warc.gz")
TRAINING_PATH = Path("local-shared-data/quality/quality_train.txt")
MODEL_PATH = Path("local-shared-data/classifiers/quality_fasttext.bin")


def eligible_texts(path: Path):
    with gzip.open(path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response, parse_http=True):
            if record.http_headers.status_code != 200:
                continue
            if not (record.http_content_type or "").startswith("text/html"):
                continue
            text = extract_text_from_html_bytes(record.reader.read())
            if not passes_gopher_quality_filters(text):
                continue
            language, confidence = identify_language(text)
            if language == "en" and confidence >= 0.7:
                yield " ".join(text.split())[:10_000]


def reservoir_sample(texts, count: int) -> list[str]:
    rng = random.Random(336)
    sample = []
    for index, text in enumerate(texts):
        if index < count:
            sample.append(text)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < count:
                sample[replacement] = text
    return sample


def main() -> None:
    positives = list(eligible_texts(POSITIVE_WARC))
    negatives = reservoir_sample(eligible_texts(NEGATIVE_WARC), len(positives))
    if len(negatives) < len(positives):
        raise RuntimeError("Not enough eligible Common Crawl examples")

    TRAINING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_PATH.open("w") as f:
        for text in positives:
            f.write(f"__label__wiki {text}\n")
        for text in negatives:
            f.write(f"__label__cc {text}\n")

    model = fasttext.train_supervised(
        input=str(TRAINING_PATH), epoch=25, lr=0.5, wordNgrams=2, dim=32, bucket=50_000
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Trained on {len(positives)} examples per class: {MODEL_PATH}")


if __name__ == "__main__":
    main()
