from functools import lru_cache

import fasttext

from cs336_data.common import get_shared_assets_path


@lru_cache
def _load_quality_model():
    model_path = get_shared_assets_path() / "classifiers" / "quality_fasttext.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Train the quality model first: {model_path}")
    return fasttext.load_model(str(model_path))


def classify_quality(text: str) -> tuple[str, float]:
    labels, scores = _load_quality_model().predict(text.replace("\n", " "), k=1)
    return labels[0].removeprefix("__label__"), float(scores[0])


def passes_gopher_quality_filters(text: str) -> bool:
    words = text.split()
    if not 50 <= len(words) <= 100_000:
        return False

    mean_word_length = sum(map(len, words)) / len(words)
    if not 3 <= mean_word_length <= 10:
        return False

    lines = text.splitlines()
    if lines and sum(line.rstrip().endswith("...") for line in lines) / len(lines) > 0.3:
        return False

    return sum(any(char.isalpha() for char in word) for word in words) / len(words) >= 0.8
