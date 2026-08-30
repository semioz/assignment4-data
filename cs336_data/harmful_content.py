from functools import lru_cache
from urllib.request import urlretrieve

import fasttext

from cs336_data.common import get_shared_assets_path

NSFW_MODEL_URL = "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
HATESPEECH_MODEL_URL = "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"

@lru_cache
def _load_nsfw_model():
    model_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(NSFW_MODEL_URL, model_path)
    return fasttext.load_model(str(model_path))


def classify_nsfw(text: str) -> tuple[str, float]:
    labels, scores = _load_nsfw_model().predict(text.replace("\n", " "), k=1)
    return labels[0].removeprefix("__label__"), float(scores[0])


@lru_cache
def _load_hatespeech_model():
    model_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(HATESPEECH_MODEL_URL, model_path)
    return fasttext.load_model(str(model_path))


def classify_toxic_speech(text: str) -> tuple[str, float]:
    labels, scores = _load_hatespeech_model().predict(text.replace("\n", " "), k=1)
    return labels[0].removeprefix("__label__"), float(scores[0])
