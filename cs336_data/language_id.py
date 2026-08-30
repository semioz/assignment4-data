from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import fasttext

from cs336_data.common import get_shared_assets_path

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


@lru_cache
def _load_model():
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(MODEL_URL, model_path)
    return fasttext.load_model(str(model_path))


def identify_language(text: str) -> tuple[str, float]:
    labels, scores = _load_model().predict(text.replace("\n", " "), k=1)
    return labels[0].removeprefix("__label__"), float(scores[0])
