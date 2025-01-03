from os import PathLike
from pathlib import Path
from typing import Union


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith('s3://')


def check_gguf_file(model: Union[str, PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
