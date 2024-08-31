from os import PathLike
from pathlib import Path
from typing import Union


def check_gguf_file(model: Union[str, PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if model.is_file():
        with open(model, "rb") as f:
            header = f.read(4)
        return header == b"GGUF"
    return False
