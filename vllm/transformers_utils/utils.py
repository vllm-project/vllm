from pathlib import Path


def check_gguf_file(model: Path) -> bool:
    """Check if the file is a GGUF model."""
    if model.is_file():
        with open(model, "rb") as f:
            header = f.read(4)
        return header == b"GGUF"
    return False
