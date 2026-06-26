# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import regex as re
from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

BLINKDL_RWKV7_G1_REPO = "BlinkDL/rwkv7-g1"

_RWKV7_G1_SPECS = {
    "0.1b": (12, 768),
    "0.4b": (24, 1024),
    "1.5b": (24, 2048),
    "2.9b": (32, 2560),
    "7.2b": (32, 4096),
    "13.3b": (61, 4096),
}

_RWKV7_G1_FILENAME_RE = re.compile(
    r"^rwkv7-g1[a-z]-"
    r"(?P<size>0\.1b|0\.4b|1\.5b|2\.9b|7\.2b|13\.3b)-"
    r"(?P<date>\d{8})-ctx(?P<context>\d+)\.pth$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RWKV7PthSource:
    filename: str
    local_path: Path | None = None
    repo_id: str | None = None
    revision: str | None = None


class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"
    architectures = ["RWKV7ForCausalLM"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        head_size: int = 64,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 8192,
        **kwargs,
    ):
        kwargs.setdefault("architectures", self.architectures)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = hidden_size // head_size
        self.max_position_embeddings = max_position_embeddings


def try_parse_rwkv7_pth_source(model: str | Path) -> RWKV7PthSource | None:
    model_str = str(model)
    path = Path(model_str)
    if path.is_file() and path.suffix == ".pth":
        return RWKV7PthSource(filename=path.name, local_path=path)

    parsed = urlparse(model_str)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != "huggingface.co":
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5:
        return None
    repo_id = "/".join(parts[:2])
    marker = parts[2]
    if repo_id != BLINKDL_RWKV7_G1_REPO or marker not in {"blob", "resolve"}:
        return None

    revision = parts[3]
    filename = "/".join(parts[4:])
    if not filename.endswith(".pth"):
        return None
    return RWKV7PthSource(filename=filename, repo_id=repo_id, revision=revision)


def build_rwkv7_config_from_pth(model: str | Path) -> RWKV7Config | None:
    source = try_parse_rwkv7_pth_source(model)
    if source is None:
        return None

    filename = Path(source.filename).name
    match = _RWKV7_G1_FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"Unsupported RWKV7 raw .pth checkpoint: {filename}. "
            "Expected a BlinkDL/rwkv7-g1 filename such as "
            "rwkv7-g1g-1.5b-20260526-ctx8192.pth."
        )

    size = match.group("size").lower()
    num_layers, hidden_size = _RWKV7_G1_SPECS[size]
    return RWKV7Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        max_position_embeddings=int(match.group("context")),
    )


def download_rwkv7_pth_source(
    source: RWKV7PthSource,
    *,
    cache_dir: str | None,
    revision: str | None,
) -> Path:
    if source.local_path is not None:
        return source.local_path
    if source.repo_id is None:
        raise ValueError(f"Cannot download non-Hugging Face RWKV7 source: {source}")
    resolved = hf_hub_download(
        repo_id=source.repo_id,
        filename=source.filename,
        revision=revision or source.revision,
        cache_dir=cache_dir,
    )
    return Path(resolved)
