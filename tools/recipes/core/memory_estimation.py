# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Conservative model weight and standard-decoder KV-cache estimates."""

from __future__ import annotations

import math
from typing import Any

import regex as re


def estimate_weight_bytes(config: dict[str, Any]) -> tuple[int, str]:
    """Read concrete safetensors file sizes without downloading model weights.

    Model-file metadata is read through the configured HF revision. Only an
    unambiguous standard weight set is accepted. Quantization is reflected in
    actual file sizes rather than a guessed bytes/parameter ratio.
    """
    from vllm.transformers_utils.repo_utils import hf_api

    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Model ID is missing")
    revision = config.get("revision")
    info = hf_api().model_info(model, revision=revision, files_metadata=True)
    siblings = {f.rfilename: f.size for f in info.siblings or []}
    files = [name for name in siblings if name.endswith(".safetensors")]
    # Ignore non-weight auxiliary files, but reject alternate weight sets.
    if files == ["model.safetensors"]:
        selected = files
    else:
        indexed = [
            name
            for name in files
            if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
        ]
        if len(indexed) != len(files) or not indexed:
            raise ValueError(
                "Cannot identify an unambiguous standard safetensors weight set"
            )
        counts = {int(re.search(r"-of-(\d{5})", name).group(1)) for name in indexed}
        if len(counts) != 1 or next(iter(counts)) != len(indexed):
            raise ValueError("Incomplete or ambiguous safetensors shards")
        selected = indexed
    sizes = [siblings[name] for name in selected]
    if any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError(
            "Hub file metadata does not contain reliable safetensors sizes"
        )
    return sum(sizes), "Hugging Face safetensors file metadata"


def estimate_kv_bytes(config: dict[str, Any], tp: int) -> int:
    """Estimate at least one max-length request's uncompressed KV capacity."""
    from vllm.transformers_utils.config import get_config

    hf = get_config(
        config["model"],
        trust_remote_code=config.get("trust-remote-code", False),
        revision=config.get("revision"),
        code_revision=config.get("code-revision"),
        config_format=config.get("config-format", "auto"),
    )
    if getattr(hf, "vision_config", None) is not None:
        raise ValueError("Multimodal KV/runtime footprint is not currently estimated")
    text = hf.get_text_config()
    layers = getattr(text, "num_hidden_layers", None)
    heads = getattr(text, "num_attention_heads", None)
    kv_heads = getattr(text, "num_key_value_heads", heads)
    hidden = getattr(text, "hidden_size", None)
    maximum = config.get("max-model-len") or getattr(
        text, "max_position_embeddings", None
    )
    if not all(
        isinstance(x, int) and x > 0 for x in (layers, heads, kv_heads, hidden, maximum)
    ):
        raise ValueError("Model context limit and standard decoder dimensions required")
    if hidden % heads:
        raise ValueError("Unsupported attention head dimension")
    if config.get("kv-cache-dtype", "auto") not in ("auto", "bfloat16", "float16"):
        raise ValueError("Unsupported KV-cache dtype for conservative fit estimate")
    # vLLM partitions KV heads unless they are fewer than TP, when replicated.
    local_kv_heads = math.ceil(kv_heads / tp) if kv_heads >= tp else 1
    # At least 2 bytes/element; include 25% padding for cache block allocation.
    return math.ceil(
        2 * layers * local_kv_heads * (hidden // heads) * maximum * 2 * 1.25
    )
