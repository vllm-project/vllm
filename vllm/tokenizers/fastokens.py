# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`fastokens` backend patch.

When `VLLM_USE_FASTOKENS=1` is set, `fastokens.patch_transformers()` swaps
the inner Rust tokenizer of every HF fast tokenizer loaded afterwards with the
fastokens shim and rebinds `tokenizers.decoders.DecodeStream` so the
streaming detokenizer accepts the shim. The patch is process-global and
idempotent, so it applies to any tokenizer mode that ends up loading an HF
fast tokenizer (`hf`, `deepseek_v32`, `deepseek_v4`, …).
"""

import json
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

_MIN_FASTOKENS_VERSION = "0.3.6"


def get_fastokens_decoder_config(backend) -> dict | None:
    """Read the immutable decoder config hidden by fastokens' decoder shim."""
    if type(backend).__module__.partition(".")[0] != "fastokens":
        return None
    cache_key = "_vllm_prompt_decoder_config"
    try:
        if not hasattr(backend, cache_key):
            config = json.loads(backend.to_str()).get("decoder")
            setattr(backend, cache_key, config)
        return getattr(backend, cache_key)
    except (AttributeError, TypeError, ValueError):
        return None


def apply_fastokens_patch() -> None:
    try:
        import fastokens
    except ImportError as e:
        raise ImportError(
            f"The 'fastokens' package (>= {_MIN_FASTOKENS_VERSION}) is required "
            "when VLLM_USE_FASTOKENS=1."
        ) from e

    try:
        installed = version("fastokens")
    except PackageNotFoundError:
        installed = None
    if installed is None or Version(installed) < Version(_MIN_FASTOKENS_VERSION):
        raise ImportError(
            f"fastokens >= {_MIN_FASTOKENS_VERSION} is required when "
            f"VLLM_USE_FASTOKENS=1 (found {installed or 'unknown'})."
        )

    fastokens.patch_transformers()
