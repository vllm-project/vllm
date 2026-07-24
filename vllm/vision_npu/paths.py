# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Path helpers for NPU vision model bundles."""

from __future__ import annotations

import os


def normalize_vision_npu_cache_path(model_path: str) -> str:
    """Return absolute path to the .rai file for VLLM_VISION_NPU_CACHE."""
    path = os.path.abspath(model_path)
    if not path.lower().endswith(".rai"):
        raise ValueError(
            f"VLLM_VISION_NPU_CACHE must point to a .rai file, got: {model_path}"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RAI file not found: {path}")
    return path


def resolve_vision_bundle_dir(model_path: str) -> str:
    """Directory containing deploy artifacts (ONNX) next to the .rai file."""
    rai_path = normalize_vision_npu_cache_path(model_path)
    return os.path.dirname(rai_path)
