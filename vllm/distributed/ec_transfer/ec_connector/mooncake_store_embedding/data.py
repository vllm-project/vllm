# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content identities and output contracts for the shared Encoder Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EMBEDDING_CACHE_KEY_VERSION = "v3"


def make_embedding_key(namespace: str, identifier: str) -> str:
    return f"{namespace}@id:{identifier}"


@dataclass(frozen=True)
class TensorSpec:
    """Contiguous output contract used by the Encoder and Store codec."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def build_embedding_namespace(vllm_config: VllmConfig) -> str:
    model_config = vllm_config.model_config
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None
    extra_config = ec_config.ec_connector_extra_config

    multimodal_config = model_config.multimodal_config
    encoder_hash = (
        multimodal_config.compute_hash()
        if multimodal_config is not None
        else "encoder:default"
    )
    cache_prefix = str(extra_config.get("embedding_cache_prefix", ""))
    model_name = str(extra_config.get("embedding_model_identity", model_config.model))
    prefix = f"{cache_prefix}@" if cache_prefix else ""
    return (
        f"{prefix}embedding"
        f"@model:{model_name}"
        f"@revision:{model_config.revision or 'default'}"
        f"@encoder:{encoder_hash}"
        f"@dtype:{model_config.dtype}"
        f"@protocol:{EMBEDDING_CACHE_KEY_VERSION}"
    )
