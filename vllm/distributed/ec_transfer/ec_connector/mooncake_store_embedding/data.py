# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content identities and output contracts for the shared Encoder Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EMBEDDING_PROTOCOL_VERSION = "v2"
MOONCAKE_TENSOR_METADATA_NBYTES = 304


def _escape_key_part(value: str) -> str:
    return quote(value, safe="-_.~")


@dataclass(frozen=True)
class EmbeddingKeyMetadata:
    """Metadata that defines the semantic namespace for embedding reuse."""

    cache_prefix: str
    model_name: str
    model_revision: str
    encoder: str
    dtype: str
    protocol_version: str


@dataclass(frozen=True)
class EmbeddingPoolKey:
    """Key for addressing one embedding tensor in the distributed store."""

    key_metadata: EmbeddingKeyMetadata
    identifier: str

    def to_string(self) -> str:
        meta = self.key_metadata
        prefix = f"{_escape_key_part(meta.cache_prefix)}@" if meta.cache_prefix else ""
        return (
            f"{prefix}embedding"
            f"@model:{_escape_key_part(meta.model_name)}"
            f"@revision:{_escape_key_part(meta.model_revision)}"
            f"@encoder:{_escape_key_part(meta.encoder)}"
            f"@dtype:{_escape_key_part(meta.dtype)}"
            f"@protocol:{_escape_key_part(meta.protocol_version)}"
            f"@id:{_escape_key_part(self.identifier)}"
        )


@dataclass(frozen=True)
class TensorSpec:
    """Contiguous output contract used by the Encoder and Store codec."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def build_embedding_key_metadata(vllm_config: VllmConfig) -> EmbeddingKeyMetadata:
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
    return EmbeddingKeyMetadata(
        cache_prefix=str(extra_config.get("embedding_cache_prefix", "")),
        model_name=str(
            extra_config.get("embedding_model_identity", model_config.model)
        ),
        model_revision=str(model_config.revision or "default"),
        encoder=str(encoder_hash),
        dtype=str(model_config.dtype),
        protocol_version=EMBEDDING_PROTOCOL_VERSION,
    )
