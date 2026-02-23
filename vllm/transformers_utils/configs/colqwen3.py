# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColQwen3 configuration that extends Qwen3VLConfig with embedding projection
fields. This allows ColQwen3 models to be loaded without trust_remote_code
by mapping their custom model_type (colqwen3, ops_colqwen3, etc.) to a
standard config class that vLLM understands.

Supported model_types:
- colqwen3 (TomoroAI/tomoro-colqwen3-embed-8b)
- ops_colqwen3 (OpenSearch-AI/Ops-Colqwen3-4B)
- qwen3_vl_nemotron_embed (nvidia/nemotron-colembed-vl-8b-v2)
"""

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class ColQwen3Config(Qwen3VLConfig):
    """Configuration class for ColQwen3 models.

    Extends Qwen3VLConfig with additional fields used by ColQwen3 variants
    for the embedding projection layer.
    """

    # Accept any ColQwen3 variant model_type
    model_type = "colqwen3"

    def __init__(
        self,
        embed_dim: int | None = None,
        dims: int | None = None,
        dim: int | None = None,
        projection_dim: int | None = None,
        colbert_dim: int | None = None,
        pooling: str | None = None,
        **kwargs,
    ):
        # Store embedding projection config fields
        self.embed_dim = embed_dim
        self.dims = dims
        self.dim = dim
        self.projection_dim = projection_dim
        self.colbert_dim = colbert_dim
        self.pooling = pooling

        super().__init__(**kwargs)


class OpsColQwen3Config(ColQwen3Config):
    """Configuration for OpenSearch-AI ColQwen3 variants."""

    model_type = "ops_colqwen3"


class Qwen3VLNemotronEmbedConfig(ColQwen3Config):
    """Configuration for NVIDIA Nemotron ColEmbed variants."""

    model_type = "qwen3_vl_nemotron_embed"
