# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def _default_cpu_offload() -> bool:
    """Honor the legacy environment variable only when the field is omitted."""
    if envs.is_set("VLLM_PLE_CPU_OFFLOAD"):
        logger.warning_once(
            "VLLM_PLE_CPU_OFFLOAD is a legacy setting and may be removed in a "
            "future release. Use --engram-config.cpu_offload instead."
        )
    return envs.VLLM_PLE_CPU_OFFLOAD


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = Field(default_factory=_default_cpu_offload)
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Defaults to False, or VLLM_PLE_CPU_OFFLOAD when set for compatibility.
    An explicit value takes precedence over the legacy environment variable."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings. Currently only the CUDA Qwen4Exp and DeepSeek "
                "V4.1 implementations with non-empty n-gram layer ids are "
                "supported."
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
