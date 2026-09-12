# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import model_validator
from typing_extensions import Self

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.load import LoadConfig
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup by default.
    Set --engram-config.cpu_offload false to keep the weights on the GPU."""

    embedding_across_dp: bool = False
    """Shard Qwen4Exp embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    enable_engram_dp_sharding: bool = True
    """Shard DeepSeek V4.1 hash heads across node-local DP replicas.
    Set to false to keep a separate TP-sharded table per replica.
    Independent of embedding_across_dp; DP shared memory takes precedence."""

    enable_engram_dp_shared_memory: bool = False
    """Share DeepSeek V4.1 CPU-offloaded weights across node-local DP replicas.
    Requires shared IPC. TP sharding is unchanged."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.enable_engram_dp_shared_memory and not self.cpu_offload:
            raise ValueError("enable_engram_dp_shared_memory requires cpu_offload=True")
        return self

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
        if self.enable_engram_dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError(
                    "enable_engram_dp_shared_memory requires data_parallel_size > 1."
                )
            if parallel_config.enable_elastic_ep:
                raise ValueError(
                    "enable_engram_dp_shared_memory is not supported with elastic EP."
                )
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def verify_load_config(self, load_config: "LoadConfig") -> None:
        """Shared tables require a loader that invokes parameter weight callbacks."""
        if self.enable_engram_dp_shared_memory and load_config.load_format not in (
            "auto",
            "safetensors",
            "pt",
        ):
            raise ValueError(
                "enable_engram_dp_shared_memory requires load_format 'auto', "
                f"'safetensors' or 'pt'; got {load_config.load_format!r}."
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
