# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import model_validator
from typing_extensions import Self

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
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
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled whenever
    the other settings allow it, falling back to per-replica tables when DP
    replicas are not co-located on one node or /dev/shm cannot hold them."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
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
            or not current_platform.is_cuda_alike()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings, non-empty n-gram layer ids, and CUDA."
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def verify_host_memory(
        self, table_bytes: int, parallel_config: "ParallelConfig"
    ) -> None:
        """Refuse to pin more host memory for Engram than this node can hold.

        Offloaded tables are pinned, so they can be neither reclaimed nor
        swapped. That makes a preflight worthwhile, and it makes the per-rank
        view misleading: every rank of an engine pins its shard inside the same
        container, so a TP8 engine needs the whole table from one cgroup even
        though each rank asks for only an eighth of it.

        Only an impossible request is rejected -- more than the hard cgroup
        limit, or more than physical RAM. Tight but possible headroom is left
        to check_cgroup_memory_available, which only warns because current
        usage may be reclaimable. Assumes an engine's TP ranks share one node.
        """
        if not self.cpu_offload or self.dp_shared_memory:
            # GPU-resident, or backed by /dev/shm and checked against it there.
            return
        import psutil

        from vllm.utils.cpu_resource_utils import (
            check_cgroup_memory_available,
            get_cgroup_memory_limit,
        )

        dp_local = parallel_config.data_parallel_size_local
        if self.embedding_across_dp:
            # One table is sharded over every DP replica; this node holds the
            # share owned by its local replicas.
            required = table_bytes * dp_local // parallel_config.data_parallel_size
        else:
            # Each local DP replica holds a full table split over its TP ranks.
            required = table_bytes * dp_local

        limit, source = psutil.virtual_memory().total, "physical RAM"
        hint = "use a host with more memory"
        cgroup_limit = get_cgroup_memory_limit()
        if cgroup_limit is not None and cgroup_limit < limit:
            limit, source = cgroup_limit, "container memory limit"
            hint = "raise the container memory limit"

        if required > limit:
            gib = 1 << 30
            raise ValueError(
                f"Engram tables need {required / gib:.1f} GiB of pinned host "
                f"memory on this node, but the {source} is "
                f"{limit / gib:.1f} GiB. Pinned memory cannot be reclaimed or "
                "swapped, so this allocation cannot succeed. Keep the tables in "
                "GPU memory with --engram-config '{\"cpu_offload\": false}', "
                f"or {hint}."
            )
        check_cgroup_memory_available(required, "Engram pinned host tables")

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
