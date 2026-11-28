# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_FLASH_ENGRAM_ARCHITECTURES = {
    "HYV4FlashForCausalLM",
    "HYV4FlashVLForConditionalGeneration",
}

_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "HYV4FlashForCausalLM": "engram_layer_ids",
    "HYV4FlashVLForConditionalGeneration": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}
# Fields of models whose tables an Engram pool can split (`peer_group_size`).
_PEER_GROUP_FIELDS = {"engram_layer_ids", "ple_layer_ids"}


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
    """Store embedding weights in host memory. NVLink peer groups use local
    Grace host memory when enabled and HBM when disabled."""

    disk_offload: bool = False
    """Keep Qwen4Exp PLE tables in the checkpoint files and read each step's
    rows on the host instead of allocating the table. Takes precedence over
    cpu_offload."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Uses host-NUMA cuMem shared
    by FABRIC handle where supported (on ROCm, HIP VMM host memory with
    VLLM_ROCM_ENGRAM_HIP_VMM=1), else /dev/shm (which then needs capacity and a
    shared IPC namespace). Defaults to enabled when the model and other settings
    allow it; Flash uses private replicas. Falls back to per-replica tables when
    DP replicas are not co-located on one node or the host cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory.
    Always on for private tables on ROCm."""

    peer_group_size: int = 0
    """Split each table's rows over this many GPUs of several engines (the sum
    of their GPU counts); lookups read rows from the owning GPU. Every engine
    passes the same value, `peer_group_store` and `peer_group_transport`.
    Requires a model with Engram pool support (DeepSeek V4.1 or Qwen4Exp),
    CUDA, TP=1 and PP=1."""

    peer_group_transport: Literal["nvlink", "rdma"] = "nvlink"
    """How a peer group reads rows: `nvlink` loads them from peer device or host memory
    through CUDA FABRIC handles and an IMEX channel.
    With `rdma`, the owners' NICs gather them from HBM, or from host memory
    with `cpu_offload`."""

    peer_group_fetch: Literal["sync", "async"] = "sync"
    """Where a peer group's fetch kernels run. `sync`: on the compute stream,
    started before the decoder layers, each Engram layer waiting for its own
    rows; over RDMA the rows move without SMs in between. `async`: on a side
    stream beside the layers, so their SM work (all of NVLink's gather)
    overlaps compute. Full CUDA graphs capture both; piecewise ones break at
    the start and at each Engram layer's wait."""

    peer_group_rdma_inflight: int = Field(default=8, ge=0)
    """With `rdma`, the most prefill-sized gathers (64 WRs each) a GPU's NIC
    has in flight for its peers; 0 means no limit. Decode-sized fetches bypass
    it, so one engine's prefill delays another engine's decode by at most this
    backlog. The default (~2 MB) keeps prefill fetches at full rate."""

    peer_group_thp_dir: str | None = None
    """With `rdma` and cpu_offload: a directory on a tmpfs mounted with
    `huge=always`, where host shards are placed on transparent huge pages.
    NICs gather rows at random, so a large shard on base pages misses the
    NIC's address-translation cache on nearly every row, which can make large
    fetches several times slower."""

    peer_group_store: str | None = None
    """File through which a peer group's engines meet once at startup. Required
    with `peer_group_size`: the same path for every engine, reachable by all of
    them and fresh per launch (e.g. under a per-job directory)."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        if self.peer_group_thp_dir is not None and not (
            self.peer_group_size
            and self.cpu_offload
            and self.peer_group_transport == "rdma"
        ):
            raise ValueError(
                "peer_group_thp_dir requires peer_group_size, cpu_offload and "
                "peer_group_transport='rdma'"
            )
        if (self.peer_group_size > 0) != (self.peer_group_store is not None):
            raise ValueError(
                "peer_group_size and peer_group_store must be set together"
            )
        if (
            self.peer_group_size
            and self.cpu_offload
            and (self.dp_shared_memory or self.use_thp)
        ):
            raise ValueError(
                "peer_group_size does not support dp_shared_memory or use_thp"
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings and non-empty n-gram layer ids."
            )
        if model_config.architecture in _FLASH_ENGRAM_ARCHITECTURES:
            for option in ("dp_shared_memory", "embedding_across_dp", "disk_offload"):
                if getattr(self, option):
                    raise ValueError(f"Flash Engram does not support {option}")
        if self.disk_offload:
            from vllm.platforms import current_platform

            if field != "ple_layer_ids" or not current_platform.is_cuda():
                raise ValueError("disk_offload requires a Qwen4Exp model on CUDA")
        if self.peer_group_size and field not in _PEER_GROUP_FIELDS:
            raise ValueError(
                "peer_group_size requires a model with Engram pool support"
            )
        if self.peer_group_size and field == "ple_layer_ids":
            if self.peer_group_fetch != "sync":
                raise ValueError(
                    "Qwen4Exp Engram pool requires peer_group_fetch='sync'"
                )
            if self.disk_offload or self.embedding_across_dp:
                raise ValueError(
                    "Qwen4Exp Engram pool is incompatible with disk_offload "
                    "and embedding_across_dp"
                )

    def resolve_dp_shared_memory(
        self,
        parallel_config: "ParallelConfig",
        model_config: "ModelConfig | None" = None,
    ) -> None:
        """Share host tables by default wherever the model supports them."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.use_thp
                and not self.peer_group_size
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
                and (
                    model_config is None
                    or model_config.architecture not in _FLASH_ENGRAM_ARCHITECTURES
                )
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
        if self.disk_offload and parallel_config.enable_elastic_ep:
            raise ValueError("disk_offload is not supported with elastic EP.")
        if self.peer_group_size:
            from vllm.platforms import current_platform

            if (
                not current_platform.is_cuda()
                # Pool rows are dequantized from FP8 (e4m3), which Triton
                # supports from SM89 on.
                or not current_platform.has_device_capability(89)
                or parallel_config.tensor_parallel_size > 1
                # Every GPU of the group must build the same tables.
                or parallel_config.pipeline_parallel_size > 1
                or parallel_config.enable_elastic_ep
            ):
                raise ValueError(
                    "peer_group_size requires CUDA SM89+, TP=1, PP=1 and no elastic EP"
                )
            if self.peer_group_size < parallel_config.data_parallel_size:
                raise ValueError(
                    f"An Engram peer group of {self.peer_group_size} GPUs cannot "
                    f"hold an engine of {parallel_config.data_parallel_size}"
                )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compile_factors(self) -> CompileFactors:
        """Hash settings that affect embedding execution and graph structure."""
        # The store only matters at startup; a per-launch path must not
        # invalidate the compile cache.
        return get_compile_factors(
            self, {"peer_group_store", "peer_group_rdma_inflight"}
        )
