# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self

from pydantic import model_validator

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


def per_token_table_path(table_dir: str, layer_id: int) -> str:
    """Table file of one Engram layer for ``mode="per_token"``."""
    return os.path.join(table_dir, f"U_L{layer_id}_bf16.npy")


def check_per_token_tables(table_dir: str, layer_ids: Sequence[int]) -> None:
    """Fail early if ``table_dir`` lacks the table of any Engram layer."""
    if not os.path.isdir(table_dir):
        raise FileNotFoundError(
            f"engram replacement_tables {table_dir!r} is not a directory"
        )
    missing = [
        per_token_table_path(table_dir, i)
        for i in layer_ids
        if not os.path.isfile(per_token_table_path(table_dir, i))
    ]
    if missing:
        raise FileNotFoundError(
            "engram mode 'per_token' needs one table per Engram layer "
            f"(layers {list(layer_ids)}) named U_L<layer>_bf16.npy; "
            f"missing: {missing}"
        )


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

    mode: Literal["ngram", "zero", "per_token"] = "ngram"
    """What the Engram table lookup returns (DeepSeek V4.1, ROCm only for the
    non-default modes). ``ngram``: the checkpoint's hashed n-gram table. The other
    modes replace only the lookup output; the Engram projection, gate and residual
    add run unchanged. ``zero``: the lookup returns zeros. ``per_token``: the lookup
    returns a row of a per-token table indexed by the compressed id of the current
    token (requires ``replacement_tables``). Neither mode needs the n-gram table in
    the checkpoint; the table tensors are skipped if present. The tables are kept
    on each rank's GPU, so ``cpu_offload`` has no effect."""

    replacement_tables: str | None = None
    """Directory with one ``U_L<layer>_bf16.npy`` per Engram layer for
    ``mode="per_token"``: bf16 bits stored as ``uint16``, shape
    ``[engram_compressed_vocab_size, n_hash_cols, head_dim]``."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory.
    Always on for private tables on ROCm."""

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        if self.mode == "per_token" and not self.replacement_tables:
            raise ValueError("engram mode 'per_token' requires replacement_tables")
        if self.mode != "per_token" and self.replacement_tables:
            raise ValueError(
                "engram replacement_tables is only used with mode 'per_token'"
            )
        if self.mode != "ngram" and (
            self.embedding_across_dp or self.dp_shared_memory or self.use_thp
        ):
            raise ValueError(
                f"engram mode {self.mode!r} keeps its tables on each rank's GPU: "
                "embedding_across_dp, dp_shared_memory and use_thp must be off"
            )
        return self

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
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
        if self.mode == "ngram":
            return
        if model_config.architecture != "DeepseekV41ForCausalLM":
            raise ValueError(
                f"engram mode {self.mode!r} is only supported for DeepSeek V4.1"
            )
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            raise NotImplementedError(
                f"engram mode {self.mode!r} is only implemented for ROCm"
            )
        if self.mode == "per_token":
            assert self.replacement_tables is not None
            check_per_token_tables(
                self.replacement_tables, model_config.hf_text_config.engram_layer_ids
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.mode == "ngram"
                and self.cpu_offload
                and not self.use_thp
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

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
