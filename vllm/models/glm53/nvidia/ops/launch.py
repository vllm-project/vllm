# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch preparation for GLM-5.3's decode segment-1 kernels, in Python."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.models.glm53.nvidia.ops.cute_dsl import registry
from vllm.models.glm53.nvidia.ops.cute_dsl.registry import ENTRIES

if TYPE_CHECKING:
    from transformers import PretrainedConfig

#: The block-table row the kernels index (vLLM's, or the kernels' zero-padded copy).
BLOCK_TABLE = "host.block_table"
ROWS = 4
#: The peer-arena slice of vLLM's embedding shard, which the prologue gathers rows from.
EMBEDDING_SLICE = "glm53.embedding.shard"


@dataclass(frozen=True)
class Region:
    """One device byte range: ``size`` bytes from ``address``."""

    address: int
    size: int


Regions = Mapping[str, Region]


Argument = int | None


@dataclass(frozen=True)
class Launch:
    """One kernel launch: the entry and its host-wrapper arguments in order (addresses,
    None, scalars).
    """

    kernel: str
    site: str
    arguments: tuple[Argument, ...]


Bound = list[tuple[Callable[..., object], tuple]]


def bind(launches: Iterable[Launch], device_index: int) -> Bound:
    """Compile (once per process) and bind ``launches`` on a device."""
    return [
        (
            registry.executor(item.kernel, device_index),
            registry.bind(item.kernel, item.arguments),
        )
        for item in launches
    ]


def run(bound: Bound, stream: int) -> None:
    """Launch bound kernels in order on ``stream`` (capturable into a CUDA graph)."""
    from cuda.bindings.driver import CUstream

    handle = CUstream(stream)
    for executor, values in bound:
        executor(*values, handle)


def sizes_of(kernel: str) -> dict[str, int]:
    """Return entry ``kernel``'s host sizes (``registry.ENTRIES``)."""
    return ENTRIES[kernel].host


def address(regions: Regions, name: str, size: int | None = None) -> int:
    """Return region ``name``'s address, checking that it holds ``size`` bytes."""
    try:
        region = regions[name]
    except KeyError:
        raise KeyError(f"no region {name!r}") from None
    if size is not None and region.size < size:
        raise ValueError(f"region {name!r} holds {region.size} bytes, needs {size}")
    return region.address


def _shared_layers(config: PretrainedConfig) -> tuple[bool, ...]:
    """Each layer's IndexCache role as vLLM's DSA attention reads it: True reuses the
    last full selection.
    """
    layers = config.num_hidden_layers
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None:
        return tuple(
            layer < len(pattern) and pattern[layer] == "S" for layer in range(layers)
        )
    frequency = getattr(config, "index_topk_freq", 1)
    offset = getattr(config, "index_skip_topk_offset", 2)
    return tuple(max(layer - offset + 1, 0) % frequency != 0 for layer in range(layers))


def index_scale_packed() -> bool:
    """Whether vLLM stores the index-key cache's per-key scale as four packed UE8M0
    bytes (its SM100 MX logits kernels) instead of one fp32; vLLM before that format
    has no such predicate.
    """
    try:
        from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
    except ImportError:
        return False
    return mqa_logits_use_mx_scales()


@dataclass(frozen=True)
class Glm53Geometry:
    """GLM-5.3's shapes on one of ``world`` TP ranks and its layer program, read once
    from the model config.
    """

    world: int
    layers: int
    dense_layers: int
    #: Per layer: whether it reuses the previous full layer's sparse selection.
    shared_layers: tuple[bool, ...]
    hidden: int
    #: Attention heads per rank.
    heads: int
    q_lora: int
    kv_lora: int
    nope_dim: int
    rope_dim: int
    value_dim: int
    index_dim: int
    index_heads: int
    index_topk: int
    experts: int
    top_k: int
    #: MoE and dense FFN intermediate rows, and vocabulary rows, per rank.
    expert_shard: int
    dense_shard: int
    vocab_shard: int
    #: The FP8 checkpoint's square weight-scale block.
    block: int
    rows: int
    #: Ranks that shard a full layer's index selection (one selector entry each).
    selector_ranks: int
    index_scale_packed: bool = False

    @classmethod
    def from_config(cls, config: PretrainedConfig, world: int) -> Glm53Geometry:
        """Read the geometry from GLM-5.3's Hugging Face config."""
        block_rows, block_columns = config.quantization_config["weight_block_size"]
        if block_rows != block_columns:
            raise ValueError(
                "GLM-5.3 decode reads square FP8 weight blocks, not "
                f"{block_rows}x{block_columns}"
            )
        return cls(
            world=world,
            layers=config.num_hidden_layers,
            dense_layers=config.first_k_dense_replace,
            shared_layers=_shared_layers(config),
            hidden=config.hidden_size,
            heads=config.num_attention_heads // world,
            q_lora=config.q_lora_rank,
            kv_lora=config.kv_lora_rank,
            nope_dim=config.qk_nope_head_dim,
            rope_dim=config.qk_rope_head_dim,
            value_dim=config.v_head_dim,
            index_dim=config.index_head_dim,
            index_heads=config.index_n_heads,
            index_topk=config.index_topk,
            experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            expert_shard=config.moe_intermediate_size // world,
            dense_shard=config.intermediate_size // world,
            vocab_shard=config.vocab_size // world,
            block=block_rows,
            rows=ROWS,
            selector_ranks=sum(
                entry.cls == "SelectorFull"
                and ("PACKED_SCALE", True) not in entry.constants
                for entry in ENTRIES.values()
            ),
            index_scale_packed=index_scale_packed(),
        )

    @property
    def qk_head(self) -> int:
        """Q-B rows per head: the no-RoPE part, then the RoPE part."""
        return self.nope_dim + self.rope_dim

    @property
    def kv_b_stride(self) -> int:
        """KV-B rows per head: the key's no-RoPE part, then the value."""
        return self.nope_dim + self.value_dim

    def is_shared(self, layer: int) -> bool:
        """Return whether ``layer`` reuses an earlier layer's sparse selection."""
        return self.shared_layers[layer]

    def ffn_kind(self, layer: int) -> str:
        """Return ``layer``'s feed-forward kernel: dense, moe or moe_seam (the next
        layer is shared).
        """
        if layer < self.dense_layers:
            return "dense"
        if layer + 1 < self.layers and self.is_shared(layer + 1):
            return "moe_seam"
        return "moe"

    def is_push_rank(self, layer: int, rank: int) -> bool:
        """Return whether ``rank``'s worker of ``layer`` pushes a selector rank's query
        (full layers).
        """
        return (
            not self.is_shared(layer)
            and self.selector_ranks <= rank < 2 * self.selector_ranks
        )

    def worker_op(self, layer: int, rank: int) -> str:
        """Return the attention worker operator of ``layer`` on ``rank`` (its launch-
        site name).
        """
        if not self.is_shared(layer):
            push = self.is_push_rank(layer, rank)
            return f"mla_worker_q4_h8{'_push' if push else ''}_fp8_full"
        seam = self.ffn_kind(layer - 1) == "moe_seam"
        return f"mla_worker_q4_h8{'_seam' if seam else ''}_fp8"

    def worker_kernel(self, layer: int, rank: int) -> str:
        """Return the kernel entry of ``layer``'s worker on ``rank``."""
        if self.is_shared(layer):
            role = "shared_seam" if self.ffn_kind(layer - 1) == "moe_seam" else "shared"
        elif rank < self.selector_ranks:
            role = "full_pushed"
        else:
            role = "full_push" if self.is_push_rank(layer, rank) else "full"
        return f"mla_worker_q4_{role}_paged_vlin"

    def selector_kernel(self, rank: int) -> str:
        """Return the kernel entry of the index selector shard on ``rank``."""
        suffix = "_mx" if self.index_scale_packed else ""
        return f"mla_selector_q4_shard{self.selector_ranks}r{rank}_paged_vlin{suffix}"

    def moe_site(self, layer: int) -> str:
        """Return the launch site of ``layer``'s MoE
        (``layer<L>.fused_moe_q4[_seam]``).
        """
        seam = self.ffn_kind(layer) == "moe_seam"
        return f"layer{layer}.fused_moe_q4{'_seam' if seam else ''}"
