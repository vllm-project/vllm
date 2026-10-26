# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch preparation for GLM-5.3's decode segment-1 kernels, in Python."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.models.glm53.nvidia.ops.cute_dsl import registry
from vllm.models.glm53.nvidia.ops.cute_dsl.registry import ENTRIES
from vllm.models.glm53.nvidia.ops.cute_dsl.row_table import (
    PACKED_SLOTS,
    SEGMENT_SLACK,
)

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

#: The block-table row the kernels index (vLLM's, or the kernels' zero-padded copy).
BLOCK_TABLE = "host.block_table"
ROWS = 4
#: MTP drafts per request of the q4 build (a request's rows are ``1 + drafts``).
DRAFTS = ROWS - 1
TIERS = (4, 8, 16)
MOE_TIERS = (4, 16)
WIDE_SELECTION_TIERS: tuple[int, ...] = (8, 16)
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


def _shared_layers(config: PreTrainedConfig) -> tuple[bool, ...]:
    layers = config.num_hidden_layers
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None:
        return tuple(
            layer < len(pattern) and pattern[layer] == "S" for layer in range(layers)
        )
    frequency = getattr(config, "index_topk_freq", 1)
    offset = getattr(config, "index_skip_topk_offset", 2)
    return tuple(max(layer - offset + 1, 0) % frequency != 0 for layer in range(layers))


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
    drafts: int
    #: Ranks that select for a full layer (one selector entry each).
    selector_ranks: int
    selector_split: str = "key"
    packed: bool = False
    w8a16_rows: int = 0
    live: int = 0

    @classmethod
    def from_config(
        cls,
        config: PreTrainedConfig,
        world: int,
        rows: int = ROWS,
        drafts: int = DRAFTS,
    ) -> Glm53Geometry:
        """Read the geometry from GLM-5.3's Hugging Face config."""
        if rows not in TIERS:
            raise ValueError(f"row capacity {rows} is not one of {TIERS}")
        if not 0 <= drafts < rows:
            raise ValueError(f"{drafts} drafts do not fit {rows} rows")
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
            rows=rows,
            drafts=drafts,
            selector_ranks=len(
                {
                    dict(entry.constants)["SHARD"]
                    for entry in ENTRIES.values()
                    if entry.cls == "SelectorFull"
                    and "ROWS" not in dict(entry.constants)
                }
            ),
        )

    def for_tier(
        self,
        rows: int,
        drafts: int | None = None,
        packed: bool = False,
        w8a16_rows: int = 0,
        live: int = 0,
    ) -> Glm53Geometry:
        """Return this geometry at row capacity ``rows`` (one of ``TIERS``) and
        ``drafts`` MTP drafts per request (default: unchanged); ``packed``: the true
        rows of up to ``PACKED_SLOTS`` requests back to back (request-parallel
        selection over all eight ranks); ``w8a16_rows``: the W8A16 MoE build
        (:attr:`w8a16_rows`); ``live``: the live rows of a lone request that has
        dead rows (:attr:`live`; 0 or ``1 + drafts``: none, and none on the q4 tier,
        whose MoE is its own).
        """
        drafts = self.drafts if drafts is None else drafts
        if rows not in TIERS or not 0 <= drafts < rows:
            raise ValueError(f"{drafts} drafts do not fit {rows} rows")
        if w8a16_rows and not (
            w8a16_rows in (ROWS, 8) and packed and 2 * (drafts + 1) <= w8a16_rows
        ):
            raise ValueError(
                f"the q{w8a16_rows} MoE serves a packed build of at least two requests "
                f"in {w8a16_rows} rows: {drafts} drafts per request, packed={packed}"
            )
        if packed and rows not in PACKED_SLOTS:
            raise ValueError(
                f"packed builds exist for the tiers {sorted(PACKED_SLOTS)}, not {rows}"
            )
        split = "request8" if rows in WIDE_SELECTION_TIERS or packed else "key"
        if rows > ROWS and rows // (drafts + 1) == 1:
            split = "group8"
        if packed and live:
            raise ValueError("a packed build has no dead rows")
        live = live if 0 < live < drafts + 1 and rows > ROWS else 0
        return dataclasses.replace(
            self,
            rows=rows,
            drafts=drafts,
            packed=packed,
            w8a16_rows=w8a16_rows,
            live=live,
        ).with_split(split)

    def with_split(self, split: str) -> Glm53Geometry:
        """Return this geometry with the index selection split ``split`` (``key``,
        ``request8`` or ``group8``) and the selector ranks it implies.
        """
        if self.packed and split != "request8":
            raise ValueError(
                f"a packed build selects request-parallel over eight ranks "
                f"('request8', the only split with segments), not {split!r}"
            )
        ranks = registry.SELECTOR_RANKS
        if split in ("request8", "group8"):
            ranks = registry.WIDE_SELECTOR_RANKS
        return dataclasses.replace(self, selector_split=split, selector_ranks=ranks)

    @property
    def row_tag(self) -> str:
        return "q4" if self.rows == ROWS and self.drafts == DRAFTS else f"r{self.rows}"

    @property
    def tag(self) -> str:
        return f"r{self.rows}p" if self.packed else self.row_tag

    @property
    def moe_rows(self) -> int:
        if self.w8a16_rows:
            return self.w8a16_rows
        if self.rows == 8 and not self.packed:
            return 8
        return min(tier for tier in MOE_TIERS if tier >= self.rows)

    @property
    def slack_rows(self) -> int:
        return SEGMENT_SLACK if self.packed else 0

    @property
    def buffer_rows(self) -> int:
        return max(self.rows, self.moe_rows) + self.slack_rows

    @property
    def moe_tag(self) -> str:
        if self.moe_rows == ROWS and (self.w8a16_rows or self.drafts == DRAFTS):
            return "q4"
        if self.moe_rows == 8:
            return f"q8d{self.drafts + 1}x{self.live}" if self.live else "q8"
        return f"r{self.moe_rows}"

    @property
    def max_requests(self) -> int:
        requests = self.rows // (self.drafts + 1)
        return min(requests, PACKED_SLOTS[self.rows]) if self.packed else requests

    @property
    def table_requests(self) -> int:
        return PACKED_SLOTS[self.rows] if self.packed else self.rows // ROWS

    @property
    def selector_slots(self) -> int:
        groups = self.rows // ROWS
        return max(groups, PACKED_SLOTS[self.rows]) if self.packed else groups

    @property
    def qk_head(self) -> int:
        return self.nope_dim + self.rope_dim

    @property
    def kv_b_stride(self) -> int:
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

    def is_pushed_rank(self, layer: int, rank: int) -> bool:
        """Return whether ``rank``'s worker of ``layer`` is the pushed role: a selector
        rank of a full layer whose head group a push rank computes (none when every
        rank selects).
        """
        return (
            not self.is_shared(layer)
            and rank < self.selector_ranks
            and self.world >= 2 * self.selector_ranks
        )

    def worker_op(self, layer: int, rank: int) -> str:
        """Return the attention worker operator of ``layer`` on ``rank`` (its launch-
        site name).
        """
        if not self.is_shared(layer):
            push = self.is_push_rank(layer, rank)
            return f"mla_worker_{self.tag}_h8{'_push' if push else ''}_fp8_full"
        seam = self.ffn_kind(layer - 1) == "moe_seam"
        return f"mla_worker_{self.tag}_h8{'_seam' if seam else ''}_fp8"

    def worker_kernel(self, layer: int, rank: int) -> str:
        """Return the kernel entry of ``layer``'s worker on ``rank``."""
        if self.is_shared(layer):
            role = "shared_seam" if self.ffn_kind(layer - 1) == "moe_seam" else "shared"
        elif self.is_pushed_rank(layer, rank):
            role = "full_pushed"
        else:
            role = "full_push" if self.is_push_rank(layer, rank) else "full"
            if role == "full" and self.selector_split in ("request8", "group8"):
                role = "full_solo"
        return f"mla_worker_{self.tag}_{role}_paged_vlin"

    def mailbox_kernel(self, layer: int) -> str:
        """Return the worker entry whose mailbox size is a full layer's output slice
        (declared on every rank): the push worker's, whose mailbox the workers of the
        layer's selector ranks write; a row tier (no push rank) sizes it as its
        shared worker's.
        """
        if self.rows > ROWS:
            return f"mla_worker_{self.tag}_shared_paged_vlin"
        return self.worker_kernel(layer, self.selector_ranks)

    def selector_kernel(self, rank: int) -> str:
        """Return the kernel entry of the index selector on ``rank``."""
        split = {"key": "shard", "group8": "gserve"}.get(self.selector_split, "serve")
        return f"mla_selector_{self.tag}_{split}{self.selector_ranks}r{rank}_paged_vlin"

    def moe_site(self, layer: int) -> str:
        """Return the launch site of ``layer``'s MoE (``layer<L>.fused_moe_<moe
        tag>[_seam]``).
        """
        name = self.moe_kernel(layer).removesuffix("_ckpt_shared")
        return f"layer{layer}.{name}"

    def moe_kernel(self, layer: int) -> str:
        """Return the kernel entry of ``layer``'s MoE."""
        seam = self.ffn_kind(layer) == "moe_seam"
        return f"fused_moe_{self.moe_tag}{'_seam' if seam else ''}_ckpt_shared"

    @property
    def dense_kernel(self) -> str:
        return f"dense_ffn_{self.row_tag}_vllm"

    def prologue_kernel(self, replicated: bool) -> str:
        """Return the prologue entry: a replicated embedding table, or vLLM's shards."""
        return f"prologue_{self.tag}" if replicated else f"prologue_vllm_{self.tag}"
