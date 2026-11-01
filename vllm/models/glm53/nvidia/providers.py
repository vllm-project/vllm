# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch providers of GLM-5.3's latency-specialized decode (segment 1)."""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import ClassVar

import torch

from vllm.models.glm53.nvidia.arena import Glm53Placement, Slice
from vllm.models.glm53.nvidia.ops.cute_dsl import registry
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_selector import StateWords
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import FlagWords
from vllm.models.glm53.nvidia.ops.launch import (
    BLOCK_TABLE,
    EMBEDDING_SLICE,
    ROWS,
    Argument,
    Glm53Geometry,
    Launch,
    sizes_of,
)
from vllm.models.glm53.nvidia.weights import ExpertInputs

#: The kernel record field and the segment input that carry the step's row table.
ROW_TABLE_ARGUMENT = "RowTable"
ROW_TABLE_INPUT = "row_table"
BLOCK_STRIDE_ARGUMENT = "BlockStride"

Peers = Callable[[str], list[int]]
"""A slice's address on each of the eight ranks."""


def ordered_key(score: float) -> int:
    """Return the selector's order-preserving u32 key of an FP32 ``score``."""
    bits = struct.unpack("<I", struct.pack("<f", score))[0]
    if bits == 0x80000000:
        bits = 0
    return bits ^ 0xFFFFFFFF if bits & 0x80000000 else bits | 0x80000000


def _signed(word: int) -> int:
    return struct.unpack("<i", struct.pack("<I", word))[0]


def _words(value: int) -> list[int]:
    return [_signed(part) for part in (value & 0xFFFFFFFF, value >> 32)]


class Provider(ABC):
    """One launch of the segment: its entry, site, results, owned state and
    peer slices.
    """

    kernel: str
    site: str

    def __init__(
        self,
        geometry: Glm53Geometry,
        device: torch.device,
        inputs: Mapping[str, torch.Tensor],
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.device: torch.device = device
        self.inputs: Mapping[str, torch.Tensor] = inputs
        self.results: dict[str, torch.Tensor] = {}
        self.state: dict[str, torch.Tensor] = {}
        self.slices: list[Slice] = []
        self.launch: Launch | None = None

    def result(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate result ``name`` (uninitialized: the kernel writes it before any
        reader).
        """
        self.results[name] = torch.empty(shape, dtype=dtype, device=self.device)
        return self.results[name]

    def rows(self, name: str) -> torch.Tensor:
        """Allocate BF16 result ``name`` of the step's hidden rows."""
        return self.result(
            name, (self.geometry.buffer_rows, self.geometry.hidden), torch.bfloat16
        )

    def owned(
        self, name: str, words: int, fill: int = 0, placed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Allocate the int32 protocol-state tensor ``name`` (or adopt its placed
        tensor).
        """
        if placed is None:
            placed = torch.full((words,), fill, dtype=torch.int32, device=self.device)
        self.state[name] = placed
        return placed

    def address(self, name: str) -> int:
        """Return input ``name``'s device address."""
        return self.inputs[name].data_ptr()

    def bind(self, peers: Peers) -> None:
        """Fix the launch's arguments once the peer arena exists."""
        arguments = list(self.arguments(peers))
        names = registry.arguments(self.kernel)
        if ROW_TABLE_ARGUMENT in names:
            # A row-tier kernel takes the step's row table where its record puts it.
            arguments.insert(
                names.index(ROW_TABLE_ARGUMENT), self.address(ROW_TABLE_INPUT)
            )
        if BLOCK_STRIDE_ARGUMENT in names:
            arguments.insert(
                names.index(BLOCK_STRIDE_ARGUMENT), self.inputs[BLOCK_TABLE].stride(0)
            )
        self.launch = Launch(self.kernel, self.site, tuple(arguments))

    @abstractmethod
    def arguments(self, peers: Peers) -> list[Argument]:
        """Return the entry's host-wrapper arguments in order."""
        raise NotImplementedError


class Prologue(Provider):
    """Embed the step's tokens (vLLM's embedding shards, or a replicated table)."""

    site = "prologue"
    FREQ: ClassVar[str] = "prologue.freq"

    def __init__(
        self, geometry: Glm53Geometry, inputs: Mapping[str, torch.Tensor]
    ) -> None:
        super().__init__(geometry, inputs["token_ids"].device, inputs)
        self.replicated: bool = "embedding" in inputs
        self.kernel = geometry.prologue_kernel(self.replicated)
        self.hidden: torch.Tensor = self.rows("hidden")
        self.freq: torch.Tensor = self.result(
            "freq",
            (geometry.rows + geometry.slack_rows, geometry.rope_dim),
            torch.float32,
        )

    def arguments(self, peers: Peers) -> list[Argument]:
        """Token IDs, position, table(s), the rows and RoPE rows, the
        table's capacity.
        """
        rope = self.inputs["rope_table"]
        rows = [self.hidden.data_ptr(), self.freq.data_ptr(), rope.shape[0]]
        step = [self.address("token_ids"), self.address("position")]
        if self.replicated:
            return [*step, self.address("embedding"), rope.data_ptr(), *rows]
        return [*step, rope.data_ptr(), *rows, *peers(EMBEDDING_SLICE)]


class Selector(Provider):
    """A full layer's index selector shard (ranks 0-2): scores, selects, pushes the
    top-k keys.
    """

    WEIGHTS = (
        "gamma_x",
        "probe.input_weight",
        "probe.input_scales",
        "wis_weight",
        "key_gamma",
        "key_bias",
        "gamma_q",
        "wqb_weight",
        "wqb_scales",
    )
    RESULTS = ("qa", "raw_key", "head_weights", "raw_query", "index_query", "scores")
    #: Per node, a nominal top-k score threshold a request's selection starts from.
    NOMINAL_SEED_SCORES: ClassVar[dict[str, float]] = {"layers.1": 100.0}

    def __init__(
        self,
        geometry: Glm53Geometry,
        layer: int,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
    ) -> None:
        super().__init__(geometry, x.device, inputs)
        self.layer: int = layer
        self.x: torch.Tensor = x
        self.kernel = geometry.selector_kernel(rank)
        self.site = (
            f"layer{layer}.mla_selector_{geometry.tag}_shard{geometry.selector_ranks}"
        )
        self.prefix: str = f"layers.{layer}.attention."
        sizes = sizes_of(self.kernel)
        # The RoPE table covers the block table's positions.
        self.capacity: int = inputs["rope_table"].shape[0]
        if self.capacity % sizes["kKeyTile"] or self.capacity > sizes["kMaxContext"]:
            raise ValueError(f"selector key capacity {self.capacity} does not fit")
        rows = geometry.rows + geometry.slack_rows
        index = (geometry.index_heads, geometry.index_dim)
        shapes = (
            (rows, geometry.q_lora),
            (rows, geometry.index_dim),
            (rows, geometry.index_heads),
            (rows, *index),
            (rows, *index),
        )
        for name, shape in zip(self.RESULTS, shapes):
            self.result(name, shape, torch.bfloat16)
        scratch = -(-128 * rows // self.capacity)
        self.result("scores", (rows + scratch, self.capacity), torch.float32)
        self.final: torch.Tensor = self.rows("final")
        self.owned(self.prefix + "selected_ids", rows * geometry.index_topk, -1)
        blocks = geometry.selector_slots
        self.owned(self.prefix + "full_epoch", ROWS * blocks)
        self.owned(self.prefix + "flags", sizes["kFlagWords"])
        self.select_state: torch.Tensor = self.owned(
            self.prefix + "select_state", sizes["kStateWords"]
        )
        words = self.nominal_seed_words(f"layers.{layer}", ROWS * blocks)
        if geometry.packed:
            for slot in range(blocks):
                first_absent = self.packed_slot_rows(geometry.drafts + 1, slot)
                for row in range(first_absent, ROWS):
                    words[slot * ROWS + row] = 0
        # The seeds are constant: keep them on the device so a reset is a device copy.
        self.seeds: torch.Tensor = torch.tensor(
            words, dtype=torch.int32, device=self.select_state.device
        )

    @classmethod
    def nominal_seed_words(cls, node: str, rows: int) -> list[int]:
        """Return the seed words a request starts with on selector ``node``."""
        score = cls.NOMINAL_SEED_SCORES.get(node)
        return [_signed(0 if score is None else ordered_key(score) | 1)] * rows

    @staticmethod
    def packed_slot_rows(per_request: int, slot: int) -> int:
        """Return the live rows of segment ``slot`` of a packed build."""
        shift = (per_request + ROWS - 1) // ROWS - 1
        sub = slot & ((1 << shift) - 1)
        return max(0, min(ROWS, per_request - ROWS * sub))

    def reset_seeds(self) -> None:
        """Reset the selector's seed words (stream-ordered)."""
        groups = self.seeds.numel() // ROWS
        stride = self.select_state.numel() // groups
        for group in range(groups):
            first = group * stride + StateWords.SEEDS
            self.select_state[first : first + ROWS].copy_(
                self.seeds[group * ROWS : (group + 1) * ROWS]
            )

    def arguments(self, peers: Peers) -> list[Argument]:
        """Rows, weights, RoPE, position, keys, results, state, capacity,
        peers, blocks.
        """
        state = [
            self.state[self.prefix + n] for n in ("full_epoch", "flags", "select_state")
        ]
        node = f"glm53.layer{self.layer}.attention."
        return [
            self.x.data_ptr(),
            *(self.address(self.prefix + name) for name in self.WEIGHTS),
            self.address(Prologue.FREQ),
            self.address("position"),
            self.address(self.prefix + "index_key_cache"),
            *(self.results[name].data_ptr() for name in self.RESULTS),
            self.state[self.prefix + "selected_ids"].data_ptr(),
            self.final.data_ptr(),
            *(tensor.data_ptr() for tensor in state),
            self.capacity,
            *peers(node + "rows"),
            *peers(node + "selection")[1:],
            self.address(BLOCK_TABLE),
        ]


@dataclass(frozen=True)
class MoePrefetch:
    """The worker's prefetch of the next MoE's expert weights, configured in its flags
    at build.
    """

    delay: int
    rate: int
    stop: int
    #: Experts listed at most (the kernel clamps the count to the same bound).
    list_max: int = 12
    source: int = 1

    def words(self, routed_up: int) -> list[int]:
        """Return the flag words from ``FlagWords.MOE_COUNT`` on."""
        return [
            self.list_max,
            self.delay,
            self.rate,
            self.stop,
            *_words(routed_up),
            self.source,
            0,
        ]


class Worker(Provider):
    """A layer's MLA attention worker: query, KV latent, split attention, O
    projection.
    """

    PREFETCH_FULL: ClassVar[MoePrefetch] = MoePrefetch(
        delay=6_000, rate=3_000_000, stop=34_000
    )
    PREFETCH_SHARED: ClassVar[MoePrefetch] = MoePrefetch(
        delay=6_000, rate=6_000_000, stop=13_000
    )

    def __init__(
        self,
        geometry: Glm53Geometry,
        layer: int,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        xg: torch.Tensor | None,
        placement: Glm53Placement,
    ) -> None:
        super().__init__(geometry, x.device, inputs)
        self.layer: int = layer
        self.rank: int = rank
        self.x: torch.Tensor = x
        self.xg: torch.Tensor | None = xg
        self.shared: bool = geometry.is_shared(layer)
        self.kernel = geometry.worker_kernel(layer, rank)
        self.site = f"layer{layer}.{geometry.worker_op(layer, rank)}"
        probe = "probe." if not self.shared and rank < geometry.selector_ranks else ""
        self.prefix: str = f"layers.{layer}.attention.{probe}"
        sizes = sizes_of(self.kernel)
        f32 = torch.float32.itemsize
        self.result("partial_o", (sizes["kPartialOBytes"] // f32,), torch.float32)
        self.result("partial_lse", (sizes["kPartialLseBytes"] // f32,), torch.float32)
        self.final: torch.Tensor = self.rows("final")
        self.owned(self.prefix + "full_epoch", geometry.rows)
        flags = self.owned(self.prefix + "flags", sizes["kFlagWords"])
        groups = geometry.rows // ROWS
        placed = None
        if self.shared:
            placed = placement.worker_handoff(
                geometry.ffn_kind(layer - 1) == "moe_seam", groups
            )
        self.owned(
            self.prefix + "handoff",
            sizes["kHandoffWords"],
            placed=None if placed is None else placed.handoff,
        )
        if placed is not None:
            self.owned(
                self.prefix + "handoff_mirror",
                sizes["kHandoffWords"],
                placed=placed.mirror,
            )
        stride = sizes["kFlagWords"] // groups  # a row tier: one flags block per group
        for first, values in self.flag_words().items():
            for group in range(groups):
                flags[
                    group * stride + first : group * stride + first + len(values)
                ].copy_(torch.tensor(values, dtype=torch.int32))
        if self.shared:
            self.output = "glm53.shared_layers.attention.rows"
            self.slices.append(Slice(self.output, sizes["kMailboxBytes"]))
        else:
            self.output = f"glm53.layer{layer}.attention.rows"
            push = sizes_of(geometry.mailbox_kernel(layer))["kMailboxBytes"]
            selection = sizes_of(geometry.selector_kernel(0))["kShardSelectionBytes"]
            self.slices += [
                Slice(self.output, push),
                Slice(f"glm53.layer{layer}.attention.selection", selection),
            ]

    def flag_words(self) -> dict[int, list[int]]:
        """The flags' build-time words: the MoE prefetch configuration, a shared layer's
        mailbox index.
        """
        geometry = self.geometry
        words: dict[int, list[int]] = {}
        if self.layer >= geometry.dense_layers:
            full = not self.shared
            prefetch = self.PREFETCH_FULL if full else self.PREFETCH_SHARED
            ffn = f"layers.{self.layer}.ffn."
            words[FlagWords.MOE_COUNT] = prefetch.words(self.address(ffn + "routed_up"))
            words[FlagWords.ROUTER_BIAS] = _words(self.address(ffn + "correction_bias"))
            if full and not geometry.is_pushed_rank(self.layer, self.rank):
                pages = self.address(f"layers.{self.layer}.attention.router_pages")
                words[FlagWords.ROUTER_WEIGHT] = _words(pages)
        if self.shared:
            index = [n for n in range(geometry.layers) if geometry.is_shared(n)].index(
                self.layer
            )
            words[FlagWords.SHARED_INDEX] = [index]
        return words

    def arguments(self, peers: Peers) -> list[Argument]:
        """Head and tail records, XG after a seam MoE, then the paged pool's bytes."""
        prefix, layer = self.prefix, self.layer
        query = (
            f"layers.{layer}.attention.router_pages" if self.shared else "absorb_weight"
        )
        names = (
            "gamma_x",
            "input_weight",
            "input_scales",
            "gamma_q",
            query,
            "absorb_weight",
            "gamma_kv",
        )
        weights = [name if "." in name else prefix + name for name in names]
        pool = self.inputs[f"layers.{layer}.attention.kv_cache"]
        full = max(n for n in range(layer + 1) if not self.geometry.is_shared(n))
        selection = peers(f"glm53.layer{full}.attention.selection")[self.rank]
        state = [self.state[prefix + n] for n in ("handoff",)]
        results = [self.results[n] for n in ("partial_o", "partial_lse")]
        tail = [
            self.address(prefix + n)
            for n in ("kvb_weight", "kvb_scales", "o_weight", "o_scales")
        ]
        epoch, flags = (self.state[prefix + n] for n in ("full_epoch", "flags"))
        arguments: list[Argument] = [
            self.x.data_ptr(),
            *(self.address(name) for name in weights),
            self.address(Prologue.FREQ),
            self.address("position"),
            pool.data_ptr(),
            pool.data_ptr(),
            selection,
            *(tensor.data_ptr() for tensor in (*state, *results)),
            *tail,
            self.x.data_ptr(),
            self.final.data_ptr(),
            epoch.data_ptr(),
            flags.data_ptr(),
            self.rank,
            *peers(self.output),
            self.address(BLOCK_TABLE),
        ]
        if self.xg is not None:
            arguments.append(self.xg.data_ptr())
        return [*arguments, pool.numel() * pool.element_size()]


class Ffn(Provider):
    """A layer's TP8 feed-forward: the dense FFN (the first layers) or the fused MoE."""

    def __init__(
        self,
        geometry: Glm53Geometry,
        layer: int,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        placement: Glm53Placement,
        work: torch.Tensor | None = None,
    ) -> None:
        super().__init__(geometry, x.device, inputs)
        self.layer: int = layer
        self.rank: int = rank
        self.x: torch.Tensor = x
        self.work: torch.Tensor | None = None
        kind = geometry.ffn_kind(layer)
        ffn = f"layers.{layer}.ffn."
        rows = geometry.rows if kind == "dense" else geometry.moe_rows
        placed = mirror = None
        if kind == "dense":
            self.kernel = geometry.dense_kernel
            self.site = f"layer{layer}.dense_ffn_{geometry.row_tag}"
            self.weights: list[str] = [
                ffn + n
                for n in (
                    "gamma",
                    "dense_up",
                    "dense_up_scales",
                    "dense_down",
                    "dense_down_scales",
                )
            ]
            width = geometry.dense_shard + (geometry.hidden if rows > 4 else 0)
            self.result("activation", (rows, width), torch.bfloat16)
            node = f"dense_{geometry.row_tag}"
        else:
            self.kernel = geometry.moe_kernel(layer)
            rows = sizes_of(self.kernel).get("kRows", rows)
            self.site, node = (
                geometry.moe_site(layer),
                self.kernel.removeprefix("fused_").removesuffix("_ckpt_shared"),
            )
            experts = (field.name for field in fields(ExpertInputs))
            names = ("gamma", "router_weight", "correction_bias", *experts)
            self.weights = [ffn + n for n in names]
            top_k = geometry.top_k
            self.result("router_logits", (rows, geometry.experts), torch.float32)
            # One activation plane of the shared expert, then one per routed pick.
            tier = work is not None
            if tier:
                self.work = work
            else:
                planes = 1 + rows * top_k
                self.result(
                    "activation", (planes * rows, geometry.expert_shard), torch.bfloat16
                )
            self.result("route_ids", (rows, top_k), torch.int32)
            self.result("route_weights", (rows, top_k), torch.float32)
            # A row-tier entry keeps plain zeroed flags (no mirrored domain words).
            if not tier:
                placed, mirror = (
                    placement.moe_flags(rows=rows)
                    if rows == 8
                    else placement.moe_flags()
                )
        self.rows("local_o")
        last = layer == geometry.layers - 1
        self.final: torch.Tensor = (
            inputs["segment.hidden"] if last else self.rows("final")
        )
        self.xg: torch.Tensor | None = self.rows("xg") if kind == "moe_seam" else None
        sizes = sizes_of(self.kernel)
        self.owned(ffn + "full_epoch", 1)
        self.owned(ffn + "flags", sizes["kFlagWords"], placed=placed)
        if placed is not None:
            self.owned(ffn + "flags_mirror", sizes["kFlagWords"], placed=mirror)
        self.rows_slice: str = f"glm53.layer{layer}.{node}.rows"
        self.slices.append(Slice(self.rows_slice, sizes["kMailboxBytes"]))

    def scratch_pointers(self) -> list[int]:
        """The result addresses the kernel takes after the weights, in its order; a row-
        tier MoE's ``Work`` follows the router logits where the q4 build's activation
        planes are.
        """
        pointers: list[int] = []
        for name, tensor in self.results.items():
            if name in ("final", "xg"):
                continue
            pointers.append(tensor.data_ptr())
            if name == "router_logits" and self.work is not None:
                pointers.append(self.work.data_ptr())
        return pointers

    def arguments(self, peers: Peers) -> list[Argument]:
        """Rows, weights, results, state, peers; a seam MoE adds the next RMSNorm weight
        and XG.
        """
        ffn = f"layers.{self.layer}.ffn."
        arguments: list[Argument] = [
            self.x.data_ptr(),
            *(self.address(name) for name in self.weights),
            *self.scratch_pointers(),
            self.final.data_ptr(),
            self.state[ffn + "full_epoch"].data_ptr(),
            self.state[ffn + "flags"].data_ptr(),
            self.rank,
            *peers(self.rows_slice),
        ]
        if self.xg is not None:
            gamma = self.address(f"layers.{self.layer + 1}.attention.gamma_x")
            arguments += [gamma, self.xg.data_ptr()]
        return arguments


def program(
    geometry: Glm53Geometry,
    rank: int,
    inputs: Mapping[str, torch.Tensor],
    placement: Glm53Placement,
    layers: int | None = None,
) -> list[Provider]:
    """Return rank ``rank``'s providers in launch order."""
    if not 0 <= rank < geometry.world:
        raise ValueError(f"rank {rank} is not one of {geometry.world} TP ranks")
    prologue = Prologue(geometry, inputs)
    inputs = ChainMap({Prologue.FREQ: prologue.freq}, inputs)  # type: ignore[arg-type]
    providers: list[Provider] = [prologue]
    work: torch.Tensor | None = None
    x, xg = prologue.hidden, None
    for layer in range(geometry.layers if layers is None else layers):
        shared = geometry.is_shared(layer)
        if not shared and rank < geometry.selector_ranks:
            selector = Selector(geometry, layer, rank, inputs, x)
            providers.append(selector)
            x = selector.final
        worker = Worker(
            geometry, layer, rank, inputs, x, xg if shared else None, placement
        )
        kind = geometry.ffn_kind(layer)
        # A W8A8 row-tier MoE: one scratch for every MoE layer of the segment.
        uses_work = kind != "dense" and "Work" in registry.arguments(
            geometry.moe_kernel(layer)
        )
        if uses_work and work is None:
            words = sizes_of(geometry.moe_kernel(layer))["kWorkBytes"] // 4
            work = torch.zeros(words, dtype=torch.int32, device=x.device)
        ffn = Ffn(
            geometry,
            layer,
            rank,
            inputs,
            worker.final,
            placement,
            work if uses_work else None,
        )
        providers += [worker, ffn]
        x, xg = ffn.final, ffn.xg
    return providers
