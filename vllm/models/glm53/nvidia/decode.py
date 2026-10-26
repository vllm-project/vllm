# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 decode segment 1 owned by the model: install, build, capture and dispatch."""

from __future__ import annotations

import logging
from collections import ChainMap
from collections.abc import Container, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re
import torch

from vllm.models.glm53.nvidia import arena as glm53_arena
from vllm.models.glm53.nvidia import kv as glm53_kv
from vllm.models.glm53.nvidia import providers, switch
from vllm.models.glm53.nvidia import rows as glm53_rows
from vllm.models.glm53.nvidia import weights as glm53_weights
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.cute_dsl import registry as glm53_registry
from vllm.models.glm53.nvidia.ops.cute_dsl.row_table import RowTableLayout
from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry, Region

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.block_table import BlockTables

logger = logging.getLogger(__name__)

_CACHE_NAME = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<kind>attn|indexer\.k_cache)$"
)


def segment_enabled() -> bool:
    """Return whether the latency-specialized decode path is on (``switch.py``)."""
    from vllm.models.glm53.nvidia import switch

    return switch.current().enabled


def vllm_caches(
    layers: dict[str, object], wanted: Container[int]
) -> dict[str, torch.Tensor]:
    """Return vLLM's allocated MLA and index-key caches of layers ``wanted`` by module
    name.
    """
    caches: dict[str, torch.Tensor] = {}
    for name, module in layers.items():
        match = _CACHE_NAME.search(name)
        if match is None:
            continue
        if int(match.group("layer")) not in wanted:
            continue
        cache = getattr(module, "kv_cache", None)
        if isinstance(cache, list):
            cache = cache[0] if cache else None
        if isinstance(cache, torch.Tensor) and cache.numel():
            caches[name] = cache
    return caches


def check_mla_caches(
    layers: dict[str, object],
    caches: dict[str, torch.Tensor],
    geometry: Glm53Geometry,
    block_size: int,
) -> None:
    """Refuse MLA caches among ``caches`` that the kernels cannot read."""
    for name, cache in caches.items():
        match = _CACHE_NAME.search(name)
        if match is None or match.group("kind") != "attn":
            continue
        glm53_kv.check_mla_pool(cache, geometry, block_size)
        scale = float(getattr(layers[name], "_k_scale_float", 1.0))
        if scale != 1.0:
            raise RuntimeError(
                f"GLM-5.3 segment: {name} has k_scale {scale}; the segment needs 1.0"
            )


def _layer_caches(
    caches: dict[str, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    mla: dict[int, torch.Tensor] = {}
    index: dict[int, torch.Tensor] = {}
    for name, cache in caches.items():
        match = _CACHE_NAME.search(name)
        assert match is not None
        (mla if match.group("kind") == "attn" else index)[int(match.group("layer"))] = (
            cache
        )
    return mla, index


def rope_table(
    theta: float, capacity: int, rope_dim: int, device: torch.device
) -> torch.Tensor:
    """Return the FP32 ``[capacity, rope_dim]`` RoPE table vLLM's rotary cache holds."""
    inverse = 1.0 / (
        theta
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float, device=device) / rope_dim)
    )
    angles = torch.einsum(
        "i,j -> ij", torch.arange(capacity, dtype=torch.float, device=device), inverse
    )
    cosine = angles.cos().to(torch.bfloat16).float()
    sine = angles.sin().to(torch.bfloat16).float()
    return torch.stack((cosine, sine), dim=-1).reshape(capacity, rope_dim).contiguous()


def kernel_block_table(
    table: torch.Tensor, max_model_len: int, rows: int, block_size: int, key_tile: int
) -> torch.Tensor | None:
    """Return a zero-padded table for the kernels when vLLM's rows are too short, else
    None (``table`` is one request's row, or the rows of several requests).
    """
    entries = (max_model_len + rows - 2) // block_size + 2
    multiple = key_tile // block_size
    entries = -(-entries // multiple) * multiple
    if entries <= table.shape[-1]:
        return None
    return torch.zeros(
        (*table.shape[:-1], entries), dtype=table.dtype, device=table.device
    )


def null_padding_block(
    table: torch.Tensor, position: int, rows: int, block_size: int, live: int = 1
) -> None:
    """Point the block of the dead rows (``rows`` rows run, the first ``live`` of them
    real) at the null block when it is past the last live row's.
    """
    last = (position + rows - 1) // block_size
    if last != (position + live - 1) // block_size and last < table.numel():
        table[last : last + 1].zero_()


def ledger_stop(computed: int, scheduled: int, reach: int, dead: int) -> int:
    """Return where the positions a step may have written end (the KV ledger's stop):
    the request's tokens, the drafter's ``reach`` and, for a lone request on an
    aligned build, its ``dead`` rows past the tokens (0 on a packed tier: its rows
    are the request's tokens, and tail rows store nothing).
    """
    return computed + scheduled + dead + reach


def router_pages(
    weights: dict[str, torch.Tensor], geometry: Glm53Geometry, rank: int
) -> dict[str, torch.Tensor]:
    """Return the router pages of every MoE layer whose worker runs on ``rank``."""
    pages: dict[str, torch.Tensor] = {}
    for layer in range(geometry.dense_layers, geometry.layers):
        if geometry.is_pushed_rank(layer, rank):
            continue
        pages[f"layers.{layer}.attention.router_pages"] = glm53_weights.router_pages(
            weights[f"layers.{layer}.ffn.router_weight"],
            weights[f"layers.{layer}.ffn.gamma"],
            weights[f"layers.{layer}.attention.gamma_x"],
        ).contiguous()
    return pages


@dataclass
class TierState:
    """One row tier's built segment and the vLLM buffers it was built over."""

    segment: Glm53Segment
    tensors: dict[str, torch.Tensor]
    table: torch.Tensor
    padded: torch.Tensor | None
    token_ids: int
    positions: int
    captures: int = 0
    wide: bool = False
    tokens: int = 0
    #: A packed tier: the kernels' rows are vLLM's rows, request after request.
    packed: bool = False

    def sync_block_table(self) -> None:
        """Copy vLLM's block-table rows into the kernels' padded copy, if any (stream-
        ordered).
        """
        if self.padded is not None:
            self.padded[..., : self.table.shape[-1]].copy_(self.table)

    def stage_inputs(self) -> None:
        """Give the kernels their rows: the dead rows of a lone request repeat its last
        token in vLLM's buffer (a packed tier has none: rows past the step's are tail
        rows, which the prologue skips).
        """
        rows = self.segment.geometry.rows
        if not self.packed and self.tokens < rows:
            tokens = self.tensors["token_ids"]
            tokens[self.tokens : rows].copy_(
                tokens[self.tokens - 1 : self.tokens].expand(rows - self.tokens)
            )

    def output(self) -> torch.Tensor:
        """Return the hidden rows vLLM's head reads: the live rows in vLLM's order."""
        return self.tensors["segment.hidden"][: self.tokens]


class SpecializedDecode:
    """The model's segment-1 hook."""

    def __init__(
        self,
        model: torch.nn.Module,
        forms: glm53_weights.SegmentWeights,
        claims: Mapping[int, switch.Build | int] | None = None,
        max_model_len: int = 0,
        rows_per_request: int = glm53_launch.ROWS,
        row_tiers: bool = False,
    ) -> None:
        self.geometry: Glm53Geometry = forms.geometry
        self.claims: dict[int, switch.Build] = {
            tokens: build if isinstance(build, switch.Build) else switch.Build(build)
            for tokens, build in (
                {self.geometry.rows: self.geometry.rows} if claims is None else claims
            ).items()
        }
        self.rows_per_request: int = rows_per_request
        self.lone_rows: int = switch.lone_rows(rows_per_request)
        self.uniform_steps: int = 0
        self.unclaimed_steps: int = 0
        # The step being run is a uniform decode step of a claimed size.
        self._uniform_claim: bool = False
        #: Step sizes logged once each (claimed or stock).
        self._logged: set[int] = set()
        for rows, build in self.claims.items():
            # A tier serves whole requests of tokens, at most its own row count.
            whole = rows <= build.tier and rows % rows_per_request == 0
            if build.tier not in glm53_launch.TIERS or not whole:
                raise ValueError(
                    f"cannot serve {rows} rows with the {build.tier}-row tier"
                )
        self._model: torch.nn.Module = model
        self._forms: glm53_weights.SegmentWeights = forms
        self._tiers: dict[switch.Build, TierState] = {}
        self._kv: glm53_kv.Glm53KvHygiene | None = None
        # Kept across rebuilds (close): the adopted embedding shard.
        self._embedding: glm53_arena.PeerArena | None = None
        self._computed: dict[str, int] = {}
        self._reported: dict[str, int] = {}
        self._scheduled: dict[str, int] = {}
        # Each request's slot in vLLM's persistent block tables, and those tables.
        self._slots: dict[str, int] = {}
        self._block_tables: BlockTables | None = None
        self._mla_group: int = 0
        self._batch: tuple[str, ...] = ()
        self._drafter_partial: bool = False
        self._max_model_len: int = max_model_len
        self._block_size: int = 0
        self._row_tiers: bool = row_tiers
        self._reach: int = 0

    @property
    def built(self) -> bool:
        return bool(self._tiers)

    @property
    def captures(self) -> int:
        return sum(tier.captures for tier in self._tiers.values())

    @property
    def kv(self) -> glm53_kv.Glm53KvHygiene | None:
        return self._kv

    def _tier_of(self, capacity: int | switch.Build | None) -> TierState:
        if capacity is None:
            (tier,) = self._tiers.values()
            return tier
        return self._tiers[min(self._builds_of(capacity), key=lambda b: b.w8a16_rows)]

    def _builds_of(self, capacity: int | switch.Build) -> list[switch.Build]:
        if isinstance(capacity, switch.Build):
            # One build of a tier with several (K = 1: q4 and q8 MoE builds of tier 8).
            return [capacity] if capacity in self._tiers else []
        return [build for build in self._tiers if build.tier == capacity]

    def has_tier(self, capacity: int | switch.Build) -> bool:
        """Whether a build of row capacity ``capacity`` (or that build) is built."""
        return bool(self._builds_of(capacity))

    def sync_block_table(self, capacity: int | switch.Build | None = None) -> None:
        """Copy vLLM's block-table rows into the padded copy of tier ``capacity``'s
        kernels (default: the only tier), if any (stream-ordered).
        """
        self._tier_of(capacity).sync_block_table()

    def shared_tensors(
        self, capacity: int | switch.Build | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the tensors the drafter segment of a tier shares: the RoPE table and
        the block table.
        """
        if capacity is not None and not self.has_tier(capacity):
            return {}
        names = ("rope_table", glm53_launch.BLOCK_TABLE)
        tensors = self._tier_of(capacity).tensors
        return {name: tensors[name] for name in names}

    @property
    def embedding_arena(self) -> glm53_arena.PeerArena:
        assert self._embedding is not None
        return self._embedding

    def add_layer(
        self, layer: int, caches: dict[str, torch.Tensor], reach: int
    ) -> None:
        """Extend the KV hygiene to the drafter's layer, whose steps write ``reach``
        more rows.
        """
        from vllm.distributed import get_tp_group

        assert self._kv is not None
        if any(key[0] == layer for key in self._kv.planes):
            return  # another request tier of the drafter added it
        geometry = self.geometry
        planes = {
            **self._kv.planes,
            **glm53_kv.pool_planes(caches, geometry, self._kv.block_size),
        }
        if any(key[0] != layer for key in planes if key not in self._kv.planes):
            raise ValueError(f"caches of layers other than {layer}")
        pending = self._kv.pending
        selectors = frozenset(range(glm53_registry.SELECTOR_RANKS))
        self._kv = glm53_kv.Glm53KvHygiene(
            planes,
            geometry,
            rank=get_tp_group().rank_in_group,
            block_size=self._kv.block_size,
            partial_writers={
                key: selectors
                for key in planes
                if key[1] in ("index_key", "index_scale")
            },
        )
        self._kv.pending = pending
        self._reach = reach

    def owns_attn_metadata(self, num_tokens: int) -> bool:
        """Return whether a FULL step of ``num_tokens`` tokens replays only claimed
        graphs.
        """
        from vllm.models.glm53.nvidia import draft as glm53_draft

        if self.captures == 0 or num_tokens not in self.claims:
            return False
        return num_tokens == 1 or glm53_draft.claims_draft_prefill(
            num_tokens // self.rows_per_request
        )

    def forward(
        self, input_ids: torch.Tensor | None, positions: torch.Tensor
    ) -> torch.Tensor | None:
        """Claim a captured forward of a served row count; return its residual stream
        rows.
        """
        if input_ids is None or not input_ids.is_cuda:
            return None
        rows = int(input_ids.shape[0])
        build = self.claims.get(rows)
        if not torch.cuda.is_current_stream_capturing():
            tier = None if build is None else self._tiers.get(build)
            if self._uniform_claim and tier is not None and tier.captures:
                # vLLM runs a claimed step eagerly: a stock forward.
                self.unclaimed_steps += 1
            self._uniform_claim = False
            if build is not None and tier is None:
                self._try_build(build, input_ids, positions)
            if self._tiers:
                if self._kv is not None and self._kv.pending:
                    from vllm.distributed import get_tp_group

                    self._kv.complete_pending(self._request_blocks, get_tp_group())
                for tier in self._tiers.values():
                    tier.segment.reset_seeds()
            return None
        if build is None:
            return None
        tier = self._tiers.get(build)  # type: ignore[assignment]
        if tier is None:
            if not self._kv_allocated():
                return None
            raise RuntimeError(
                f"GLM-5.3 segment: vLLM captures a {rows}-row forward before the "
                "segment was built "
                "(no eager warm-up of that size ran with the KV cache allocated)"
            )
        if not self._binds_live_cache(tier):
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures over another KV cache than the segment "
                "was built for "
                "(the KV cache was re-initialized without "
                "Glm53ModelState.initialize_kv_cache)"
            )
        if input_ids.data_ptr() != tier.token_ids:
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures other input_ids than the segment binds"
            )
        if positions.data_ptr() != tier.positions:
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures other positions than the segment binds"
            )
        tier.sync_block_table()
        tier.stage_inputs()
        tier.segment.launch(torch.cuda.current_stream().cuda_stream)
        tier.captures += 1
        return tier.output()

    def _kv_allocated(self) -> bool:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return False
        layers = range(self.geometry.layers)
        mla, _ = _layer_caches(
            vllm_caches(get_forward_context().no_compile_layers, layers)
        )
        return len(mla) == self.geometry.layers

    def _binds_live_cache(self, tier: TierState) -> bool:
        from vllm.forward_context import get_forward_context

        mla, _ = _layer_caches(
            vllm_caches(get_forward_context().no_compile_layers, (0,))
        )
        built = tier.tensors.get("layers.0.attention.kv_cache")
        return 0 in mla and built is not None and mla[0].data_ptr() == built.data_ptr()

    def bind_block_tables(
        self, block_tables: BlockTables, kv_cache_config: KVCacheConfig
    ) -> None:
        """Remember vLLM's persistent block tables and the MLA layers' group: a
        request's blocks stay in its slot's row until the slot is reused.
        """
        self._block_tables = block_tables
        for index, group in enumerate(kv_cache_config.kv_cache_groups):
            if any(_CACHE_NAME.search(name) for name in group.layer_names):
                self._mla_group = index
                return

    def request_added(
        self, request_id: str, num_computed_tokens: int, slot: int | None = None
    ) -> None:
        """Track a request vLLM's runner just added (``Glm53ModelState.add_request``;
        ``slot`` is its row in the persistent tables).
        """
        self._computed[request_id] = self._reported[request_id] = num_computed_tokens
        self._scheduled.pop(request_id, None)
        if slot is not None and self._row_tiers:
            self._slots[request_id] = slot

    def _request_blocks(self, request_id: str) -> torch.Tensor:
        slot = self._slots.get(request_id)
        if slot is not None and self._block_tables is not None:
            return self._block_tables.block_tables[self._mla_group].gpu[slot]
        (tier,) = self._tiers.values()
        return tier.table if tier.table.dim() == 1 else tier.table[0]

    def request_removed(self, request_id: str) -> None:
        """Complete a finished request's index keys on every rank and forget it."""
        if self._kv is not None:
            from vllm.distributed import get_tp_group

            if self._kv.pending.has(request_id):
                self._kv.release(
                    request_id, self._request_blocks(request_id), get_tp_group()
                )
        for table in (self._computed, self._reported, self._scheduled, self._slots):
            table.pop(request_id, None)

    def step(
        self,
        request_ids: list[str],
        num_computed: list[int],
        num_scheduled: list[int],
        num_tokens: int,
    ) -> None:
        """Account one forward before it runs: the rows a replay may write, the row
        table of the tier it replays.
        """
        for request_id, count, scheduled in zip(
            request_ids, num_computed, num_scheduled
        ):
            if self._reported.get(request_id) != count:
                self._computed[request_id] = self._reported[request_id] = count
            elif request_id in self._scheduled:
                self._computed[request_id] += self._scheduled[request_id]
            self._scheduled[request_id] = scheduled
        build = self.claims.get(num_tokens)
        tier = None if build is None else self._tiers.get(build)
        uniform = set(num_scheduled) == {self.rows_per_request}
        self._uniform_claim = uniform and build is not None
        if uniform:
            self.uniform_steps += 1
            self.unclaimed_steps += build is None
            if num_tokens not in self._logged or self.uniform_steps % 1000 == 0:
                self._logged.add(num_tokens)
                logger.debug(
                    "GLM-5.3 uniform decode steps: %d, vLLM's stock forward %d "
                    "(this step: %d tokens, %s)",
                    self.uniform_steps,
                    self.unclaimed_steps,
                    num_tokens,
                    "stock" if build is None else f"tier {build.tier}",
                )
        if build is None:
            self._before_stock_step()
        if tier is None or self._kv is None or self.captures == 0:
            return
        if not uniform:
            return  # not a uniform decode step: vLLM runs it outside the graph
        self._complete_for_stock_drafter(len(request_ids))
        geometry = tier.segment.geometry
        if geometry.max_requests == 1 and geometry.rows > self.rows_per_request:
            assert tier.table.dim() == 1
            null_padding_block(
                tier.table,
                num_computed[0],
                geometry.rows,
                self._block_size,
                self.rows_per_request,
            )
        tier.segment.row_table.set_uniform_step(
            len(request_ids),
            self.rows_per_request if tier.packed else self.lone_rows,
        )
        if tuple(request_ids) != self._batch:
            tier.segment.reset_seeds()
        self._batch = tuple(request_ids)
        if tier.wide:
            # Rows the narrow tiers decoded sit on the writer ranks only.
            from vllm.distributed import get_tp_group

            self._kv.complete_narrow(request_ids, self._request_blocks, get_tp_group())
        for request_id, scheduled in zip(request_ids, num_scheduled):
            # The host's count runs up to two steps ahead of the device.
            start = self._reported[request_id] - 2 * self.rows_per_request
            stop = ledger_stop(
                self._computed[request_id],
                scheduled,
                self._reach,
                0 if tier.packed else self.lone_rows - self.rows_per_request,
            )
            self._kv.pending.decoded(request_id, start, stop)
            if not tier.wide:
                self._kv.narrow.decoded(request_id, start, stop)

    def _before_stock_step(self) -> None:
        self._batch = ()
        self._drafter_partial = False
        if self._kv is not None and self._kv.pending:
            from vllm.distributed import get_tp_group

            self._kv.complete_pending(self._request_blocks, get_tp_group())

    def _complete_for_stock_drafter(self, requests: int) -> None:
        from vllm.models.glm53.nvidia import draft as glm53_draft

        draft = glm53_draft.current()
        if draft is None or not draft.built:
            return
        if draft.tier_of(requests) is not None:
            self._drafter_partial = True
            return
        if self._drafter_partial and self._kv is not None and self._kv.pending:
            from vllm.distributed import get_tp_group

            self._kv.complete_pending(self._request_blocks, get_tp_group())
        self._drafter_partial = False

    def _try_build(
        self, build: switch.Build, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> None:
        from vllm.distributed import get_tp_group
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return
        context = get_forward_context()
        base = self.geometry
        capacity = build.tier
        packed = switch.packed_tier(capacity, self.rows_per_request)
        geometry = base.for_tier(
            capacity,
            (self.rows_per_request if packed else self.lone_rows) - 1,
            packed,
            build.w8a16_rows,
            0 if packed else self.rows_per_request,
        )
        caches = vllm_caches(context.no_compile_layers, range(geometry.layers))
        mla, index = _layer_caches(caches)
        if len(mla) != geometry.layers:
            return  # the profile run, before vLLM allocates its pool
        mla_name = next(
            name for name in caches if name.endswith("layers.0.self_attn.attn")
        )
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or mla_name not in metadata:
            return
        block_table = getattr(metadata[mla_name], "block_table", None)
        if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2:
            raise RuntimeError(
                f"GLM-5.3 segment: {mla_name}'s attention metadata has no block table"
            )
        tokens = max(size for size, claimed in self.claims.items() if claimed == build)
        requests = tokens // self.rows_per_request
        if block_table.shape[0] < requests:
            return  # a forward of this size that is not a full decode batch
        if input_ids.dtype != torch.int32 or not input_ids.is_contiguous():
            raise RuntimeError(
                f"GLM-5.3 segment: vLLM's input_ids are {input_ids.dtype}, not "
                "contiguous int32"
            )
        if positions.dtype != torch.int64 or not positions.is_contiguous():
            raise RuntimeError(
                f"GLM-5.3 segment: vLLM's positions are {positions.dtype}, not "
                "contiguous int64"
            )
        group = get_tp_group()
        rank = group.rank_in_group
        device = mla[0].device
        block_size = int(mla[0].shape[-2])
        one = not self._row_tiers or requests == 1
        table = block_table[0] if one else block_table[:requests]
        key_tile = glm53_launch.sizes_of(geometry.selector_kernel(0))["kKeyTile"]
        padded = kernel_block_table(
            table,
            self._max_model_len,
            self.rows_per_request if packed else self.lone_rows,
            block_size,
            key_tile,
        )
        bound = table if padded is None else padded
        capacity_tokens = bound.shape[-1] * block_size
        theta = float(self._model.config.rope_parameters["rope_theta"])
        weights = self._forms.tensors()
        rows = geometry.rows
        position = positions[:1].view(torch.int32)[:1]
        tensors: dict[str, torch.Tensor] = {
            **weights,
            **router_pages(weights, geometry, rank),
            **{
                f"layers.{layer}.attention.kv_cache": cache
                for layer, cache in mla.items()
            },
            **{
                f"layers.{layer}.attention.index_key_cache": cache
                for layer, cache in index.items()
            },
            glm53_launch.BLOCK_TABLE: bound,
            "token_ids": input_ids.as_strided((rows,), (1,)),
            "position": position,
            "rope_table": rope_table(theta, capacity_tokens, geometry.rope_dim, device),
            "segment.hidden": torch.zeros(
                (geometry.buffer_rows, geometry.hidden),
                dtype=torch.bfloat16,
                device=device,
            ),
        }
        check_mla_caches(context.no_compile_layers, caches, geometry, block_size)
        segment = Glm53Segment(
            geometry,
            rank,
            tensors,
            group.device_group,
            embedding=self._model.embed_tokens.weight
            if self._embedding is None
            else self._embedding,
            requests=requests,
        )
        self._embedding = segment.embedding_arena
        # Create the NCCL communicator now, not inside a request's TTFT.
        group.broadcast(torch.zeros(1, dtype=torch.uint8, device=device), src=0)
        if self._kv is None:
            self._kv = glm53_kv.Glm53KvHygiene.from_caches(
                caches, geometry, rank=rank, block_size=block_size
            )
        self._block_size = block_size
        tier = TierState(
            segment,
            tensors,
            table,
            padded,
            input_ids.data_ptr(),
            positions.data_ptr(),
            wide=geometry.selector_ranks >= geometry.world,
            tokens=tokens,
            packed=packed,
        )
        self._tiers[build] = tier
        kept = table.clone()
        kept_positions = positions[:rows].clone()
        table.zero_()
        positions[:rows].zero_()
        segment.launch(torch.cuda.current_stream().cuda_stream)
        table.copy_(kept)
        positions[:rows].copy_(kept_positions)
        torch.cuda.current_stream().synchronize()

    def close(self) -> None:
        """Drop the segments and their KV-bound state; the next eager forward of a
        served size rebuilds them.
        """
        from vllm.models.glm53.nvidia import draft as glm53_draft

        draft = glm53_draft.current()
        if draft is not None:
            draft.close()
        if self._tiers:
            torch.accelerator.synchronize()
            for tier in self._tiers.values():
                tier.segment.close()
        self._tiers = {}
        self._kv = None
        self._block_tables = None
        self._batch = ()


_CURRENT: SpecializedDecode | None = None


def current() -> SpecializedDecode | None:
    """Return this process's installed segment-1 hook, if any."""
    return _CURRENT


def install(
    model: torch.nn.Module,
    forms: glm53_weights.SegmentWeights | None,
    drafts: int | None,
    *,
    max_num_seqs: int,
    v2_runner: bool,
    max_model_len: int,
) -> None:
    """Install the hook on ``model`` (``Glm53Model``) when the segment is enabled."""
    if not segment_enabled():
        return
    if forms is None:
        raise ValueError(
            "GLM-5.3 latency-specialized decode needs the segment's weight forms"
        )
    tiers = switch.current().row_tiers
    per_step = 1 if drafts is None else drafts + 1
    if not tiers and max_num_seqs != 1:
        raise ValueError("GLM-5.3 latency-specialized decode needs --max-num-seqs 1")
    if not v2_runner:
        # switch.resolve refuses this first; kept for callers that install without it.
        raise ValueError(
            "GLM-5.3 latency-specialized decode needs Model Runner V2 (its model-state "
            "hooks complete the index keys)"
        )
    rows = forms.geometry.rows
    claims: dict[int, switch.Build]
    if tiers:
        claims = switch.tier_claims(max_num_seqs, per_step)
    else:
        claims = {per_step: switch.Build(rows)}
    decode = SpecializedDecode(
        model,
        forms,
        claims,
        max_model_len=max_model_len,
        rows_per_request=per_step,
        row_tiers=bool(tiers),
    )
    model.glm53_segment = decode
    global _CURRENT
    _CURRENT = decode
    logger.info(
        "GLM-5.3 segment 1 installed (row tiers %s, claims %s-token forwards)",
        sorted({build.tier for build in claims.values()}),
        sorted(claims),
    )


def tensor_region(tensor: torch.Tensor) -> Region:
    """Return the bytes a tensor view spans."""
    span = 0
    if tensor.numel():
        last = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        span = (last + 1) * tensor.element_size()
    return Region(tensor.data_ptr(), span)


def adopt_embedding(
    weight: torch.nn.Parameter, geometry: Glm53Geometry, group: ProcessGroup
) -> glm53_arena.PeerArena:
    """Move vLLM's embedding shard into an eight-rank peer arena and re-point ``weight``
    at it (one copy).
    """
    shape = (geometry.vocab_shard, geometry.hidden)
    if (
        weight.dtype != torch.bfloat16
        or tuple(weight.shape) != shape
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"vLLM's embedding shard is {weight.dtype} {tuple(weight.shape)}, not "
            f"contiguous BF16 {shape}"
        )
    arena = glm53_arena.PeerArena(
        [
            glm53_arena.Slice(
                glm53_launch.EMBEDDING_SLICE, weight.numel() * weight.element_size()
            )
        ],
        group,
        {glm53_launch.EMBEDDING_SLICE: weight.data},
    )
    weight.data = (
        arena.view(glm53_launch.EMBEDDING_SLICE).view(torch.bfloat16).view(shape)
    )
    torch.accelerator.empty_cache()  # return vLLM's old shard storage to the device
    return arena


class Glm53Segment:
    """One rank's segment 1: its launch providers in order, bound to the peer arena
    (``providers``).
    """

    def __init__(
        self,
        geometry: Glm53Geometry,
        rank: int,
        tensors: Mapping[str, torch.Tensor],
        group: ProcessGroup,
        layers: int | None = None,
        embedding: torch.nn.Parameter | glm53_arena.PeerArena | None = None,
        requests: int | None = None,
    ) -> None:
        if embedding is not None and "embedding" in tensors:
            raise ValueError(
                "pass either the replicated embedding or vLLM's embedding shard, not "
                "both"
            )
        device = next(iter(tensors.values())).device
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        self.geometry: Glm53Geometry = geometry
        self.row_table: glm53_rows.RowTable = glm53_rows.RowTable(
            RowTableLayout(geometry.rows, geometry.table_requests),
            device,
            packed=geometry.packed,
        )
        self.row_table.set_uniform_step(
            geometry.max_requests if requests is None else requests,
            geometry.drafts + 1,
        )
        tensors = ChainMap({providers.ROW_TABLE_INPUT: self.row_table.tensor}, tensors)  # type: ignore[arg-type]
        self.placement: glm53_arena.Glm53Placement = glm53_arena.Glm53Placement(device)
        self.providers: list[providers.Provider] = providers.program(
            geometry, rank, tensors, self.placement, layers
        )
        slices = {
            item.name: item for provider in self.providers for item in provider.slices
        }
        self.arena: glm53_arena.PeerArena = glm53_arena.PeerArena(
            list(slices.values()), group
        )
        self.embedding_arena: glm53_arena.PeerArena | None = None
        if embedding is not None:
            self.embedding_arena = (
                embedding
                if isinstance(embedding, glm53_arena.PeerArena)
                else adopt_embedding(embedding, geometry, group)
            )
        arenas = [
            arena for arena in (self.arena, self.embedding_arena) if arena is not None
        ]

        def peers(name: str) -> list[int]:
            arena = next(arena for arena in arenas if name in arena.offsets)
            return [base + arena.offsets[name] for base in arena.peer_addresses]

        for provider in self.providers:
            provider.bind(peers)
        self.launches: list[glm53_launch.Launch] = [
            p.launch for p in self.providers if p.launch is not None
        ]
        self.bound: glm53_launch.Bound = glm53_launch.bind(self.launches, device.index)
        self.state: dict[str, torch.Tensor] = {
            k: v for p in self.providers for k, v in p.state.items()
        }
        self.results: dict[str, torch.Tensor] = {
            f"%{p.site}.{name}": tensor
            for p in self.providers
            for name, tensor in p.results.items()
        }

    @property
    def regions(self) -> dict[str, Region]:
        return {
            **{name: tensor_region(t) for name, t in self.results.items()},
            **self.arena.regions(),
        }

    def launch(self, stream: int) -> None:
        """Launch the segment's kernels in order on ``stream`` (capturable)."""
        glm53_launch.run(self.bound, stream)

    def reset_seeds(self) -> None:
        """Reset the selectors' seeds (stream-ordered)."""
        for provider in self.providers:
            if isinstance(provider, providers.Selector):
                provider.reset_seeds()

    def close(self) -> None:
        """Release the peer arena (after every graph that launched the segment is
        destroyed) and give the placed flags and handoffs back to their pools.
        """
        self.arena.close()
        self.placement.release()
