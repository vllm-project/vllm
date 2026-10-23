# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 decode segment 1 owned by the model: install, build, capture and dispatch."""

from __future__ import annotations

import logging
import re
from collections.abc import Container, Mapping
from typing import TYPE_CHECKING

import torch

from vllm.models.glm53.nvidia import arena as glm53_arena
from vllm.models.glm53.nvidia import kv as glm53_kv
from vllm.models.glm53.nvidia import providers
from vllm.models.glm53.nvidia import weights as glm53_weights
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry, Region

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

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
    """Return a zero-padded table for the kernels when vLLM's row is too short, else
    None.
    """
    entries = (max_model_len + rows - 2) // block_size + 2
    multiple = key_tile // block_size
    entries = -(-entries // multiple) * multiple
    if entries <= table.numel():
        return None
    return torch.zeros(entries, dtype=table.dtype, device=table.device)


def null_padding_block(
    table: torch.Tensor, position: int, rows: int, block_size: int
) -> None:
    """Point the block of padding rows p+1..p+rows-1 at the null block when it is past
    p's.
    """
    last = (position + rows - 1) // block_size
    if last != position // block_size and last < table.numel():
        table[last : last + 1].zero_()


def router_pages(
    weights: dict[str, torch.Tensor], geometry: Glm53Geometry, rank: int
) -> dict[str, torch.Tensor]:
    """Return the router pages of every MoE layer whose worker runs on ``rank``."""
    pages: dict[str, torch.Tensor] = {}
    for layer in range(geometry.dense_layers, geometry.layers):
        if not geometry.is_shared(layer) and rank < geometry.selector_ranks:
            continue
        pages[f"layers.{layer}.attention.router_pages"] = glm53_weights.router_pages(
            weights[f"layers.{layer}.ffn.router_weight"],
            weights[f"layers.{layer}.ffn.gamma"],
            weights[f"layers.{layer}.attention.gamma_x"],
        ).contiguous()
    return pages


class SpecializedDecode:
    """The model's segment-1 hook."""

    def __init__(
        self,
        model: torch.nn.Module,
        forms: glm53_weights.SegmentWeights,
        claim: int | None = None,
        max_model_len: int = 0,
    ) -> None:
        self.geometry: Glm53Geometry = forms.geometry
        self.rows: int = self.geometry.rows
        claim = self.rows if claim is None else claim
        if claim not in (1, self.rows):
            raise ValueError(
                f"the segment serves T = {self.rows} rows or one padded row, not "
                f"{claim}"
            )
        self.claim: int = claim
        self._model: torch.nn.Module = model
        self._forms: glm53_weights.SegmentWeights = forms
        self._segment: Glm53Segment | None = None
        self._kv: glm53_kv.Glm53KvHygiene | None = None
        # Kept across rebuilds (close): the adopted embedding shard.
        self._embedding: glm53_arena.PeerArena | None = None
        self._computed: dict[str, int] = {}
        self._reported: dict[str, int] = {}
        self._scheduled: dict[str, int] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._max_model_len: int = max_model_len
        # vLLM's gathered block-table row, and the kernels' padded copy of it (or None).
        self._table: torch.Tensor | None = None
        self._padded: torch.Tensor | None = None
        self._block_size: int = 0
        self._token_ids: int = 0
        self.captures: int = 0
        self._positions: int | None = None

    @property
    def built(self) -> bool:
        return self._segment is not None

    @property
    def kv(self) -> glm53_kv.Glm53KvHygiene | None:
        """Return the KV hygiene (None before the build)."""
        return self._kv

    def forward(
        self, input_ids: torch.Tensor | None, positions: torch.Tensor
    ) -> torch.Tensor | None:
        """Claim a captured forward of ``claim`` tokens; return its residual stream
        rows.
        """
        if input_ids is None or not input_ids.is_cuda:
            return None
        rows = int(input_ids.shape[0])
        if not torch.cuda.is_current_stream_capturing():
            if self._segment is None and rows == self.claim:
                self._try_build(input_ids, positions)
            if self._segment is not None:
                if self._kv is not None and self._kv.pending:
                    from vllm.distributed import get_tp_group

                    self._kv.complete_pending(self._request_blocks, get_tp_group())
                self._segment.reset_seeds()
            return None
        if rows != self.claim:
            return None
        if self._segment is None:
            if not self._kv_allocated():
                return None
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures a T-row forward before the segment was "
                "built "
                "(no eager warm-up of that size ran with the KV cache allocated)"
            )
        if not self._binds_live_cache():
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures over another KV cache than the segment "
                "was built for "
                "(the KV cache was re-initialized without "
                "Glm53ModelState.initialize_kv_cache)"
            )
        if input_ids.data_ptr() != self._token_ids:
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures other input_ids than the segment binds"
            )
        if positions.data_ptr() != self._positions:
            raise RuntimeError(
                "GLM-5.3 segment: vLLM captures other positions than the segment binds"
            )
        if self._padded is not None:
            assert self._table is not None
            self._padded[: self._table.numel()].copy_(self._table)
        if self.claim == 1:
            tokens = self._tensors["token_ids"]
            tokens[1:].copy_(tokens[:1].expand(self.rows - 1))
        self._segment.launch(torch.cuda.current_stream().cuda_stream)
        self.captures += 1
        return self._tensors["segment.hidden"][:rows]

    def _kv_allocated(self) -> bool:
        """Whether vLLM's forward context binds every target layer's MLA cache."""
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

    def _binds_live_cache(self) -> bool:
        """Whether layer 0's MLA cache bound in vLLM's layers is the one the segment was
        built over.
        """
        from vllm.forward_context import get_forward_context

        mla, _ = _layer_caches(
            vllm_caches(get_forward_context().no_compile_layers, (0,))
        )
        built = self._tensors.get("layers.0.attention.kv_cache")
        return 0 in mla and built is not None and mla[0].data_ptr() == built.data_ptr()

    def request_added(self, request_id: str, num_computed_tokens: int) -> None:
        """Track a request vLLM's runner just added
        (``Glm53ModelState.add_request``).
        """
        self._computed[request_id] = self._reported[request_id] = num_computed_tokens
        self._scheduled.pop(request_id, None)

    def _request_blocks(self, request_id: str) -> torch.Tensor:
        """The physical blocks of the one request the segment serves (B = 1), on the
        device.
        """
        assert self._table is not None
        return self._table

    def request_removed(self, request_id: str) -> None:
        """Complete a finished request's index keys on every rank and forget it."""
        if self._kv is not None:
            from vllm.distributed import get_tp_group

            if self._kv.pending.has(request_id):
                self._kv.release(
                    request_id, self._request_blocks(request_id), get_tp_group()
                )
        for table in (self._computed, self._reported, self._scheduled):
            table.pop(request_id, None)

    def step(
        self,
        request_ids: list[str],
        num_computed: list[int],
        num_scheduled: list[int],
        num_tokens: int,
    ) -> None:
        """Account one forward before it runs: the rows a T-row replay may write."""
        for request_id, count, scheduled in zip(
            request_ids, num_computed, num_scheduled
        ):
            if self._reported.get(request_id) != count:
                self._computed[request_id] = self._reported[request_id] = count
            elif request_id in self._scheduled:
                self._computed[request_id] += self._scheduled[request_id]
            self._scheduled[request_id] = scheduled
        if (
            self._kv is None
            or self.captures == 0
            or len(request_ids) != 1
            or num_tokens != self.claim
        ):
            return
        (request_id,) = request_ids
        if self.claim == 1:
            assert self._table is not None
            null_padding_block(
                self._table, num_computed[0], self.rows, self._block_size
            )
        # The host's count runs up to two steps ahead of the device.
        start = self._reported[request_id] - 2 * self.rows
        self._kv.pending.decoded(
            request_id, start, self._computed[request_id] + num_scheduled[0]
        )

    def _try_build(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Build the segment once vLLM's KV cache and metadata exist."""
        from vllm.distributed import get_tp_group
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return
        context = get_forward_context()
        geometry = self.geometry
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
        table = block_table[0]
        key_tile = glm53_launch.sizes_of(geometry.selector_kernel(0))["kKeyTile"]
        padded = kernel_block_table(
            table, self._max_model_len, self.rows, block_size, key_tile
        )
        bound = table if padded is None else padded
        capacity = bound.numel() * block_size
        theta = float(self._model.config.rope_parameters["rope_theta"])
        weights = self._forms.tensors()
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
            "token_ids": input_ids.as_strided((self.rows,), (1,)),
            "position": positions[:1].view(torch.int32)[:1],
            "rope_table": rope_table(theta, capacity, geometry.rope_dim, device),
            "segment.hidden": torch.zeros(
                (self.rows, geometry.hidden), dtype=torch.bfloat16, device=device
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
        )
        self._embedding = segment.embedding_arena
        # Create the NCCL communicator now, not inside a request's TTFT.
        group.broadcast(torch.zeros(1, dtype=torch.uint8, device=device), src=0)
        self._kv = glm53_kv.Glm53KvHygiene.from_caches(
            caches, geometry, rank=rank, block_size=block_size
        )
        self._tensors = tensors
        self._table, self._padded = table, padded
        self._block_size = block_size
        self._token_ids = input_ids.data_ptr()
        self._positions = positions.data_ptr()
        self._segment = segment
        kept = table.clone()
        kept_position = positions[:1].clone()
        table.zero_()
        positions[:1].zero_()
        segment.launch(torch.cuda.current_stream().cuda_stream)
        table.copy_(kept)
        positions[:1].copy_(kept_position)
        torch.cuda.current_stream().synchronize()

    def close(self) -> None:
        """Drop the segment and its KV-bound state; the next eager T-row forward
        rebuilds it.
        """
        if self._segment is not None:
            torch.cuda.synchronize()
            self._segment.close()
        self._segment = None
        self._kv = None
        self._tensors = {}
        self._table = self._padded = None
        self._token_ids = 0
        self._positions = None
        self.captures = 0


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
    if max_num_seqs != 1:
        # switch.resolve refuses this first.
        raise ValueError("GLM-5.3 latency-specialized decode needs --max-num-seqs 1")
    if not v2_runner:
        # switch.resolve refuses this first; kept for callers that install without it.
        raise ValueError(
            "GLM-5.3 latency-specialized decode needs Model Runner V2 (its model-state "
            "hooks complete the index keys)"
        )
    claim = 1 if drafts is None else drafts + 1
    rows = forms.geometry.rows
    decode = SpecializedDecode(model, forms, claim=claim, max_model_len=max_model_len)
    model.glm53_segment = decode
    global _CURRENT
    _CURRENT = decode
    logger.info(
        "GLM-5.3 segment 1 installed (T = %d, claims %d-token forwards)", rows, claim
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
    torch.cuda.empty_cache()  # return vLLM's old shard storage to the device
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
    ) -> None:
        if embedding is not None and "embedding" in tensors:
            raise ValueError(
                "pass either the replicated embedding or vLLM's embedding shard, not "
                "both"
            )
        device = next(iter(tensors.values())).device
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
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
            """Slice ``name``'s address on every rank."""
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
        """Every result and arena slice by name (the multi-rank tests compare them)."""
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
