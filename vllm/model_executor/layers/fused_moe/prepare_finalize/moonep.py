# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoonEP (https://github.com/MoonshotAI/MoonEP) prepare/finalize.

BF16 correctness-first integration on top of vLLM's modular kernel
interface.

MoonEP differs from DeepEP-style backends in two ways that shape this
integration:

- ``dispatch`` returns tokens already grouped by expert *row* (a fixed
  ``[NvS, H]`` layout with ``NvS = S x K`` real slots plus padding) together
  with a ``cu_seqlens[2 * epn]`` segment table and an opaque ``plan``. There
  is no per-token topk id tensor after dispatch; the expert compute must be a
  grouped GEMM over ``cu_seqlens`` segments.
- Rows ``[epn, 2 * epn)`` of the weight/segment space are dynamic
  redundant-expert prefetch slots (``epn = E / R`` local experts per rank).
  ``plan.experts_to_copy`` names the source expert of each slot and
  ``Buffer.prefetch_weight`` must run between dispatch and expert compute.

Weight ownership follows MoonEP's contract: each rank keeps only its own
``epn`` experts; the prefetch slots live in one process-global symmetric
pool per projection that every layer shares. See
:class:`MoonEPExpertWeightPools`.

Limitations:
- BF16 / unquantized only, eager only.
- Route weights are applied inside the expert compute and MoonEP's
  ``combine`` performs the K-sum, so ``finalize`` requires
  ``TopKWeightAndReduceNoOP``.
"""

import os
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

MOONEP_DEFAULT_TOKEN_PADDING = 128
MOONEP_DEFAULT_NUM_SMS = 32
# MoonEP's weight-prefetch kernel tiles hidden and intermediate dims by 128.
MOONEP_WEIGHT_TILE = 128


class MoonEPExpertWeights(NamedTuple):
    """One MoE layer's expert weights in MoonEP's compute layout.

    ``gate``/``up``/``down`` are contiguous ``[2 * epn, ...]`` compute views:
    rows ``[0, epn)`` are this rank's own experts (global ids
    ``[ep_rank * epn, (ep_rank + 1) * epn)``), rows ``[epn, 2 * epn)`` alias
    this rank's slice of the process-global prefetch pool, filled by
    ``Buffer.prefetch_weight``. ``*_prefetch_buffer`` are the ``[R, epn, ...]``
    all-rank pool views that ``prefetch_weight`` writes through.

    The prefetch slots never need re-zeroing between calls: the planner
    marks unused slots with ``-1`` in ``plan.experts_to_copy`` (skipped by
    ``prefetch_weight``) and gives them an empty ``cu_seqlens`` segment, so
    stale slot contents are never read; used slots are fully overwritten.
    """

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    gate_prefetch_buffer: torch.Tensor
    up_prefetch_buffer: torch.Tensor
    down_prefetch_buffer: torch.Tensor
    num_local_experts: int

    @property
    def local_gate(self) -> torch.Tensor:
        return self.gate[: self.num_local_experts]

    @property
    def local_up(self) -> torch.Tensor:
        return self.up[: self.num_local_experts]

    @property
    def local_down(self) -> torch.Tensor:
        return self.down[: self.num_local_experts]

    def prefetch_kwargs(self) -> dict[str, torch.Tensor]:
        return dict(
            local_gate_weight=self.local_gate,
            local_up_weight=self.local_up,
            local_down_weight=self.local_down,
            gate_prefetch_buffer=self.gate_prefetch_buffer,
            up_prefetch_buffer=self.up_prefetch_buffer,
            down_prefetch_buffer=self.down_prefetch_buffer,
        )


class _MoonEPPrefetchPool:
    """This rank's prefetch slots for one projection shape, mapped everywhere.

    Allocates one VMM chunk of ``num_local_experts`` expert rows (padded to
    the VMM granularity) and exports it so that:

    - every EP rank maps all ranks' chunks as the ``[R, epn, ...]``
      prefetch-buffer view that ``prefetch_weight`` writes through, and
    - each layer on this rank can map the chunk again, right behind its own
      expert weights, to form the contiguous ``[2 * epn, ...]`` compute view.
    """

    def __init__(
        self,
        expert_shape: tuple[int, ...],
        num_local_experts: int,
        dtype: torch.dtype,
        group: dist.ProcessGroup | None,
        use_fabric: bool,
    ):
        # Private MoonEP helpers, valid at the pinned MOONEP_COMMIT_HASH; see
        # the MoonEPExpertWeightPools docstring before bumping the pin.
        from moonep._C import (  # type: ignore[import-not-found]
            nvl_dist_alloc,
            nvl_release_mem_handle,
        )
        from moonep.buffer import (  # type: ignore[import-not-found]
            _map_nvl_dist_tensor,
            pad_dim0_for_alignment,
        )

        self.num_local_experts = num_local_experts
        self.dtype = dtype
        self.use_fabric = use_fabric
        self.chunk_shape = [
            pad_dim0_for_alignment([num_local_experts, *expert_shape], dtype),
            *expert_shape,
        ]

        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        keepalive, shareable, owned_handle = nvl_dist_alloc(
            shape=self.chunk_shape, dtype=dtype, use_fabric=use_fabric
        )
        nvl_release_mem_handle(owned_handle)

        # Keep a handle for the per-layer compute-view mappings. In fd mode
        # _map_nvl_dist_tensor closes the fd once peers have imported it, so
        # dup it first; a fabric handle can be imported any number of times.
        self._own_fd: int | None = None
        if use_fabric:
            self._shareable = shareable
        else:
            self._own_fd = os.dup(int(shareable.item()))
            self._shareable = torch.tensor([self._own_fd], dtype=torch.int64)

        all_ranks = _map_nvl_dist_tensor(
            self.chunk_shape,
            dtype,
            shareable,
            keepalive,
            rank,
            world_size,
            group,
            use_fabric,
        )

        # The payload sits at the start of each rank's chunk; the rank stride
        # skips the alignment padding, which prefetch_weight allows.
        self.all_rank_view = all_ranks.view(world_size, *self.chunk_shape)[
            :, :num_local_experts
        ]

    def map_local_chunk_before_prefetch_slots(
        self, local_chunk_shareable: torch.Tensor
    ) -> torch.Tensor:
        """Map ``[local chunk, this rank's prefetch chunk]`` back to back.

        Returns the ``[2 * padded_epn, ...]`` mapping. The caller places its
        expert weights at the end of the first chunk so they abut the
        prefetch payload at the start of the second.
        """
        from moonep._C import nvl_dist_map  # type: ignore[import-not-found]

        local = local_chunk_shareable.reshape(1, -1)
        own = self._shareable.reshape(1, -1)
        shareables = torch.cat([local, own])
        if not self.use_fabric:
            shareables = shareables.reshape(-1)

        return nvl_dist_map(
            chunk_shape=self.chunk_shape,
            dtype=self.dtype,
            shareables=shareables.contiguous(),
            local_rank=0,
            world_size=2,
            use_fabric=self.use_fabric,
        )

    def close(self) -> None:
        if self._own_fd is not None:
            os.close(self._own_fd)
            self._own_fd = None
        self.all_rank_view = None


class MoonEPExpertWeightPools:
    """Process-global MoonEP prefetch pools plus per-layer weight placement.

    One instance per EP group (owned by the MoonEP all2all manager). The
    pools are created collectively on first use and shared by every MoE
    layer, so the extra memory is ``epn`` expert weights per projection per
    rank in total, not per layer.

    Why the local experts live in CUDA VMM allocations rather than
    ``torch.empty``: the expert GEMM addresses weights by row in one
    contiguous ``[2 * epn, ...]`` tensor whose second half must be the
    symmetric prefetch chunk (imported and written by every EP rank, shared
    by all layers). Only ``cuMemCreate``-backed physical memory can be mapped
    into a chosen virtual range next to another mapping, and caching
    allocator memory has no such handle, so each layer's local experts are
    copied once into a VMM chunk and mapped directly in front of the
    prefetch chunk. Consequences:

    - Memory profiling: the chunks are outside the torch allocator and show
      up as non-torch memory in vLLM's memory snapshot, like NCCL and MoonEP
      communication buffers, so they are inside the gpu_memory_utilization
      budget; the source parameters are released after the copy.
    - Sleep mode: the chunks are not in the sleep-mode allocator pool, so
      they would neither be offloaded nor discarded; MoonEP is rejected at
      config time when sleep mode is enabled.
    - Weight reloads and pre-processed loads are unsupported, as before.
    - A grouped GEMM that takes separate base pointers for the local and
      prefetch halves would remove the need for VMM-backed local weights;
      that is the tuned grouped GEMM item of RFC #52095.

    Single-inflight invariant: every same-shaped layer's prefetch rows
    ``[epn, 2 * epn)`` alias the same physical slots, so layer N+1's
    ``prefetch_weight`` must not start before layer N's expert GEMMs have
    finished reading them. The current path holds this trivially:
    ``MoonEPPrepareAndFinalize`` is synchronous (``supports_async`` is
    False) and prefetch runs on the main stream right before the compute
    that consumes it. Enabling MoonEP's async prefetch / DBO requires
    per-inflight-layer (double-buffered) pools or explicit cross-layer
    events; see :meth:`check_synchronous_use`.

    Relies on MoonEP's VMM primitives (``moonep._C.nvl_dist_alloc`` /
    ``nvl_dist_map``) and two private helpers in ``moonep.buffer``
    (``_map_nvl_dist_tensor``, ``_use_fabric_for_group``), as of the
    MoonEP commit pinned in ``tools/ep_kernels/install_python_libraries.sh``
    (``MOONEP_COMMIT_HASH``). Re-check this class against
    ``moonep/buffer.py`` and MoonEP's weight-buffer README section whenever
    that pin moves: the chunk layout assumed here (local weights at the end
    of their VMM block, prefetch payload at the start of the next) is what
    makes the ``[2 * epn]`` view contiguous.
    """

    def __init__(self, group: dist.ProcessGroup | None):
        self.group = group
        self._pools: dict[tuple[str, tuple[int, ...], torch.dtype], Any] = {}
        self._use_fabric: bool | None = None

    @staticmethod
    def check_synchronous_use(prepare_finalize: mk.FusedMoEPrepareAndFinalize):
        # A hard check rather than assert: violating this corrupts expert
        # weights silently, and assert is stripped under python -O.
        if prepare_finalize.supports_async():
            raise RuntimeError(
                "MoonEPExpertWeightPools shares one prefetch pool across layers; "
                "async prefetch requires per-inflight-layer pools or explicit "
                "cross-layer events"
            )

    def build_expert_weights(
        self, w13_local: torch.Tensor, w2_local: torch.Tensor
    ) -> MoonEPExpertWeights:
        """Place a layer's local experts into MoonEP's compute layout.

        Args:
            w13_local: ``[epn, 2I, H]`` BF16, gate rows first.
            w2_local: ``[epn, H, I]`` BF16.

        """
        epn, intermediate_size = _validate_local_expert_weights(w13_local, w2_local)

        gate_local = w13_local[:, :intermediate_size, :]
        up_local = w13_local[:, intermediate_size:, :]

        gate, gate_prefetch = self._place_local_experts("gate", gate_local)
        up, up_prefetch = self._place_local_experts("up", up_local)
        down, down_prefetch = self._place_local_experts("down", w2_local)

        return MoonEPExpertWeights(
            gate=gate,
            up=up,
            down=down,
            gate_prefetch_buffer=gate_prefetch,
            up_prefetch_buffer=up_prefetch,
            down_prefetch_buffer=down_prefetch,
            num_local_experts=epn,
        )

    def close(self) -> None:
        for pool in self._pools.values():
            pool.close()
        self._pools.clear()

    def _place_local_experts(
        self, projection: str, local_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy ``local_weight`` ``[epn, ...]`` into a VMM chunk mapped right in
        front of this rank's prefetch slots.

        Returns:
            The ``[2 * epn, ...]`` compute view (local rows, then the
            prefetch slots) and the ``[R, epn, ...]`` all-rank prefetch view.

        """
        epn = local_weight.size(0)
        expert_shape = tuple(local_weight.shape[1:])
        pool = self._get_or_create_prefetch_pool(
            projection, expert_shape, epn, local_weight.dtype
        )

        local_chunk, local_chunk_shareable = _allocate_vmm_chunk(
            pool.chunk_shape, local_weight.dtype, pool.use_fabric
        )
        try:
            both_chunks = pool.map_local_chunk_before_prefetch_slots(
                local_chunk_shareable
            )
        finally:
            if not pool.use_fabric:
                os.close(int(local_chunk_shareable.item()))
        # The compute-view mapping now keeps the physical chunk alive, so the
        # allocation's own mapping can go.
        del local_chunk

        # Local weights occupy the end of their (padded) chunk so that they
        # abut the prefetch payload at the start of the pool chunk.
        padded_epn = pool.chunk_shape[0]
        compute_view = both_chunks[padded_epn - epn : padded_epn + epn]
        compute_view[:epn].copy_(local_weight)

        return compute_view, pool.all_rank_view

    def _get_or_create_prefetch_pool(
        self,
        projection: str,
        expert_shape: tuple[int, ...],
        epn: int,
        dtype: torch.dtype,
    ) -> _MoonEPPrefetchPool:
        key = (projection, (epn, *expert_shape), dtype)
        pool = self._pools.get(key)
        if pool is not None:
            return pool

        if self._use_fabric is None:
            # Private MoonEP helper, valid at the pinned MOONEP_COMMIT_HASH.
            from moonep.buffer import (  # type: ignore[import-not-found]
                _use_fabric_for_group,
            )

            self._use_fabric = _use_fabric_for_group(self.group)

        pool = _MoonEPPrefetchPool(
            expert_shape, epn, dtype, self.group, self._use_fabric
        )
        self._pools[key] = pool
        return pool


def _validate_local_expert_weights(
    w13_local: torch.Tensor, w2_local: torch.Tensor
) -> tuple[int, int]:
    """Check the local expert shapes and return ``(epn, intermediate_size)``."""
    if w13_local.dtype != torch.bfloat16 or w2_local.dtype != torch.bfloat16:
        raise NotImplementedError("MoonEP supports BF16 weights only.")

    epn, two_i, hidden_size = w13_local.shape
    intermediate_size = two_i // 2
    if tuple(w2_local.shape) != (epn, hidden_size, intermediate_size):
        raise ValueError(
            f"w2_weight shape {tuple(w2_local.shape)} does not match "
            f"w13_weight shape {tuple(w13_local.shape)}"
        )

    if hidden_size % MOONEP_WEIGHT_TILE or intermediate_size % MOONEP_WEIGHT_TILE:
        raise ValueError(
            "MoonEP weight prefetch requires hidden_size and intermediate_size "
            f"to be multiples of {MOONEP_WEIGHT_TILE}; got H={hidden_size}, "
            f"I={intermediate_size}"
        )

    return epn, intermediate_size


def _allocate_vmm_chunk(
    chunk_shape: list[int], dtype: torch.dtype, use_fabric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an exportable VMM chunk.

    Returns the allocation's own mapping and its shareable handle. The
    physical memory is released once no mapping references it, so the
    caller must hold the returned mapping until it has mapped the chunk
    elsewhere.
    """
    from moonep._C import (  # type: ignore[import-not-found]
        nvl_dist_alloc,
        nvl_release_mem_handle,
    )

    chunk, shareable, owned_handle = nvl_dist_alloc(
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric
    )
    nvl_release_mem_handle(owned_handle)
    return chunk, shareable


MOONEP_MIN_DISPATCH_TOKENS = 128


class MoonEPBufferPool:
    """Lazily creates ``moonep.Buffer`` instances at power-of-two capacities.

    ``Buffer`` fixes its token capacity ``S`` at construction, so a single
    full-capacity buffer would pad every dispatch to
    ``max_num_batched_tokens``. The pool instead serves the smallest
    power-of-two capacity that fits the step's token count, so decode-sized
    batches dispatch a few hundred slots rather than the full capacity.

    Selection must be identical on every EP rank (dispatch is collective and
    ``Buffer`` construction is itself a collective): callers derive the token
    count from ``dp_metadata`` (the max across DP ranks) so all ranks create
    and pick the same buffer at the same step.
    """

    def __init__(self, buffer_kwargs: dict[str, Any], max_tokens_per_rank: int):
        self._kwargs = {k: v for k, v in buffer_kwargs.items() if k != "S"}
        self.max_tokens_per_rank = max_tokens_per_rank
        self._buffers: dict[int, Any] = {}

    def capacity_for(self, num_tokens: int) -> int:
        if num_tokens >= self.max_tokens_per_rank:
            return self.max_tokens_per_rank
        cap = 1 << max(num_tokens - 1, 0).bit_length()
        return min(max(cap, MOONEP_MIN_DISPATCH_TOKENS), self.max_tokens_per_rank)

    def get(self, num_tokens: int) -> tuple[int, Any]:
        """Return ``(capacity, buffer)`` for the given step token count."""
        from moonep import Buffer  # type: ignore[import-not-found]

        capacity = self.capacity_for(num_tokens)
        buffer = self._buffers.get(capacity)
        if buffer is None:
            buffer = Buffer(S=capacity, **self._kwargs)
            self._buffers[capacity] = buffer
        return capacity, buffer

    def destroy(self) -> None:
        for buffer in self._buffers.values():
            buffer.destroy()
        self._buffers.clear()


class MoonEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using MoonEP balanced dispatch/combine.

    ``prepare`` pads the batch to the selected buffer's static token
    capacity, dispatches, runs ``prefetch_weight`` for the planned redundant
    experts, and stashes the ``plan`` for ``finalize`` (the same pattern
    DeepEP-HT uses for its handle). Downstream expert compute must consume
    the expert-grouped ``[NvS, H]`` layout via ``cu_seqlens``.
    """

    def __init__(
        self,
        buffer_pool: "MoonEPBufferPool",
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_global_experts: int,
        expert_weights: MoonEPExpertWeights | None = None,
    ):
        super().__init__()
        self.buffer_pool = buffer_pool
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_global_experts = num_global_experts
        self.expert_weights = expert_weights
        self._fused_experts: Any = None

        # dispatch state consumed by finalize (and the expert runner)
        self._plan: Any = None
        self._cu_seqlens: torch.Tensor | None = None
        self._num_tokens: int = 0
        self._buffer: Any = None  # the pool buffer used by the current step

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        # The expert weights are attached to the experts by their
        # process_weights_after_loading hook (after this runs), so keep the
        # reference and resolve them lazily in prepare().
        self._fused_experts = fused_experts

    def _resolve_expert_weights(self) -> MoonEPExpertWeights:
        if self.expert_weights is None and self._fused_experts is not None:
            weights = getattr(self._fused_experts, "expert_weights", None)
            if isinstance(weights, MoonEPExpertWeights):
                self.expert_weights = weights
        # Redundant experts' weights must be in the prefetch slots before
        # the expert compute reads them; skipping prefetch silently corrupts
        # output.
        assert self.expert_weights is not None, (
            "MoonEPPrepareAndFinalize: expert weights not available (the "
            "experts' process_weights_after_loading has not run)"
        )
        return self.expert_weights

    @property
    def num_dispatched_slots(self) -> int:
        """``NvS`` of the buffer selected by the current step's prepare()."""
        assert self._buffer is not None, "prepare() has not been called"
        return int(self._buffer._ctx["NvS"])

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        # combine returns the fully weighted+reduced token-major output
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    @property
    def cu_seqlens(self) -> torch.Tensor:
        assert self._cu_seqlens is not None, "prepare() has not been called"
        return self._cu_seqlens

    @property
    def plan(self) -> Any:
        assert self._plan is not None, "prepare() has not been called"
        return self._plan

    def _pad_to_capacity(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_tokens = a1.shape[0]
        if num_tokens > capacity:
            raise ValueError(
                f"MoonEP static buffer capacity exceeded: {num_tokens} tokens "
                f"> S={capacity}."
            )
        a1 = a1.contiguous()
        topk_ids = topk_ids.to(dtype=torch.int32)
        topk_weights = topk_weights.to(dtype=torch.float32)
        # MoonEP's planner has no sentinel for invalid expert ids, so map
        # negative ids (padded/invalid slots from the engine) to expert 0
        # with zero route weight — the same convention as the capacity
        # padding rows below. Wasted compute, never wrong output.
        invalid = topk_ids < 0
        topk_ids = torch.where(invalid, torch.zeros_like(topk_ids), topk_ids)
        topk_weights = torch.where(
            invalid, torch.zeros_like(topk_weights), topk_weights
        ).contiguous()
        topk_ids = topk_ids.contiguous()
        if num_tokens == capacity:
            return a1, topk_ids, topk_weights, num_tokens
        pad = capacity - num_tokens
        return (
            F.pad(a1, (0, 0, 0, pad)),
            F.pad(topk_ids, (0, 0, 0, pad)),
            F.pad(topk_weights, (0, 0, 0, pad)),
            num_tokens,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if a1.dtype != torch.bfloat16:
            raise NotImplementedError("MoonEP supports BF16 hidden states only.")
        if quant_config.quant_dtype is not None:
            raise NotImplementedError("MoonEP does not support quantized dispatch.")
        assert num_experts == self.num_global_experts
        assert self._plan is None, (
            "MoonEPPrepareAndFinalize.prepare() called again before finalize()"
        )
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input requires top-1 routing"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Buffer choice must be rank-symmetric: use the max token count
        # across DP ranks (every rank sees the same dp_metadata), falling
        # back to the local count outside a forward context (tests).
        num_tokens = a1.shape[0]
        try:
            dp_meta = get_forward_context().dp_metadata
        except AssertionError:
            dp_meta = None
        if dp_meta is not None:
            num_tokens = int(dp_meta.num_tokens_across_dp_cpu.max())
        capacity, self._buffer = self.buffer_pool.get(num_tokens)

        a1, topk_ids, topk_weights, self._num_tokens = self._pad_to_capacity(
            a1, topk_ids, topk_weights, capacity
        )
        tokens_per_expert = torch.bincount(
            topk_ids.reshape(-1).to(dtype=torch.int64),
            minlength=num_experts,
        ).to(dtype=torch.int32)

        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = self._buffer.dispatch(
            a1,
            topk_weights,
            topk_ids,
            tokens_per_expert,
        )
        self._plan = plan
        self._cu_seqlens = cu_seqlens

        # Synchronous on purpose: the prefetch slots are shared by every
        # layer (see MoonEPExpertWeightPools), so this must complete on the
        # main stream before the expert GEMMs and before the next layer's
        # prefetch. Do not switch to async_finish=True without giving each
        # in-flight layer its own slots.
        MoonEPExpertWeightPools.check_synchronous_use(self)
        self._buffer.prefetch_weight(
            plan=plan, **self._resolve_expert_weights().prefetch_kwargs()
        )

        # Segment sizes per [2 * epn] weight row. NOTE: NvS rows are
        # expert-grouped rather than token-major — only MoonEP-aware expert
        # implementations can consume this activation layout.
        expert_num_tokens = torch.diff(cu_seqlens, prepend=cu_seqlens.new_zeros(1))
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        )

        # MoonEP has no post-dispatch per-token topk ids/weights; route
        # weights come back in NvS order.
        return hidden_nvsh, None, expert_tokens_meta, None, route_weights_nvs

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # Route weights are applied in the expert compute and MoonEP's
        # combine performs the K-sum, so any weight/reduce work in finalize
        # means MoonEP was paired with the wrong kind of experts.
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP), (
            "MoonEP requires TopKWeightAndReduceNoOP, got "
            f"{type(weight_and_reduce_impl).__name__}"
        )
        combined, _, _ = self._buffer.combine(
            plan=self.plan,
            hidden_nvsh=fused_expert_output,
            route_weights_nvs=None,
        )
        output.copy_(combined[: self._num_tokens])
        self._plan = None
        self._cu_seqlens = None
