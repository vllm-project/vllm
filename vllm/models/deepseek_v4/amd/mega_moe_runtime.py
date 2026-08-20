# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide MegaMoEV2 runtime for the ROCm DeepSeek V4 path.

aiter's ``MegaMoEV2`` owns two very different kinds of state:

* **Symmetric (mori shmem) buffers** -- the dispatch receive staging, the P2P
  handshake flags, and the combine output. These are sized from
  ``(world_size, max_tok_per_rank, hidden, experts_per_rank, topk)`` and are
  completely independent of *which* expert weights are in play. At
  ``max_tok_per_rank=16384`` they run to ~7.7 GiB per rank.
* **Weight pointers** -- ``w1``/``w1_scale``/``w2``/``w2_scale``, four plain
  attributes that the kernels read at launch time.

DeepSeek V4 has 61 layers. Allocating the symmetric buffers per layer is not
merely wasteful, it is impossible: 61 x 7.7 GiB does not fit on any GPU. But
because the buffers do not depend on the weights, one instance can serve every
layer as long as the weight attributes are swapped before each launch. That is
what this module does -- a single cached ``MegaMoEV2`` keyed on the shapes that
actually size the buffers, plus a ``bind_weights`` context manager that points
it at one layer's weights for the duration of a call.

This is safe because MoE layers execute sequentially within a forward pass and
the kernels consume the weight pointers synchronously at launch. It would not
be safe under concurrent execution of two MoE layers on the same stream, which
vLLM does not do.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# aiter's fixed-slot dispatch path is selected when max_tok_per_rank <= 255.
# Above that the compact path is used, whose symmetric capacity grows as
# world_size * mtpr * topk rather than experts_per_rank * ll_cap.
_FIXED_SLOT_MAX_MTPR = 255

# all2all backends whose manager brings up the mori symmetric heap itself.
# "flydsl_intranode" is the overlay's backend, which also calls
# shmem_torch_process_group_init.
_MORI_ALL2ALL_BACKENDS = frozenset(
    {"mori_high_throughput", "mori_low_latency", "flydsl_intranode"}
)


@dataclass(frozen=True)
class MegaMoEShape:
    """The parameters that size the symmetric buffers."""

    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tok_per_rank: int
    swiglu_limit: float


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def resolve_max_tok_per_rank(max_num_tokens: int) -> int:
    """Round a scheduler token budget up to a power of two.

    ``MegaMoEV2`` requires a power-of-two ``max_tok_per_rank`` and rejects any
    forward pass carrying more tokens than that, so this has to round up rather
    than down.
    """
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    return _next_power_of_two(max_num_tokens)


def estimate_symmetric_bytes(shape: MegaMoEShape) -> int:
    """Approximate the per-rank mori shmem footprint for ``shape``.

    Mirrors the capacity arithmetic in aiter's
    ``FlyDSLDispatchCombineIntraNodeOp``. Used only for logging and for the
    up-front sanity check -- the authoritative allocation happens inside aiter.
    """
    world = shape.world_size
    mtpr = shape.max_tok_per_rank
    epr = shape.experts // world
    compact = mtpr > _FIXED_SLOT_MAX_MTPR
    unit = 128 if compact else 32

    if compact:
        num_valid_max = world * mtpr * shape.topk + epr * unit
    else:
        ll_cap = ((world * mtpr + unit - 1) // unit) * unit
        num_valid_max = epr * ll_cap + 256

    # fp8 dispatch payload, e8m0 group scales (packed to int32), the int32/
    # float32 side tables, and the stage-1 fp8 output staging buffer.
    scale_i32_per_row = ((shape.model_dim // 32) + 3) // 4
    per_row_bytes = (
        shape.model_dim  # rx_em, one byte per fp8 element
        + scale_i32_per_row * 4  # scale_em
        + 3 * 4  # idx_em, wts_em, srcmap_em
        + shape.inter_dim  # stage-1 fp8 output
    )
    return num_valid_max * per_row_bytes


class MegaMoERuntime:
    """Owns the one shared ``MegaMoEV2`` instance for this process."""

    _instances: dict[MegaMoEShape, MegaMoERuntime] = {}

    def __init__(self, shape: MegaMoEShape, rank: int):
        self.shape = shape
        self.rank = rank
        self._moe = None
        self._bound: object | None = None

    @classmethod
    def get(cls, shape: MegaMoEShape, rank: int) -> MegaMoERuntime:
        runtime = cls._instances.get(shape)
        if runtime is None:
            runtime = cls(shape, rank)
            cls._instances[shape] = runtime
        elif runtime.rank != rank:
            raise RuntimeError(
                f"MegaMoERuntime cached for rank {runtime.rank} but requested "
                f"for rank {rank}; the runtime is per-process."
            )
        return runtime

    def _ensure_shmem_initialized(self) -> None:
        """Bring up mori's symmetric heap if the all2all manager has not.

        Two hazards, pulling in opposite directions:

        * Initializing twice is not a no-op. ``shmem_torch_process_group_init``
          does a ``dist.broadcast_object_list`` on the EP CPU group and builds
          a socket bootstrap *before* the "already initialized" guard inside
          ``libmori_shmem.so`` short-circuits, so a partial re-entry deadlocks;
          and re-registering the process group silently rebinds the ``"mori"``
          registry entry rather than raising.
        * Probing whether it is already up is not possible from Python.
          ``shmem_npes()`` **segfaults** when the heap has not been initialized
          -- it dereferences the uninitialized global state singleton, which no
          ``except`` can catch. Neither is there an ``is_initialized()``.

        So decide from configuration instead of asking mori. Within a vLLM
        process the heap has exactly one other possible owner:
        ``MoriAll2AllManager``, constructed iff ``--all2all-backend`` is one of
        the mori values. That is knowable up front, needs no probe, and is
        evaluated identically on every rank -- which matters, because the
        initialize branch is collective.
        """
        import mori  # type: ignore[import-not-found]

        from vllm.config import get_current_vllm_config
        from vllm.distributed.parallel_state import get_ep_group

        if getattr(MegaMoERuntime, "_shmem_ready", False):
            return

        all2all_backend = get_current_vllm_config().parallel_config.all2all_backend
        owned_by_all2all = all2all_backend in _MORI_ALL2ALL_BACKENDS

        if owned_by_all2all:
            logger.info_once(
                "MegaMoEV2: reusing the mori symmetric heap brought up by "
                "all2all backend %r",
                all2all_backend,
            )
        else:
            # Mirrors MoriAll2AllManager.__init__. Collective: every EP rank
            # must arrive here, which holds because weight finalization runs on
            # every rank at model load.
            ep_group = get_ep_group()
            torch._C._distributed_c10d._register_process_group(
                "mori", ep_group.cpu_group
            )
            mori.shmem.shmem_torch_process_group_init("mori")
            logger.info_once(
                "MegaMoEV2: mori symmetric heap initialized (all2all backend "
                "%r does not own it)",
                all2all_backend,
            )

        MegaMoERuntime._shmem_ready = True

    def build(
        self,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Construct the shared instance, using ``w1..w2_scale`` as the
        initial binding. Subsequent layers rebind via :meth:`bind_weights`."""
        if self._moe is not None:
            return

        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

        self._ensure_shmem_initialized()

        shape = self.shape
        approx_gib = estimate_symmetric_bytes(shape) / 2**30
        logger.info(
            "MegaMoEV2: building shared instance world=%d model_dim=%d "
            "inter_dim=%d experts=%d topk=%d max_tok_per_rank=%d "
            "(~%.2f GiB symmetric per rank)",
            shape.world_size,
            shape.model_dim,
            shape.inter_dim,
            shape.experts,
            shape.topk,
            shape.max_tok_per_rank,
            approx_gib,
        )

        self._moe = MegaMoEV2(
            rank=self.rank,
            world_size=shape.world_size,
            model_dim=shape.model_dim,
            inter_dim=shape.inter_dim,
            experts=shape.experts,
            topk=shape.topk,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=shape.max_tok_per_rank,
            swiglu_limit=shape.swiglu_limit,
        )
        self._bound = None

    @contextlib.contextmanager
    def bind_weights(
        self,
        owner: object,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ):
        """Point the shared instance at one layer's weights for a call.

        The kernels read ``w2``/``w2_scale`` by ``data_ptr()`` at launch and
        take ``w1``/``w1_scale`` as tensor arguments, so rebinding is just four
        attribute assignments. ``owner`` is recorded so a nested bind -- which
        would silently corrupt the outer layer's launch -- is caught rather
        than producing wrong numerics.
        """
        moe = self._moe
        if moe is None:
            raise RuntimeError("MegaMoERuntime.build() must be called first")
        if self._bound is not None and self._bound is not owner:
            raise RuntimeError(
                "MegaMoEV2 weights are already bound to another layer; the "
                "shared instance does not support nested or concurrent use."
            )
        previous = self._bound
        self._bound = owner
        # Match what MegaMoEV2.__init__ does to the initial binding: stage-1
        # takes uint8 views of the packed fp4 weight and its e8m0 scales,
        # stage-2 reads w2/w2_scale by data_ptr() and only needs contiguity.
        moe._s1_w1 = w1.view(torch.uint8)
        moe._s1_w1_scale = w1_scale.view(torch.uint8)
        moe.w2 = w2
        moe.w2_scale = w2_scale
        try:
            yield moe
        finally:
            self._bound = previous
