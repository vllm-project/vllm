# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Etha-style M-to-N weight transfer engine.

Using code from Etha's `comm/` module
(https://github.com/cmriat/Etha): declarative
`(DeviceMesh, Placement)` on both sides, compute one M-to-N chunk plan
per pair at init time, then move bytes straight from each source
rank's local shard into each target rank's local shard.

This module is the wiring layer. It owns:

- `EthaWeightTransferInitInfo` / `EthaWeightTransferUpdateInfo`
  — the dataclasses shipped between trainer and inference workers.
- `EthaWeightTransferEngine` — worker-side `WeightTransferEngine`
  subclass; runs `VllmEthaShardingStrategy` at init, then executes the
  recv chunks via NCCL per receive.
- `EthaTrainerWeightTransferEngine` — trainer-side engine; runs
  `TrainerEthaShardingStrategy` at init, then executes the send chunks
  via NCCL per send. Mirrors the eventual `TrainerWeightTransferEngine`
  ABC shape proposed in `trainer_send_refactor.md`, scoped to Etha for now.
- `chunk_comm` — NCCL execution for a `list[Chunk]`. Lives here
  because buffer management (contiguity, dtype intermediates, group
  batching) is engine-specific; a future RDMA/NIXL engine would own
  its own equivalent.

Planning logic lives in `etha_sharding` (transport-agnostic); the
`Chunk` dataclass lives in `etha_chunk`.

Two MVP simplifications carried over:

1. We piggyback on vLLM's `StatelessProcessGroup` + `PyNcclCommunicator`
   rendezvous (same shape `NCCLWeightTransferEngine` uses) rather than
   spawning a separate "agent" process group.
2. Broadcasts (one source slice fanning out to multiple targets) are
   degraded to fan-out P2P sends. NCCL broadcast over an arbitrary
   subgroup would require a second `PyNcclCommunicator` per subset,
   which is more rendezvous than it's worth for MVP. Bytes-heavy
   pairs (`experts_*`) are pure P2P anyway.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

from etha_chunk import Chunk
from etha_sharding import (
    TrainerEthaShardingStrategy,
    VllmEthaShardingStrategy,
)

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = init_logger(__name__)


# ============================================================================
# Init / Update info dataclasses
# ============================================================================


@dataclass
class EthaWeightTransferInitInfo(WeightTransferInitInfo):
    """Rendezvous info for the Etha backend.

    Mirrors `NCCLWeightTransferInitInfo`: trainer is the lower-rank
    half of the joint process group, vLLM workers are the upper half.

    Trainer mesh dims describe the trainer-side `att_mesh` (2D:
    dp_replicate x dp_shard) and `moe_mesh` (3D: dp_replicate x dp_shard
    x ep). vLLM mesh dims describe the inference side's (dp, tp) and
    expert-parallel size. Both sides read the same struct — the worker
    uses the trainer fields to compute peer placements, the trainer
    uses the vllm fields for the same reason.
    """

    master_address: str
    master_port: int
    rank_offset: int  # added to (dp_rank * world_size_per_dp + local_rank)
    world_size: int  # M trainer + N vLLM
    # Trainer-side mesh dims.
    trainer_attn_dp_replicate: int = 2
    trainer_attn_dp_shard: int = 2
    trainer_moe_dp_replicate: int = 1
    trainer_moe_dp_shard: int = 2
    trainer_ep_size: int = 2
    # vLLM-side mesh dims. Redundant on the worker side (worker reads
    # its own ParallelConfig) but required on the trainer side, where
    # the trainer has no access to ParallelConfig.
    vllm_dp_size: int = 2
    vllm_tp_size: int = 2
    vllm_ep_size: int = 4


@dataclass
class EthaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info. Intentionally empty — everything (pair
    placements, chunk plan) is baked in at init_transfer_engine."""

    version: int = 0


# ============================================================================
# NCCL transport
# ============================================================================


def chunk_comm(
    chunks: list[Chunk],
    pynccl: PyNcclCommunicator,
    label: str = "chunk_comm",
) -> None:
    """Execute a chunk list via NCCL.

    Sends and recvs are batched into one NCCL group; self-copies happen
    outside the group (they're host-side `Tensor.copy_`). After the
    group closes we synchronize the current stream and copy any
    dtype-converted recv intermediates back into their destination
    slices.

    Logs prep / NCCL / post-process timings via `time.monotonic` and
    reports the total bytes moved over the wire.
    """
    intermediates: list[tuple[Chunk, torch.Tensor]] = []
    ops: list[tuple[Chunk, torch.Tensor]] = []
    wire_bytes = 0

    t0 = time.monotonic()
    for chunk in chunks:
        if chunk.is_self_copy:
            assert chunk.src_slice_tuples is not None
            src = chunk.tensor[chunk.src_slice_tuples]
            dst = chunk.tensor[chunk.slice_tuples]
            if src.dtype != dst.dtype:
                dst.copy_(src.to(dst.dtype))
            else:
                dst.copy_(src)
            continue

        slot = chunk.tensor[chunk.slice_tuples]
        if chunk.is_source:
            buf = slot if slot.is_contiguous() else slot.contiguous()
            if buf.dtype != chunk.transfer_dtype:
                buf = buf.to(chunk.transfer_dtype)
        else:
            # Recv buffer: must be contiguous and in transfer_dtype. If
            # the slot already matches, recv straight into it; otherwise
            # recv into an intermediate and copy back below.
            if slot.is_contiguous() and slot.dtype == chunk.transfer_dtype:
                buf = slot
            else:
                buf = torch.empty(
                    slot.shape, dtype=chunk.transfer_dtype, device=slot.device
                )
                intermediates.append((chunk, buf))
        wire_bytes += buf.numel() * buf.element_size()
        ops.append((chunk, buf))
    t_prep = time.monotonic() - t0

    t0 = time.monotonic()
    pynccl.group_start()
    try:
        for chunk, buf in ops:
            if chunk.is_source:
                pynccl.send(buf, dst=chunk.dst_rank)
            else:
                pynccl.recv(buf, src=chunk.src_rank)
    finally:
        pynccl.group_end()

    # Stream-sync before touching recv buffers from the host side.
    torch.cuda.current_stream().synchronize()
    t_nccl = time.monotonic() - t0

    t0 = time.monotonic()
    for chunk, buf in intermediates:
        dst = chunk.tensor[chunk.slice_tuples]
        dst.copy_(buf.to(dst.dtype) if buf.dtype != dst.dtype else buf)
    t_post = time.monotonic() - t0

    total = t_prep + t_nccl + t_post
    gib = wire_bytes / (1024**3)
    bw = (gib / t_nccl) if t_nccl > 0 else float("inf")
    logger.info(
        "[%s] %d ops, %.3f GiB on the wire | "
        "prep=%.3fs nccl+sync=%.3fs post=%.3fs | "
        "total=%.3fs (%.2f GiB/s on-wire)",
        label,
        len(ops),
        gib,
        t_prep,
        t_nccl,
        t_post,
        total,
        bw,
    )


# ============================================================================
# Worker-side engine
# ============================================================================


class EthaWeightTransferEngine(
    WeightTransferEngine[EthaWeightTransferInitInfo, EthaWeightTransferUpdateInfo]
):
    """Weight transfer engine that uses M-to-N resharding + NCCL P2P.

    Designed to be used with `start_weight_update(is_checkpoint_format=False)`:
    chunks write directly into kernel-format parameter storage, so the
    `load_weights` callable passed to `receive_weights` is ignored. This
    matches what the Etha example does — call `process_weights_after_loading`
    once at vLLM startup (with `--load-format dummy`), then keep writing
    into the kernel-format tensors forever.
    """

    init_info_cls = EthaWeightTransferInitInfo
    update_info_cls = EthaWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self.stateless_pg: StatelessProcessGroup | None = None
        self.pynccl: PyNcclCommunicator | None = None
        self._recv_chunks: list[Chunk] = []
        self.rank: int = -1

    def init_transfer_engine(
        self,
        init_info: EthaWeightTransferInitInfo,
        model: torch.nn.Module | None = None,
    ) -> None:
        # We need the model to walk the module graph for per-parameter
        # placement inference. The gpu_worker passes this in for us.
        if model is None:
            raise RuntimeError(
                "EthaWeightTransferEngine.init_transfer_engine requires the "
                "loaded model — gpu_worker passes this for you. If you're "
                "calling the engine directly, pass model=<your inference model>."
            )

        # 1. Rendezvous (same shape as NCCLWeightTransferEngine).
        worker_rank = (
            self.parallel_config.data_parallel_index
            * self.parallel_config.world_size
            + self.parallel_config.rank
        )
        rank = worker_rank + init_info.rank_offset
        device = torch.accelerator.current_device_index()
        self.stateless_pg = StatelessProcessGroup.create(
            host=init_info.master_address,
            port=init_info.master_port,
            rank=rank,
            world_size=init_info.world_size,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.pynccl = PyNcclCommunicator(self.stateless_pg, device=device)
        self.rank = rank
        logger.info(
            "Etha engine rank=%d world_size=%d initialized",
            rank,
            init_info.world_size,
        )

        # 2. Plan via the worker-side sharding strategy.
        trainer_world_size = (
            init_info.trainer_attn_dp_replicate * init_info.trainer_attn_dp_shard
        )
        strategy = VllmEthaShardingStrategy(
            model=model,
            parallel_config=self.parallel_config,
            trainer_world_size=trainer_world_size,
            trainer_attn_dp_replicate=init_info.trainer_attn_dp_replicate,
            trainer_attn_dp_shard=init_info.trainer_attn_dp_shard,
            trainer_moe_dp_replicate=init_info.trainer_moe_dp_replicate,
            trainer_moe_dp_shard=init_info.trainer_moe_dp_shard,
            trainer_ep_size=init_info.trainer_ep_size,
        )
        self._recv_chunks = strategy.plan_and_specialize(
            self.stateless_pg.store, rank=rank, world_size=init_info.world_size
        )

    def receive_weights(
        self,
        update_info: EthaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self.pynccl is None:
            raise RuntimeError("Etha engine not initialized")
        t0 = time.monotonic()
        torch.cuda.synchronize()
        chunk_comm(
            self._recv_chunks,
            self.pynccl,
            label=f"recv rank={self.rank} v={update_info.version}",
        )
        torch.cuda.synchronize()
        logger.info(
            "Etha receive_weights rank=%d v=%d wall=%.3fs",
            self.rank,
            update_info.version,
            time.monotonic() - t0,
        )
        # load_weights is deliberately unused — Etha wrote in place above.
        # IMPORTANT: callers must use start_weight_update(is_checkpoint_format=False);
        # layerwise reload would clobber the writes we just did.
        del load_weights

    def shutdown(self) -> None:
        self.pynccl = None
        self.stateless_pg = None
        self._recv_chunks = []

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "Etha is symmetric — use `EthaTrainerWeightTransferEngine` from "
            "the trainer process. See examples/rl/rlhf_etha.py."
        )


# ============================================================================
# Trainer-side engine
# ============================================================================


class EthaTrainerWeightTransferEngine:
    """Trainer-side weight transfer engine.

    Per round, callers invoke `send_weights()` with no arguments —
    chunks reference tensor handles by view, so the trainer-side
    mutate-in-place pattern works without re-registration.

    `wire_dtype` must match the dtype the vLLM-side parameters hold on
    receive (typically `torch.bfloat16`). NCCL pairs ops by byte count,
    so both sides must agree on the wire dtype or the NCCL group will
    deadlock waiting for matching bytes. If the trainer's tensors are
    in a wider dtype (e.g. fp32 master weights), `chunk_comm` downcasts
    to `wire_dtype` before issuing the send.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        pynccl: PyNcclCommunicator,
        stateless_pg: StatelessProcessGroup,
        send_chunks: list[Chunk],
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.pynccl = pynccl
        self.stateless_pg = stateless_pg
        self.send_chunks = send_chunks

    @classmethod
    def trainer_init(
        cls,
        init_info: EthaWeightTransferInitInfo,
        *,
        rank: int,
        device_index: int,
        state_dict: dict[str, torch.Tensor],
        wire_dtype: torch.dtype = torch.bfloat16,
    ) -> Self:
        """Rendezvous with workers, plan send chunks, return a ready engine.

        `state_dict` maps trainer-side parameter names → local shard
        tensors (typically `DTensor.to_local()` results). The planner
        runs identically to the vLLM side; both arrive at the same
        M2M map.
        """
        pg = StatelessProcessGroup.create(
            host=init_info.master_address,
            port=init_info.master_port,
            rank=rank,
            world_size=init_info.world_size,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        pynccl = PyNcclCommunicator(pg, device=device_index)

        trainer_world_size = (
            init_info.trainer_attn_dp_replicate * init_info.trainer_attn_dp_shard
        )
        strategy = TrainerEthaShardingStrategy(
            state_dict=state_dict,
            trainer_attn_dp_replicate=init_info.trainer_attn_dp_replicate,
            trainer_attn_dp_shard=init_info.trainer_attn_dp_shard,
            trainer_moe_dp_replicate=init_info.trainer_moe_dp_replicate,
            trainer_moe_dp_shard=init_info.trainer_moe_dp_shard,
            trainer_ep_size=init_info.trainer_ep_size,
            trainer_world_size=trainer_world_size,
            vllm_dp_size=init_info.vllm_dp_size,
            vllm_tp_size=init_info.vllm_tp_size,
            vllm_ep_size=init_info.vllm_ep_size,
            wire_dtype=wire_dtype,
        )
        send_chunks = strategy.plan_and_specialize(
            pg.store, rank=rank, world_size=init_info.world_size
        )
        return cls(
            rank=rank,
            world_size=init_info.world_size,
            pynccl=pynccl,
            stateless_pg=pg,
            send_chunks=send_chunks,
        )

    def send_weights(self) -> None:
        """Trainer-side per-round send. Chunks are pre-baked and reference
        tensor views — callers mutate the underlying tensors in place
        between rounds, no re-registration needed."""
        t0 = time.monotonic()
        torch.cuda.synchronize()
        chunk_comm(
            self.send_chunks, self.pynccl, label=f"send rank={self.rank}"
        )
        torch.cuda.synchronize()
        logger.info(
            "Etha trainer send_weights rank=%d wall=%.3fs",
            self.rank,
            time.monotonic() - t0,
        )

    def shutdown(self) -> None:
        self.send_chunks = []


# ============================================================================
# Worker-side registration hook
# ============================================================================
#
# vLLM worker processes (gpu_worker) each import the factory module and only
# see backends registered in their own interpreter. The factory ships nccl +
# ipc by default; "etha" lives in this example tree, so we register it here
# and rely on a worker_extension_cls=`etha_engine.EthaWorkerExtension` lookup
# in the worker to import this module (the import side-effect performs the
# registration). The example's PYTHONPATH wiring puts this directory on
# sys.path for both the driver and the Ray actors.

from vllm.distributed.weight_transfer import WeightTransferEngineFactory  # noqa: E402

if "etha" not in WeightTransferEngineFactory._registry:
    WeightTransferEngineFactory.register_engine("etha", EthaWeightTransferEngine)


class EthaWorkerExtension:
    """Empty marker class — importing it registers the etha backend.

    Pass as `worker_extension_cls="etha_engine.EthaWorkerExtension"` when
    constructing AsyncEngineArgs so each vLLM worker imports this module
    (and thereby runs the registration above) before it creates its weight
    transfer engine.
    """
