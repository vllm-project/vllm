# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deadlock-free fused RS + short-conv + AG + residual + RMSNorm.

The public integration surface is ``LamportRSConv.rs_sconv_ag_add_norm``.

Liveness
--------
Large grids are deliberately split at the two communication dependencies:

  1. ``_publish_input_kernel`` only publishes rank partials.
  2. ``_reduce_insert_kernel`` waits, reduces, and inserts the local shard.
  3. ``_sconv_publish_kernel`` only computes and publishes the local result.
  4. ``_gather_norm_kernel`` waits, gathers, and normalizes.

Adapted from Nvidia side to work on AMD/ROCm. Uses HIP kernels and IPC buffers. 

Intially, vLLM Inkling model had support for lamport P2P on AMD using symmetric memory and
Triton kernels, but that was removed due to memory semantics not supported by that stack. 
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_TOKENS = 16384

@functools.cache
def compile_and_load_lamport_hip_ops():
    from torch.utils.cpp_extension import load

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    src = os.path.join(os.path.dirname(__file__), "csrc", "lamport_p2p.cu")
    return load(
        name="inkling_lamport_p2p",
        sources=[src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={arch.split(':')[0]}"],
        verbose=False,
    )

class LamportRSConv:
    def _initialize_lamport_buffers(self) -> None:
        if not self.buf_in.is_contiguous() or not self.buf_out.is_contiguous():
            raise RuntimeError("Lamport buffers must be contiguous")
        if self.buf_in.numel() % 2 or self.buf_out.numel() % 2:
            raise RuntimeError("Lamport buffers must container whole uint32 pairs")

        self.buf_in.view(torch.int16).fill_(-0x8000)
        self.buf_out.view(torch.int16).fill_(-0x8000)
        torch.accelerator.synchronize(self.device)
        self.tp.barrier()

        bad = torch.logical_or(
            self.buf_in.view(torch.int16).ne(-0x8000).any(),
            self.buf_out.view(torch.int16).ne(-0x8000).any(),
        ).to(dtype=torch.float32)
        bad = self.tp.all_reduce(bad)
        if int(bad.item()) != 0:
            raise RuntimeError("Lamport sentinel initialization failed on a TP rank")
        self.tp.barrier()
        
    def __init__(self, hidden_size: int, window_size: int, max_tokens: int = _MAX_TOKENS
    ) -> None:

        tp = get_tp_group()
        self.tp = tp
        self.group = tp.device_group
        self.world_size = tp.world_size
        self.rank = tp.rank_in_group
        self.device = torch.device(tp.device)
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"TP world size must be 2, 4, or 8, got {self.world_size}")
        if hidden_size % (2 * self.world_size) != 0:
            raise ValueError("hidden size must produce an even shard on every rank")
        if window_size != 4:
            raise ValueError(f"short-conv window size must be 4, got {window_size}")
        if max_tokens < 1 or max_tokens > _MAX_TOKENS:
            raise ValueError(f"max_tokens must be in [1, {_MAX_TOKENS}]")

        is_cross_node = not all(in_the_same_node_as(tp.cpu_group))
        assert is_cross_node is False, "Cannot use LamportRSConv on cross_node on ROCm"
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.shard_size = hidden_size // self.world_size

        # Three generations follow FlashInfer's Lamport layout.  A generation
        # is reused only after two intervening collective calls.
        self.num_buffers = 3
        self.input_generation_bytes = max_tokens * hidden_size * 2
        self.output_generation_bytes = max_tokens * hidden_size * 2
        
        in_shape = (self.num_buffers, max_tokens, self.world_size, self.shard_size)
        out_shape = (self.num_buffers, max_tokens, hidden_size)
        self._ipc_local: list[int] = []
        self._ipc_peers: list[int] = []
        self._ipc_tables: list[torch.Tensor] = []

        self._ext = compile_and_load_lamport_hip_ops()
        self.buf_in, self.input_peer_ptrs = self._ipc_buffer(in_shape)
        self.buf_out, self.output_peer_ptrs = self._ipc_buffer(out_shape)

        self._initialize_lamport_buffers()
        self.generation = 0

        #self.kernels = "hip"

        logger.info(
            "Lamport P2P fused RS/sconv/AG enabled (hip kernels, mem=uncached, tp=%d, max_tokens=%d)",
            self.world_size,
            max_tokens,
        )

    def _ipc_buffer(self, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
        """Allocate a hipIpc-shared bf16 buffer.

        Returns the local view and the device address of a uint64 peer-pointer
        table
        """
        nbytes = 2 # bf16
        for s in shape:
            nbytes *= s
        # 0 = uncached does hipMalloc with flag hipDeviceMallocUncached.
        # This is needed for lamport P2P to work correctly as it ensures that
        # payload is not cached in GPU's L2 and peer GPU which are polling end
        # up seeing the values. Otherwise, deadlocks can occur.
        ptr, handle = self._ext.ipc_alloc(nbytes, 0)
        self._ipc_local.append(ptr)
        handles: list[torch.Tensor | None] = [None] * self.world_size
        torch.distributed.all_gather_object(handles, handle, group=self.tp.cpu_group)
        ptrs = []

        for r, h in enumerate(handles):
            if r == self.rank:
                ptrs.append(ptr)
            else:
                ptrs.append(self._ext.ipc_open(h))
                self._ipc_peers.append(ptrs[-1])
        local = self._ext.as_tensor(ptr, list(shape), self.device.index)
        table = torch.tensor(ptrs, dtype=torch.uint64, device=self.device)
        self._ipc_tables.append(table)
        return local, table.data_ptr()

    def close(self) -> None:
        """Release hipIpc buffers"""
        torch.accelerator.synchronize(self.device)
        self.buf_in = self.buf_out = None
        for p in self._ipc_peers:
            self._ext.ipc_close(p)
        for p in self._ipc_local:
            self._ext.ipc_free(p)
        self._ipc_local, self._ipc_peers = [], []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def usable(self, num_tokens: int) -> bool:
        return 0 < num_tokens <= self.max_tokens

    def rs_sconv_ag_add_norm(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        conv_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float,
        cache: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        seq_idx: torch.Tensor,
        slot_mapping: torch.Tensor,
        off_s: int,
        ws: int,
        block_size: int,
        shared_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        tokens, hidden_size = residual.shape
        if not self.usable(tokens):
            raise ValueError(f"num_tokens must be in [1, {self.max_tokens}]")
        if hidden_size != self.hidden_size or residual.dtype != torch.bfloat16:
            raise ValueError("residual must be bf16 [T, 6144]")
        if(
            input_tensor.shape != residual.shape
            or input_tensor.dtype != torch.bfloat16
            or input_tensor.stride(1) != 1
        ):
            raise ValueError("input_tensor must be channel-contiguous bf16 [T, 6144]")
        if shared_tensor is not None and (
            shared_tensor.shape != input_tensor.shape
            or shared_tensor.dtype != input_tensor.dtype
            or shared_tensor.device != input_tensor.device
            or shared_tensor.stride(1) != 1
        ):
            raise ValueError("shared_tensor must match input_tensor")
        shard_size = hidden_size // self.world_size
        if conv_weight.shape != (shard_size, self.window_size):
            raise ValueError(
                f"conv_weight must have shape [{shard_size}, {self.window_size}]"
            )
        if (
            conv_weight.dtype != torch.bfloat16
            or conv_weight.stride(0) != self.window_size
        ):
            raise ValueError("conv_weight must be contiguous bf16")  
        if norm_weight is not None and (
            norm_weight.shape != (hidden_size, ) or norm_weight.dtype != torch.bfloat16
        ):            
            raise ValueError("norm_weight must be bf16 [6144] or None")
        if cache.dtype != torch.bfloat16 or cache.ndim != 4:
            raise ValueError("cache must be a 4-D bf16 tensor")
        if shard_size % ws != 0 or cache.shape[1] != shard_size // ws:
            raise ValueError("cache head layout is inconsistent with ws")
        if off_s < 0 or off_s + ws > cache.shape[3]:
            raise ValueError("cache channel offset is out of bounds")
        if block_size != cache.shape[2]:
            # slot // block_size indexes the block dimension, so a block_size
            # that disagrees with the cache's own page size walks off the
            # allocation and faults the GPU instead of raising here. Pass
            # owner.cache_block_size, not owner.block_size (the conv window).
            raise ValueError(
                f"block_size {block_size} does not match the cache page size "
                f"{cache.shape[2]}; slot indexing would leave the allocation"
            )

        # The NVIDIA publish kernel folds shared_tensor in via HAS_SHARED,
        # accumulating in fp32 before the bf16 store. The ROCm kernel has no
        # such path, so materialize the same sum up front -- identical numerics,
        # at the cost of one extra elementwise pass.
        if shared_tensor is not None:
            input_tensor = (
                input_tensor.to(torch.float32) + shared_tensor.to(torch.float32)
            ).to(torch.bfloat16)

        index = self.generation
        input_offset = index * self.input_generation_bytes // 4
        output_offset = index * self.output_generation_bytes // 4
        normed = torch.empty_like(residual) if norm_weight is not None else None
        residual_out = torch.empty_like(residual)
        phase_splits = 8 if tokens == 1 and shard_size % 8 == 0 else 1
        hip = self._ext
        hip.publish_input(
            input_tensor, self.input_peer_ptrs, input_offset, shard_size,
            phase_splits, self.rank, self.world_size,
        )
        hip.reduce_insert(
            self.input_peer_ptrs, input_offset, 
            #0, self.stall, 
            cache, slot_mapping, tokens, block_size, shard_size, phase_splits, ws, off_s,
            self.rank, self.world_size,
        )
        hip.sconv_publish(
            self.output_peer_ptrs, output_offset, residual, conv_weight, cache,
            positions, seq_idx, slot_mapping, block_table, block_size, shard_size,
            phase_splits, ws, off_s, self.rank, self.world_size
        )
        hip.gather_norm(
            self.output_peer_ptrs, output_offset, 
            #0, self.stall,
            norm_weight if norm_weight is not None else residual,
            normed if normed is not None else residual_out, residual_out, eps,
            self.rank, norm_weight is not None,
        )
        self.generation = (index + 1) % self.num_buffers

        return normed, residual_out



_STATE: LamportRSConv | None = None
_STATE_FAILED = False

def initialize_lamport_rs_conv(
        hidden_size: int, window_size: int, max_num_batched_tokens: int
) -> None:
    """Collectively initialize the TP-group state during model construction."""
    global _STATE, _STATE_FAILED
    if _STATE is not None:
        if _STATE.hidden_size != hidden_size or _STATE.window_size != window_size:
            raise RuntimeError("all Lamport users must share hidden and window sizes")
        return 
    if _STATE_FAILED or os.environ.get("LAMPORT_RS_SCONV", "1") == "0":
        return 
    try:
        max_tokens = min(_MAX_TOKENS, max_num_batched_tokens)
        _STATE = LamportRSConv(hidden_size, window_size, max_tokens=max_tokens)
    except Exception:
        _STATE_FAILED = True
        logger.exception("fused collective unavailable: use the RCCL fallback")

def get_lamport_rs_conv(hidden_size: int, window_size: int) -> LamportRSConv | None:
    if _STATE is not None and (
        _STATE.hidden_size != hidden_size or _STATE.window_size != window_size
    ):
        raise RuntimeError("all Lamport users must share hidden and window sizes")
    return _STATE