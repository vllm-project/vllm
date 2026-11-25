# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct EP dispatch and deferred finalize for uniform DEP decode graphs."""

import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from cutlass import Int32, Int64

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream, to_cute
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.all_reduce_mhc import (
    _QuadFinalizePublishDeviceKernel,
)
from vllm.triton_utils import tl, triton

from . import MAX_ROWS_BY_WIDTH
from .front import FrontOutput


@triton.jit
def _peer_gather_planes(
    peers,
    quant,
    scale,
    indices,
    weights,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    source = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    quant_words: tl.constexpr = ROWS * 1280
    scale_words: tl.constexpr = ROWS * 40
    index_words: tl.constexpr = ROWS * 6
    total: tl.constexpr = quant_words + scale_words + 2 * index_words
    address = tl.load(peers + source).to(tl.pointer_type(tl.uint32))
    value = tl.load(address + offset, offset < total, other=0, cache_modifier=".cg")
    tl.store(quant + source * quant_words + offset, value, offset < quant_words)
    tl.store(
        scale + source * scale_words + offset - quant_words,
        value,
        (offset >= quant_words) & (offset < quant_words + scale_words),
    )
    tl.store(
        indices + source * index_words + offset - quant_words - scale_words,
        value,
        (offset >= quant_words + scale_words)
        & (offset < quant_words + scale_words + index_words),
    )
    tl.store(
        weights
        + source * index_words
        + offset
        - quant_words
        - scale_words
        - index_words,
        value,
        (offset >= quant_words + scale_words + index_words) & (offset < total),
    )


class PeerFrontDispatch:
    """Front writes into symmetric memory; peers pull into contiguous GEMM input.

    The next owner reduction is mandatory: receiving every source's contribution
    proves that every peer finished pulling before this send buffer is reused.
    Allocation/rendezvous happens once; per-shape preparation is rank-local.
    """

    def __init__(self, device, group, rows, storage=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare dispatch before capture")
        self.width, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if not 2 <= rows <= MAX_ROWS_BY_WIDTH.get(self.width, 0) or rows % 2:
            raise ValueError(
                "uniform even rows: DEP2/4 up to 128, DEP8 up to 384 required"
            )
        self.rows = rows
        self.global_output = FrontOutput(
            *(
                torch.empty((rows * self.width, columns), device=device, dtype=dtype)
                for columns, dtype in (
                    (5120, torch.float8_e4m3fn),
                    (160, torch.uint8),
                    (6, torch.int32),
                    (6, torch.float32),
                )
            )
        )
        self.payload_bytes = rows * 5328
        if storage is None:
            self.send = symm.empty(
                (self.payload_bytes,), dtype=torch.uint8, device=device
            )
            self.handle = symm.rendezvous(self.send, group)
            self.views = [
                self.handle.get_remote_tensor(rank, self.send.shape, self.send.dtype)
                for rank in range(self.width)
            ]
            self.peers = torch.tensor(
                [value.data_ptr() for value in self.views],
                device=device,
                dtype=torch.int64,
            )
        else:
            if (
                storage.rows < rows
                or storage.width != self.width
                or storage.rank != self.rank
                or storage.send.device != device
            ):
                raise ValueError("incompatible shared dispatch storage")
            self.storage = storage
            self.send = storage.send[: self.payload_bytes]
            self.handle, self.views, self.peers = (
                storage.handle,
                storage.views,
                storage.peers,
            )
        outputs, start = [], 0
        for columns, dtype in (
            (5120, torch.float8_e4m3fn),
            (160, torch.uint8),
            (6, torch.int32),
            (6, torch.float32),
        ):
            size = rows * columns * torch.empty((), dtype=dtype).element_size()
            outputs.append(
                self.send.narrow(0, start, size).view(dtype).view(rows, columns)
            )
            start += size
        self.local_output = FrontOutput(*outputs)
        # Compile before entering any cross-rank device wait. Zeroed setup data
        # is not an inference result; the front overwrites every word each call.
        if storage is None:
            self.send.zero_()
            torch.accelerator.synchronize(device)
            dist.barrier(group)
            self._pull()
            torch.accelerator.synchronize(device)
            dist.barrier(group)
        else:
            # Rank-local shape preparation must not enter a collective or read
            # peer data. Only the common setup performs a rendezvous.
            _peer_gather_planes.warmup(
                self.peers,
                *(t.view(torch.uint32) for t in self.global_output),
                self.rows,
                256,
                num_warps=4,
                grid=(triton.cdiv(self.payload_bytes // 4, 256), self.width),
            )

    def _pull(self):
        _peer_gather_planes[(triton.cdiv(self.payload_bytes // 4, 256), self.width)](
            self.peers,
            *(t.view(torch.uint32) for t in self.global_output),
            self.rows,
            256,
            num_warps=4,
        )

    def gather_before_owner(self):
        self.handle.barrier(channel=0, timeout_ms=10000)
        self._pull()
        return self.global_output


class FinalizeOwnerPublish:
    """Reuse TP's finalize math, publishing directly to contiguous DEP owners.

    Shared experts stay owner-local and are added by the consumer exactly once.
    GEMM2 buffers must remain alive until this plan completes. Each invocation
    must be followed by the same mailbox's owner consumer on every rank.
    """

    def __init__(self, collective, routed, rows):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile before graph capture")
        total = rows * collective.width
        if not 1 <= rows <= collective.capacity:
            raise ValueError("invalid local owner row count")
        if (
            routed.gemm2_permuted.ndim != 2
            or routed.gemm2_permuted.shape[1] != 5120
            or routed.gemm2_permuted.dtype != torch.bfloat16
            or routed.expert_weights.shape != (total, 6)
            or routed.expert_weights.dtype != torch.float32
            or routed.expanded_idx_to_permuted_idx.shape != (total, 6)
            or routed.expanded_idx_to_permuted_idx.dtype != torch.int32
        ):
            raise ValueError("BF16 H5120 and FP32/int32 top6 metadata required")
        values = (
            routed.gemm2_permuted,
            routed.expert_weights,
            routed.expanded_idx_to_permuted_idx,
        )
        if any(
            not t.is_contiguous() or t.device != collective.mailbox.device
            for t in values
        ):
            raise ValueError("contiguous same-device routed tensors required")
        self.routed = routed
        self.collective = collective
        # The kernel indexes GEMM2 through the permutation map, never its padded
        # row count. A one-row pointer view avoids specializing on tactic padding.
        args = (
            to_cute(values[0].reshape(-1)[:5120], 16),
            *(to_cute(t.reshape(-1), 4) for t in values[1:]),
        )
        # Shared output is an unused argument when include_shared_expert=False.
        self.args = (
            *args,
            to_cute(routed.gemm2_permuted.reshape(-1)[:5120], 16),
            to_cute(collective.state, 4),
            Int64(collective.peers.data_ptr()),
            Int32(total),
        )
        self.compiled = cute.compile(
            _QuadFinalizePublishDeviceKernel(
                hidden=5120,
                top_k=6,
                tp=collective.width,
                rank=collective.rank,
                capacity_m=total,
                threads=128,
                routed_scaling_factor=1.0,
                include_shared_expert=False,
                load_shared_expert_before_pdl=False,
                enable_pdl=True,
                prefetch_group=1,
                fp32_weights=True,
                owner_capacity=collective.capacity,
                elements_per_thread=8,
                contiguous_owner_rows=rows,
            ),
            *self.args,
            current_cu_stream(),
        )

    def __call__(self, routed=None):
        args = self.args
        if routed is not None:
            original = (
                self.routed.gemm2_permuted,
                self.routed.expert_weights,
                self.routed.expanded_idx_to_permuted_idx,
            )
            fresh = (
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
            )
            if (
                fresh[0].ndim != 2
                or fresh[0].shape[0] < 1
                or fresh[0].shape[1] != 5120
                or any(
                    a.dtype != b.dtype or a.device != b.device or not b.is_contiguous()
                    for a, b in zip(original, fresh)
                )
                or any(a.shape != b.shape for a, b in zip(original[1:], fresh[1:]))
            ):
                raise ValueError(
                    "deferred output layout changed; prepare outside capture"
                )
            pointers = (
                to_cute(fresh[0].reshape(-1)[:5120], 16),
                *(to_cute(t.reshape(-1), 4) for t in fresh[1:]),
            )
            args = (*pointers, to_cute(fresh[0].reshape(-1)[:5120], 16), *self.args[4:])
        self.compiled(*args, current_cu_stream())
