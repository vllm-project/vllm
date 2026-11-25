# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-local EP reduction followed by shared-add, mHC, norm and quantization.

Uniform, nonempty DEP ranks publish BF16 partials to each request owner.
Reduction accumulates in FP32 in rank order, then rounds to BF16, like the
specialized TP kernel; it need not be bitwise equal to NCCL's reduction tree.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass.cutlass_dsl import dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    VEC_BF16,
    _asm,
    current_cu_stream,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_global_u64,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    to_cute,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.shared_boundary import (
    _FusedMHCDeviceKernel,
)

from . import MAX_ROWS_BY_WIDTH


@dsl_user_op
def _abort_wait(*, loc=None, ip=None):
    _asm(None, [], "trap;", "", side_effects=True, loc=loc, ip=ip)


def uniform_owner_layout(sizes, rows, width):
    return (
        width in (2, 4, 8)
        and 0 < rows <= MAX_ROWS_BY_WIDTH[width]
        and sizes is not None
        and len(sizes) == width
        and all(size == rows for size in sizes)
    )


@dataclass
class OwnerOutput:
    partials: torch.Tensor
    shared: torch.Tensor | None
    collective: "EPReduceScatterMHC"
    publish: Callable[[], None] | None = None


def owner_slice(global_row, rows, width):
    if not 1 <= rows <= MAX_ROWS_BY_WIDTH.get(width, 0):
        raise ValueError("uniform EP2/4 up to 128 rows or EP8 up to 384 rows required")
    if not 0 <= global_row < rows * width:
        raise ValueError("global row out of range")
    return divmod(global_row, rows)


class _OwnerPublish:
    def __init__(self, rank, width, capacity, hidden=5120):
        self.rank, self.width, self.capacity = rank, width, capacity
        self.hidden = hidden
        self.threads = 128
        self.fragments = hidden // VEC_BF16
        self.ctas = math.ceil(self.fragments / self.threads)

    @cute.jit
    def __call__(self, inp, state, peers: Int64, rows: Int32, stream: cuda.CUstream):
        self.kernel(inp, state, peers, rows).launch(
            grid=(rows * self.width * self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, state: cute.Tensor, peers: Int64, rows: Int32):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        token = block // self.ctas
        fragment = block % self.ctas * self.threads + thread
        cute.arch.griddepcontrol_wait()
        stage = load_volatile_u32(state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            src = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
            packed = sanitize_negative_zero_u32x4(load_global_u32x4(inp.iterator + src))
            owner = token // rows
            local_row = token % rows
            dst = (
                (Int64(stage) * self.width + self.rank) * self.capacity
                + Int64(local_row)
            ) * self.hidden + Int64(fragment) * VEC_BF16
            ptr = load_global_u64(peers + Int64(owner) * 8)
            store_global_u32x4(ptr + dst * 2, packed)
        if block == 0 and thread == 0:
            store_global_u32((state.iterator + ACTIVE_STAGE).toint(), stage)
        cute.arch.barrier()
        cute.arch.griddepcontrol_launch_dependents()


class _OwnerMHC(_FusedMHCDeviceKernel):
    def __init__(self, width, capacity, quantize, capture_aux, shared):
        super().__init__(5120, 4, quantize, capture_aux, False, shared)
        self.width, self.capacity = width, capacity

    @cute.jit
    def begin_collect(self, state):
        return load_volatile_u32(state.iterator + ACTIVE_STAGE)

    @cute.jit
    def load_contribution(self, mailbox, state, offset, stage):
        values = cute.make_rmem_tensor(cute.make_layout((self.width, 4)), Uint32)
        values.fill(Uint32(0))
        dirty = True
        attempts = Int32(0)
        while dirty and attempts < 1000000:
            dirty = False
            for source in cutlass.range_constexpr(self.width):
                index = (
                    Int64(stage) * self.width + source
                ) * self.capacity * self.hidden + offset
                packed = load_global_u32x4(mailbox.iterator + index, volatile=True)
                dirty = dirty | fragment_has_negative_zero(packed)
                for word in cutlass.range_constexpr(4):
                    values[source, word] = packed[word]
            attempts = attempts + 1
        # Never continue with incomplete contributions, even without a host
        # diagnostic read. This also bounds a lost peer's device-side wait.
        if dirty:
            _abort_wait()
        result = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        result.fill(Float32(0.0))
        for source in cutlass.range_constexpr(self.width):
            packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            for word in cutlass.range_constexpr(4):
                packed[word] = values[source, word]
            result.store(
                result.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
            )
            index = (
                Int64(stage) * self.width + source
            ) * self.capacity * self.hidden + offset
            store_lamport_sentinel_u32x4((mailbox.iterator + index).toint())
        return result.load().to(BFloat16).to(Float32)

    @cute.jit
    def end_collect(self, state, stage):
        token, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        cluster_rank = cute.arch.block_idx_in_cluster()
        if token == 0 and thread == 0 and cluster_rank == 0:
            store_global_u32(
                (state.iterator + NEXT_STAGE).toint(),
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )


class EPReduceScatterMHC:
    """All ranks call the same row sequence on the model stream.

    Every owner must consume at least one row, so each generation depends on
    all sources. This bounds rank skew before any of the three slots is reused.
    Empty/ragged batches are rejected, not handled by a rank-local fallback.
    """

    def __init__(self, group, device, capacity=128):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct before graph capture")
        self.width, self.rank = dist.get_world_size(group), dist.get_rank(group)
        owner_slice(0, capacity, self.width)
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise ValueError("DEP owner reduction requires SM103")
        self.capacity = capacity
        self.mailbox = symm.empty(
            (LAMPORT_GENERATIONS, self.width, capacity, 5120),
            dtype=torch.bfloat16,
            device=device,
        )
        self.handle = symm.rendezvous(self.mailbox, group)
        self.views = [
            self.handle.get_remote_tensor(r, self.mailbox.shape, self.mailbox.dtype)
            for r in range(self.width)
        ]
        self.peers = torch.tensor(
            [v.data_ptr() for v in self.views], dtype=torch.int64, device=device
        )
        self.mailbox.view(torch.int16).fill_(-32768)
        self.state = torch.zeros(2, dtype=torch.int32, device=device)
        self.plans = {}
        self.publishers = {}
        torch.accelerator.synchronize(device)
        dist.barrier(group)

    def prepare(
        self,
        inp,
        residual,
        post,
        comb,
        pre,
        norm,
        quant,
        aux=True,
        shared=None,
        publish_override=None,
    ):
        capturing = torch.cuda.is_current_stream_capturing()
        rows = residual.shape[0]
        owner_slice(0, rows, self.width)
        if rows > self.capacity or inp.shape != (self.width * rows, 5120):
            raise ValueError("owner layout does not match collective capacity")
        if residual.shape != (rows, 4, 5120):
            raise ValueError("HC4 H5120 required")
        if any(t.dtype != torch.bfloat16 for t in (inp, residual, norm.weight)):
            raise ValueError("BF16 owner partials and residuals required")
        if any(
            t.device != inp.device or not t.is_contiguous()
            for t in (inp, residual, post, comb, pre, norm.weight)
        ):
            raise ValueError("contiguous same-device inputs required")
        if shared is not None and (
            shared.shape != (rows, 5120)
            or shared.dtype != inp.dtype
            or not shared.is_contiguous()
        ):
            raise ValueError("shared expert must be local BF16 rows")
        out = torch.empty_like(residual)
        normalized = torch.empty_like(residual[:, 0])
        aux_out = torch.empty_like(normalized) if aux else None
        quant_data, scales = quant if quant is not None else (normalized, normalized)
        tensors = (
            self.mailbox,
            self.state,
            inp if shared is None else shared,
            residual,
            post,
            comb,
            pre,
            norm.weight,
            out,
            normalized,
            quant_data,
            scales,
            normalized if aux_out is None else aux_out,
        )
        align = (16, 4, 16, 16, 4, 4, 4, 16, 16, 16, 16, 16, 16)
        args = tuple(to_cute(t.reshape(-1), a) for t, a in zip(tensors, align))
        args += (
            Float32(norm.variance_epsilon),
            Int32(rows),
            Int32(0),
            current_cu_stream(),
        )
        key = (rows, quant is not None, aux, shared is not None)
        if key not in self.plans:
            if capturing:
                raise RuntimeError(f"owner consumer was not prepared: {key}")
            self.plans[key] = cute.compile(
                _OwnerMHC(
                    self.width,
                    self.capacity,
                    quant is not None,
                    aux,
                    shared is not None,
                ),
                *args,
            )
        publish_args = (
            to_cute(inp.reshape(-1), 16),
            to_cute(self.state, 4),
            Int64(self.peers.data_ptr()),
            Int32(rows),
            current_cu_stream(),
        )
        if rows not in self.publishers:
            if capturing:
                raise RuntimeError(f"owner publisher was not prepared: {rows}")
            self.publishers[rows] = cute.compile(
                _OwnerPublish(self.rank, self.width, self.capacity), *publish_args
            )
        publisher = self.publishers[rows]
        consumer = self.plans[key]

        def run():
            if publish_override is None:
                publisher(*publish_args[:-1], current_cu_stream())
            else:
                publish_override()
            consumer(*args[:-1], current_cu_stream())
            return out, normalized, aux_out

        return run
