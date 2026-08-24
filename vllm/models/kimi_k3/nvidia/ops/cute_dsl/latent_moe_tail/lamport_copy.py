# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lamport mailbox-to-local copy and reset."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from .primitives import (
    NUM_LAMPORT_BUFFERS,
    VEC_BF16,
    fragment_is_dirty,
    load_global_u32x4,
    store_global_u32x4,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)


class LamportCopy:
    """Consume the local physical copy of an NVLS-multicast mailbox."""

    def __init__(
        self,
        hidden_dim: int,
        ctas: int,
        threads: int,
        rotate_shared_generation: bool,
    ):
        self.hidden_dim = hidden_dim
        self.ctas = ctas
        self.threads = threads
        self.rotate_shared_generation = rotate_shared_generation

    @cute.jit
    def __call__(
        self,
        symmetric_mailbox: cute.Tensor,
        local_output: cute.Tensor,
        shared_flags: cute.Tensor,
        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(symmetric_mailbox, local_output, shared_flags, m).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        symmetric_mailbox: cute.Tensor,
        local_output: cute.Tensor,
        shared_flags: cute.Tensor,
        m: cutlass.Int32,
    ):
        # The CTA may be scheduled early, but mailbox inspection must not pass
        # the producer GEMM's programmatic completion point.
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.rotate_shared_generation):  # noqa: SIM102
            if block == 0 and tidx == 0:
                current_index = cute.arch.load(
                    (shared_flags.iterator + 0).llvm_ptr,
                    cutlass.Uint32,
                )
                next_index = (current_index + cutlass.Uint32(1)) % cutlass.Uint32(
                    NUM_LAMPORT_BUFFERS
                )
                cute.arch.store(
                    (shared_flags.iterator + 0).llvm_ptr,
                    next_index,
                )
        thread = cutlass.Int64(block * self.threads + tidx)
        stride = cutlass.Int64(self.ctas * self.threads)
        fragments = cutlass.Int64(m) * cutlass.Int64(self.hidden_dim // VEC_BF16)

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            packed = load_global_u32x4(source, volatile=True)
            while fragment_is_dirty(packed):
                packed = load_global_u32x4(source, volatile=True)

            destination = cutlass.Int64((local_output.iterator + element).toint())
            store_global_u32x4(destination, packed, volatile=False)
            fragment = fragment + stride

        # The returned ordinary tensor is complete. A same-stream successor
        # may overlap the mailbox cleanup below.
        cute.arch.griddepcontrol_launch_dependents()

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            store_lamport_sentinel_128(source)
            fragment = fragment + stride


@functools.cache
def compile_kernel(
    hidden_dim: int,
    max_m: int,
    ctas: int,
    threads: int,
    device_index: int,
    rotate_shared_generation: bool,
):
    if hidden_dim <= 0 or hidden_dim % VEC_BF16:
        raise ValueError("hidden_dim must be a positive multiple of 8")
    if max_m <= 0:
        raise ValueError("max_m must be positive")
    if ctas <= 0:
        raise ValueError("Lamport copy ctas must be positive")
    if not 1 <= threads <= 1024:
        raise ValueError("Lamport copy threads must be in [1, 1024]")
    with torch.accelerator.device_index(device_index):
        mailbox = make_fake_compact_tensor(
            cutlass.BFloat16, (max_m * hidden_dim,), assumed_align=16
        )
        output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int32(divisibility=VEC_BF16),),
            assumed_align=16,
        )
        shared_flags = make_fake_compact_tensor(cutlass.Int32, (12,), assumed_align=16)
        return cute.compile(
            LamportCopy(
                hidden_dim,
                ctas,
                threads,
                rotate_shared_generation,
            ),
            mailbox,
            output,
            shared_flags,
            cutlass.Int32(max_m),
            make_fake_stream(),
        )


def launch(
    symmetric_mailbox: torch.Tensor,
    local_output: torch.Tensor,
    *,
    m: int,
    hidden_dim: int,
    max_m: int,
    ctas: int,
    threads: int,
    shared_flags: torch.Tensor,
    rotate_shared_generation: bool,
) -> None:
    if not 1 <= m <= max_m:
        raise ValueError(f"runtime M={m} must be in [1, {max_m}]")
    if (
        symmetric_mailbox.ndim != 3
        or symmetric_mailbox.dtype != torch.bfloat16
        or not symmetric_mailbox.is_cuda
        or not symmetric_mailbox.is_contiguous()
        or symmetric_mailbox.shape[0] != 1
        or symmetric_mailbox.shape[2] != hidden_dim
    ):
        raise ValueError(
            f"symmetric_mailbox must be contiguous CUDA BF16 [1,M,{hidden_dim}]"
        )
    if symmetric_mailbox.shape[1] != max_m:
        raise ValueError("symmetric mailbox must retain its full max_m capacity")
    if (
        local_output.shape != (1, m, hidden_dim)
        or local_output.dtype != torch.bfloat16
        or local_output.device != symmetric_mailbox.device
        or not local_output.is_contiguous()
    ):
        raise ValueError("local_output must be contiguous CUDA BF16 [1,M,H]")

    device_index = symmetric_mailbox.device.index
    stream = cuda.CUstream(
        torch.cuda.current_stream(symmetric_mailbox.device).cuda_stream
    )
    compile_kernel(
        hidden_dim,
        max_m,
        ctas,
        threads,
        device_index,
        rotate_shared_generation,
    )(
        to_cute(symmetric_mailbox.flatten(), 16),
        to_cute_dynamic_m(
            local_output.flatten(),
            mode=0,
            assumed_align=16,
        ),
        to_cute(shared_flags, 16),
        cutlass.Int32(m),
        stream,
    )


class LamportCopyKernel:
    """Copy a borrowed symmetric mailbox into a fresh local tensor."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_m: int,
        ctas: int,
        threads: int,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.max_m = max_m
        self.ctas = ctas
        self.threads = threads
        device_index = torch.accelerator.current_device_index()
        self._dummy_shared_flags = torch.zeros(
            12,
            dtype=torch.int32,
            device=torch.device("cuda", device_index),
        )
        for rotate_shared_generation in (False, True):
            compile_kernel(
                hidden_dim,
                max_m,
                ctas,
                threads,
                device_index,
                rotate_shared_generation,
            )

    def __call__(
        self,
        symmetric_mailbox: torch.Tensor,
        *,
        m: int,
        shared_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not symmetric_mailbox.is_cuda:
            raise ValueError("symmetric_mailbox must be a CUDA tensor")
        device = symmetric_mailbox.device
        with torch.accelerator.device_index(device.index):
            output = torch.empty(
                (1, m, self.hidden_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            rotate_shared_generation = shared_flags is not None
            if shared_flags is None:
                shared_flags = self._dummy_shared_flags
            elif (
                shared_flags.shape != (12,)
                or shared_flags.dtype != torch.int32
                or shared_flags.device != device
                or not shared_flags.is_contiguous()
            ):
                raise ValueError("shared_flags must be contiguous CUDA int32 [12]")
            launch(
                symmetric_mailbox,
                output,
                m=m,
                hidden_dim=self.hidden_dim,
                max_m=self.max_m,
                ctas=self.ctas,
                threads=self.threads,
                shared_flags=shared_flags,
                rotate_shared_generation=rotate_shared_generation,
            )
        return output
