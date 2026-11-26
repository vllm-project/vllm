# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch: each warp walks a persistent device table of (address, bytes),
spreading 16-KiB chunks over the grid in table order. The owner retains the weights
and table through graph replay; activations and mutable caches may not be ranges
before their producer completes."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from .host import fake
from .primitives import launch, prefetch_bulk, thread


class L2Prefetch:
    def __init__(self, grid, carveout):
        self.grid, self.carveout = grid, carveout

    @cute.jit
    def __call__(self, *args):
        options = dict(pdl=False, preferred_smem_carveout=self.carveout)
        launch(self.kernel, args, self.grid, 32, **options)

    @cute.kernel
    def kernel(self, args):
        (ranges,) = args
        cute.arch.griddepcontrol_launch_dependents()
        tid, _, _, block = thread()
        worker = cutlass.Int64(block * 32 + tid)
        workers = cutlass.Int64(self.grid * 32)
        phase, row = cutlass.Int64(0), cutlass.Int32(0)
        while row < ranges.shape[0]:
            base, nbytes = ranges[row, 0], ranges[row, 1]
            if (base != 0) & (nbytes > 0):
                chunks = (nbytes + 16384 - 1) // 16384
                index = worker - phase % workers
                if index < 0:
                    index += workers
                while index < chunks:
                    offset = index * 16384
                    size = nbytes - offset
                    if size > 16384:
                        size = cutlass.Int64(16384)
                    size = size & cutlass.Int64(-16)
                    if size > 0:
                        address = (base + offset) & cutlass.Int64(-16)
                        prefetch_bulk(address, size.to(cutlass.Int32))
                    index += workers
                phase += chunks
            row += 1
        # Early exit must not let a later grid bypass its producer.
        cute.arch.griddepcontrol_wait()


@functools.cache
def compile_kernel(count, grid, carveout, device_index):
    ranges = fake((count, 2), None, torch.int64)
    with torch.accelerator.device_index(device_index):
        return cute.compile(L2Prefetch(grid, carveout), ranges, make_fake_stream())
