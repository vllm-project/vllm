# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local CuTe vocabulary gather plus the ordinary framework TP all-reduce."""

from functools import cache

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.distributed import get_tp_group

from .host import architecture, fake
from .primitives import launch


class EmbeddingGather:
    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[0].shape[0], 1), 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        ids, weight, out, start, vocab = args[0:5]
        lane, _, _ = cute.arch.thread_idx()
        tile, token, _ = cute.arch.block_idx()
        token_id = c.Int64(ids[token])
        row, column = token_id - start, tile * 1024 + lane * 8
        values = cute.make_rmem_tensor((8,), c.BFloat16)
        values.fill(c.BFloat16(0))
        if (token_id >= 0) & (token_id < vocab) & (row >= 0) & (row < weight.shape[0]):
            source = weight.iterator + row * 7168 + column
            cute.autovec_copy(cute.make_tensor(source, cute.make_layout((8,))), values)
        destination = out.iterator + token * 7168 + column
        cute.autovec_copy(values, cute.make_tensor(destination, cute.make_layout((8,))))


@cache
def compile_gather(dtype, shard, arch):
    batch = cute.sym_int32()
    args: tuple = (fake((batch,), None, dtype, dtype.itemsize), fake((shard, 7168)))
    args += (fake((batch, 7168)), c.Int64(0), c.Int64(0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(EmbeddingGather(), *args, stream, options=options)


class PreparedEmbedding:
    def __init__(self, embedding, tp, rank):
        weight = embedding.weight
        # The gather moves 7 x 1024 BF16 columns of contiguous rows.
        if weight.dim() != 2 or weight.shape[1] != 7168 or not weight.is_contiguous():
            raise ValueError(
                f"embedding shard must be a contiguous [V, 7168] tensor, got "
                f"{tuple(weight.shape)}"
            )
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"embedding shard must be BF16, got {weight.dtype}")
        self.group, self.weight = get_tp_group(), weight.detach()
        self.start, self.vocab = rank * weight.shape[0], embedding.org_vocab_size
        with torch.accelerator.device_index(weight.device.index):
            arch, ids = architecture(weight.device), (torch.int32, torch.int64)
            self.kernels = {t: compile_gather(t, weight.shape[0], arch) for t in ids}

    def run(self, ids, out):
        if ids.dim() != 1 or ids.dtype not in self.kernels:
            raise ValueError(
                f"token ids must be 1-D int32/int64, got {ids.dtype} {tuple(ids.shape)}"
            )
        if out.shape != (ids.shape[0], 7168) or not out.is_contiguous():
            raise ValueError(
                f"embedding output must be contiguous [{ids.shape[0]}, 7168]"
            )
        self.kernels[ids.dtype](ids, self.weight, out, self.start, self.vocab)
        reduced = self.group.all_reduce(out)
        if reduced is not out:
            out.copy_(reduced)
        return out
