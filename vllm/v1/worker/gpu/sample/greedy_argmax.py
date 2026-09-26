# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _argmax_pair(av, ai, bv, bi):
    an, bn = av != av, bv != bv
    take_a = (an & ~bn) | ((an == bn) & ((av > bv) | (((av == bv) | an) & (ai < bi))))
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


@triton.jit
def _argmax_partial(
    X,
    PV,
    PI,
    N: tl.constexpr,
    SX: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row.to(tl.int64) * SX + col, col < N, float("-inf"))
    index = tl.where(col < N, col, N)
    value, index = tl.reduce((value.to(tl.float32), index), 0, _argmax_pair)
    tl.store(PV + row * SPLITS + part, value)
    tl.store(PI + row * SPLITS + part, index)


@triton.jit
def _argmax_finish(PV, PI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(PV + row * SPLITS + col, col < SPLITS, float("-inf"))
    index = tl.load(PI + row * SPLITS + col, col < SPLITS, 0x7FFFFFFF)
    _, index = tl.reduce((value, index), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))


def greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    if (
        not logits.is_cuda
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or logits.ndim != 2
        or logits.stride(1) != 1
        or logits.shape[1] < 4096
    ):
        return logits.argmax(dim=-1)
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    block = 4096
    splits = triton.cdiv(vocab, block)
    values = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    _argmax_partial[(rows, splits)](
        logits, values, indices, vocab, logits.stride(0), splits, block, num_warps=4
    )
    _argmax_finish[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out
