# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _argmax_partial(
    X,
    BIAS,
    PV,
    PI,
    N: tl.constexpr,
    SX: tl.constexpr,
    SB: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row.to(tl.int64) * SX + col, col < N, float("-inf"))
    if HAS_BIAS:
        bias = tl.load(BIAS + row.to(tl.int64) * SB + col, col < N, 0.0)
        # Preserve the rounding of the materialized `logits + bias` tensor.
        value = (value.to(tl.float32) + bias.to(tl.float32)).to(value.dtype)
    value = value.to(tl.float32)
    # NaN breaks tl.argmax index bounds, as in the rejection sampler.
    value = tl.where(value != value, float("-inf"), value)
    index = tl.argmax(value, axis=0)
    tl.store(PV + row * SPLITS + part, tl.max(value, axis=0))
    tl.store(PI + row * SPLITS + part, part * BLOCK + index)


@triton.jit
def _argmax_finish(PV, PI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(PV + row * SPLITS + col, col < SPLITS, float("-inf"))
    part = tl.argmax(value, axis=0)
    tl.store(OUT + row, tl.load(PI + row * SPLITS + part).to(tl.int64))


def greedy_argmax(
    logits: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Argmax of `logits + bias`, treating NaNs as -inf on every path.
    Ties resolve to the first index. The optional bias addition is fused for
    matching dtypes and shapes with contiguous vocabulary dimensions.
    """
    if (
        not logits.is_cuda
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or logits.ndim != 2
        or logits.stride(1) != 1
        or logits.shape[1] < 4096
        or (
            bias is not None
            and (
                bias.shape != logits.shape
                or bias.dtype != logits.dtype
                or bias.device != logits.device
                or bias.stride(1) != 1
            )
        )
    ):
        if bias is not None:
            logits = logits + bias
        return logits.masked_fill(logits.isnan(), float("-inf")).argmax(dim=-1)
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    block = 4096
    splits = triton.cdiv(vocab, block)
    values = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    _argmax_partial[(rows, splits)](
        logits,
        bias,
        values,
        indices,
        vocab,
        logits.stride(0),
        bias.stride(0) if bias is not None else 0,
        splits,
        block,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    _argmax_finish[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out
