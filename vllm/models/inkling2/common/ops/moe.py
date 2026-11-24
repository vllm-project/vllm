# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE routing and combine: the router top-k with its routing
weights, the routed rows' softcapped SwiGLU, and the top-k combine.

`select_experts` / `swiglu` keep the rows in the flat (token * topk + slot)
order of the expert GEMMs over checkpoint-layout weights; the routing
helpers are shared with the fused routing kernels of `nvidia/ops/moe.py`.
"""

import torch

from vllm.models.inkling2.nvidia.ops.seq_parallel import HALO_ROWS, SeqParallel
from vllm.triton_utils import tl, tldevice, triton

from .activation import _situ
from .linear import k_group_partials
from .pdl import gdc_launch_dependents, gdc_wait
from .platform import pdl_launch_kwargs

# Up to this many tokens, kernels let the next kernel launch (and wait) as soon
# as they start, which hides its launch; with more, its waiting programs would
# hold back theirs.
LAUNCH_DEPENDENTS_MAX_TOKENS = 256


@triton.jit
def _sum8(x):
    # ((x0 + x2) + (x1 + x3)) + ((x4 + x6) + (x5 + x7)) of each row of [R, 8]:
    # the summation order of the reference renormalization kernel, so routing
    # weights match bitwise.
    low, high = tl.split(tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2)), (0, 1, 3, 2)))
    low, high = tl.split(low + high)
    low, high = tl.split(low + high)
    return low + high


@triton.jit
def _selection_keys(logits, bias, experts, E: tl.constexpr):
    """Sort keys of the experts `experts` (padding: 0): order-preserving
    bits of sigmoid(logit) + bias in the high word and the reversed expert
    index in the low bits, so ties prefer the lower expert index."""
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-logits))
    selection = sigmoid + bias
    bits = selection.to(tl.uint32, bitcast=True)
    bits = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    keys = (bits.to(tl.uint64) << 16) | (E - experts).to(tl.uint64)
    return tl.where(experts < E, keys, 0)


@triton.jit
def _key_values(values, keys, top):
    """`values` [R, C] where `keys` [R, C] equals each of `top` [R, K]."""
    hit = keys[:, None, :] == top[:, :, None]
    return tl.max(tl.where(hit, values[:, None, :], float("-inf")), 2)


@triton.jit
def _log_sigmoid(logits):
    return tl.minimum(logits, 0.0) - tldevice.log1p(tldevice.exp(-tl.abs(logits)))


@triton.jit
def _routing_weights(log_probs, route_scale, TOPK: tl.constexpr):
    """Renormalized sigmoid probabilities [R, TOPK] of the selected experts
    from their log-sigmoids."""
    max_log_probs = tl.max(log_probs, 1)
    shifted = tldevice.exp(log_probs - max_log_probs[:, None])
    total = _sum8(shifted) if TOPK == 8 else tl.sum(shifted, 1)
    logsumexp = max_log_probs + tldevice.log(total)
    return tldevice.exp(log_probs - logsumexp[:, None]) * route_scale


@triton.jit
def _group_sum(Partials, rows, cols, mask, T, E: tl.constexpr, GROUPS: tl.constexpr):
    """In-order sum of the K-group partials `[GROUPS, T, E]` at `(rows,
    cols)`."""
    total = tl.load(Partials + rows * E + cols, mask, other=0.0, cache_modifier=".cg")
    for group in tl.static_range(1, GROUPS):
        total = total + tl.load(
            Partials + (group * T + rows) * E + cols,
            mask,
            other=0.0,
            cache_modifier=".cg",
        )
    return total


@triton.jit
def _load_bias(Bias, E: tl.constexpr, E_POW2: tl.constexpr):
    """The selection bias [1, E_POW2] (padding: 0) for `_select_experts`; a
    weight, so it may load before `gdc_wait`."""
    offs = tl.arange(0, E_POW2)[None, :]
    return tl.load(Bias + offs, offs < E, other=0.0)


@triton.jit
def _top_keys(
    Partials,
    bias,
    tokens,
    T,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
):
    """The logits and `_selection_keys` [R, E_POW2] of tokens `tokens` [R]
    and their top-k keys [R, TOPK] (`_select_experts`)."""
    offs = tl.arange(0, E_POW2)[None, :]
    logits = _group_sum(Partials, tokens[:, None], offs, offs < E, T, E, GROUPS)
    keys = _selection_keys(logits, bias, offs, E)
    return logits, keys, tl.topk(keys, TOPK)


@triton.jit
def _select_experts(
    Partials,
    bias,
    tokens,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Top-k ids and routing weights [R, TOPK] of tokens `tokens` [R] from
    the K-group logit partials `[GROUPS, T, E]` and the `_load_bias`
    `bias`, by `tl.topk` (`SORT`) or k max passes."""
    rows = tokens[:, None]
    offs = tl.arange(0, E_POW2)[None, :]
    if SORT:
        logits, keys, top = _top_keys(
            Partials, bias, tokens, T, E, E_POW2, GROUPS, TOPK
        )
        ids = E - (top & 0xFFFF).to(tl.int32)
        selected = _key_values(logits, keys, top)
    else:
        logits = _group_sum(Partials, rows, offs, offs < E, T, E, GROUPS)
        keys = _selection_keys(logits, bias, offs, E)
        # Fewer instructions than the bitonic top-k: better for large batches.
        slots = tl.arange(0, TOPK)[None, :]
        ids = tl.zeros((tokens.shape[0], TOPK), tl.int32)
        for k in tl.static_range(TOPK):
            best = tl.max(keys, 1)[:, None]
            ids = tl.where(slots == k, E - (best & 0xFFFF).to(tl.int32), ids)
            keys = tl.where(keys == best, 0, keys)
        # The selected logits, summed again in the same order.
        selected = _group_sum(Partials, rows, ids, ids >= 0, T, E, GROUPS)
    return ids, _routing_weights(_log_sigmoid(selected), route_scale, TOPK)


@triton.jit
def _local_experts(ids, lo, hi, padding):
    """Local index of the experts `ids` [R, TOPK] among the local experts
    `[lo, hi)` (-1: not local, or the token is `padding`; None: none
    is)."""
    local = tl.where((ids >= lo) & (ids < hi), ids - lo, -1)
    if padding is not None:
        local = tl.where(padding[:, None], -1, local)
    return local


@triton.jit
def _combine_kernel(
    Down,
    Src2Dst,
    Out,
    SlotBases,
    tokens,
    shard,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
    PUSH: tl.constexpr,
    HALO_ROWS: tl.constexpr,
):
    gdc_wait()
    if LAUNCH_DEPENDENTS:
        gdc_launch_dependents()
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        row = tl.load(Src2Dst + token * TOPK + slot).to(tl.int64)
        acc += tl.load(Down + row * H + cols, row >= 0, other=0).to(tl.float32)
    if PUSH:
        # Into the owner's slot of this rank, after its halo rows; the last
        # halo rows of a shard also into the next owner's.
        owner = token // shard
        local = token - owner * shard
        dst = tl.load(SlotBases + owner).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst + (local + HALO_ROWS) * H + cols, acc.to(tl.bfloat16))
        if local >= shard - HALO_ROWS and (owner + 1) * shard < tokens:
            dst = tl.load(SlotBases + owner + 1).to(tl.pointer_type(tl.bfloat16))
            tl.store(dst + (local - shard + HALO_ROWS) * H + cols, acc.to(tl.bfloat16))
    else:
        tl.store(Out + token * H + cols, acc)


def combine(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Sum each token's rows of `down` (rows -1 count 0)."""
    tokens = src2dst.numel() // topk
    out = torch.empty((tokens, down.shape[1]), dtype=out_dtype, device=down.device)
    _combine(down, src2dst, topk, out, None)
    return out


def _combine(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    out: torch.Tensor | None,
    sp: SeqParallel | None,
) -> None:
    hidden = down.shape[1]
    tokens = src2dst.numel() // topk
    block = 512 if hidden % 512 == 0 else 256
    assert hidden % block == 0
    _combine_kernel[(tokens, hidden // block)](
        down,
        src2dst,
        out,
        None if sp is None else sp.slot_bases,
        tokens,
        1 if sp is None else sp.shard(tokens)[2],
        hidden,
        topk,
        block,
        tokens <= LAUNCH_DEPENDENTS_MAX_TOKENS,
        sp is not None,
        HALO_ROWS,
        num_warps=2,
        **pdl_launch_kwargs(),
    )


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere).
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        num_warps=4 if tokens <= 256 else 1,
    )
    return ids, weights, src2dst


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out
