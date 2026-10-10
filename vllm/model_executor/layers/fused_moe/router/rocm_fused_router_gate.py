# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused router gate for gfx950: gate GEMM plus sqrtsoftplus expert selection.

A split-K BF16 matrix multiplication reuses gate weights across token rows.
The second kernel fuses the FP32 partial-sum reduction, sqrtsoftplus scoring,
and deterministic expert selection.
"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import gl, gluon, tl, triton

# (hidden_size, num_experts): DeepSeek-V4.1-Flash and DeepSeek-V4-Pro.
ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES = frozenset({(5120, 384), (7168, 384)})
_MAX_TOKENS = 1536


@triton.jit
def _router_gate_softplus_sqrt(logits):
    exp_value = tl.exp(-tl.abs(logits))
    rounded_sum = 1.0 + exp_value
    # Recover exp_value when adding it to one rounds away a negative tail.
    correction = exp_value - (rounded_sum - 1.0)
    softplus_tail = tl.log(rounded_sum) + correction / rounded_sum
    return tl.sqrt(tl.maximum(logits, 0.0) + softplus_tail)


@gluon.jit
def _router_gate_softplus_sqrt_gluon(logits):
    exp_value = gl.exp(-gl.abs(logits))
    rounded_sum = 1.0 + exp_value
    correction = exp_value - (rounded_sum - 1.0)
    softplus_tail = gl.log(rounded_sum) + correction / rounded_sum
    return gl.sqrt(gl.maximum(logits, 0.0) + softplus_tail)


@gluon.jit
def _router_gate_row_bias_gluon(
    correction_bias_ptr,
    bias_vl_ptr,
    input_ids_ptr,
    image_sentinel_lo,
    row,
    experts,
    N: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_BIAS_VL: gl.constexpr,
):
    bias = gl.zeros_like(experts.to(gl.float32))
    if HAS_BIAS:
        bias = gl.load(correction_bias_ptr + experts, experts < N, 0.0)
    if HAS_BIAS_VL:
        token = gl.load(input_ids_ptr + row).to(gl.int64)
        is_image = (token >= image_sentinel_lo) & (token < image_sentinel_lo + 5)
        if is_image:
            bias = gl.load(bias_vl_ptr + experts, experts < N, 0.0)
    return bias


@gluon.jit
def _router_gate_reduce_topk_gluon(
    partial_logits_ptr,
    correction_bias_ptr,
    bias_vl_ptr,
    input_ids_ptr,
    is_padding_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    routed_scaling_factor,
    image_sentinel_lo,
    M,
    N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    TOPK: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_BIAS_VL: gl.constexpr,
    HAS_PADDING: gl.constexpr,
    RENORMALIZE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_S: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
):
    gl.static_assert(TOPK == 6 or TOPK == 8)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    row = gl.program_id(0)
    experts = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    splits = gl.arange(0, BLOCK_S, layout=gl.SliceLayout(1, layout))
    partial = gl.load(
        partial_logits_ptr + splits[:, None] * M * N + row * N + experts[None, :],
        mask=(splits[:, None] < SPLIT_K) & (experts[None, :] < N),
        other=0.0,
    )
    logits = gl.sum(partial, axis=0)
    scores = _router_gate_softplus_sqrt_gluon(logits)
    ranked = scores
    if HAS_BIAS or HAS_BIAS_VL:
        ranked += _router_gate_row_bias_gluon(
            correction_bias_ptr,
            bias_vl_ptr,
            input_ids_ptr,
            image_sentinel_lo,
            row,
            experts,
            N,
            HAS_BIAS,
            HAS_BIAS_VL,
        )
    ranked = gl.where(experts < N, ranked, -float("inf"))
    ranked = gl.where(ranked == 0.0, 0.0, ranked)
    bits = ranked.to(gl.uint32, bitcast=True)
    ordered = gl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(gl.uint64) << 32) | (BLOCK_N - experts).to(gl.uint64)

    slots = gl.arange(0, BLOCK_TOPK, layout=gl.SliceLayout(0, layout))
    selected_ids = gl.full((BLOCK_TOPK,), 0, gl.int32, layout=gl.SliceLayout(0, layout))
    for slot in gl.static_range(TOPK):
        selected_key = gl.max(keys, axis=0)
        expert = BLOCK_N - (selected_key & 0xFFFFFFFF).to(gl.int32)
        selected_ids = gl.where(slots == slot, expert, selected_ids)
        keys = gl.where(experts == expert, 0, keys)

    selected_weights = gl.gather(scores, selected_ids, axis=0)
    selected_weights = gl.where(slots < TOPK, selected_weights, 0.0)
    scale = routed_scaling_factor
    if RENORMALIZE:
        weight_sum = gl.sum(selected_weights, axis=0)
        scale = scale / gl.where(weight_sum > 0.0, weight_sum, 1.0)
    selected_weights = selected_weights * scale
    # A constexpr guard: merging it into the load's condition would trace a
    # load from the absent padding pointer.
    if HAS_PADDING:  # noqa: SIM102
        # Padding rows get expert -1, which the fused MoE skips.
        if gl.load(is_padding_ptr + row):
            selected_ids = gl.full_like(selected_ids, -1)
            selected_weights = gl.zeros_like(selected_weights)
    gl.store(
        topk_weights_ptr + row * TOPK + slots,
        selected_weights,
        mask=slots < TOPK,
    )
    gl.store(
        topk_ids_ptr + row * TOPK + slots,
        selected_ids,
        mask=slots < TOPK,
    )


@triton.jit(do_not_specialize=["M"])
def _router_gate_gemv(
    hidden_states_ptr,
    router_weight_ptr,
    logits_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) % M
    expert = tl.program_id(0) // M
    offsets = tl.arange(0, BLOCK_K)
    hidden = tl.load(hidden_states_ptr + row * K + offsets, offsets < K, 0.0)
    weight = tl.load(router_weight_ptr + expert * K + offsets, offsets < K, 0.0)
    logits = tl.sum(hidden.to(tl.float32) * weight.to(tl.float32), axis=0)
    tl.store(logits_ptr + row * N + expert, logits)


@triton.jit(do_not_specialize=["M"])
def _router_gate_gemm(
    hidden_states_ptr,
    router_weight_ptr,
    partial_logits_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
):
    if XCD_SWIZZLE:
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        num_blocks = num_m * num_n * tl.cdiv(K, CHUNK_K)
        pid = tl.program_id(0)
        # Assign adjacent expert tiles to the same XCD's L2 cache.
        pid = (pid % 8) * tl.cdiv(num_blocks, 8) + pid // 8
        if pid >= num_blocks:
            return
        split = pid // (num_m * num_n)
        tile = pid % (num_m * num_n)
        row_block, expert_block = tile // num_n, tile % num_n
    else:
        row_block, expert_block, split = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    experts = expert_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = split * CHUNK_K + tl.arange(0, BLOCK_K)
    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(CHUNK_K, BLOCK_K)):
        hidden = tl.load(
            hidden_states_ptr + rows[:, None] * K + offsets_k[None, :],
            mask=(rows[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            router_weight_ptr + experts[None, :] * K + offsets_k[:, None],
            mask=(experts[None, :] < N) & (offsets_k[:, None] < K),
            other=0.0,
        )
        logits = tl.dot(hidden, weight, logits)
        offsets_k += BLOCK_K
    tl.store(
        partial_logits_ptr + split * M * N + rows[:, None] * N + experts[None, :],
        logits,
        mask=(rows[:, None] < M) & (experts[None, :] < N),
    )


@triton.jit(do_not_specialize=["M", "image_sentinel_lo"])
def _router_gate_reduce_topk(
    partial_logits_ptr,
    correction_bias_ptr,
    bias_vl_ptr,
    input_ids_ptr,
    is_padding_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    routed_scaling_factor,
    image_sentinel_lo,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BIAS_VL: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    splits = tl.arange(0, BLOCK_S)
    partial = tl.load(
        partial_logits_ptr + splits[:, None] * M * N + row * N + experts[None, :],
        mask=(splits[:, None] < SPLIT_K) & (experts[None, :] < N),
        other=0.0,
    )
    logits = tl.sum(partial, axis=0)
    scores = _router_gate_softplus_sqrt(logits)
    ranked = scores
    if HAS_BIAS:
        bias = tl.load(correction_bias_ptr + experts, mask=experts < N, other=0.0)
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS_VL:
        # Image sentinel tokens (five consecutive ids from image_sentinel_lo)
        # rank experts with bias_vl instead of the text correction bias.
        token = tl.load(input_ids_ptr + row).to(tl.int64)
        if (token >= image_sentinel_lo) & (token < image_sentinel_lo + 5):
            bias = tl.load(bias_vl_ptr + experts, mask=experts < N, other=0.0)
    ranked += bias
    ranked = tl.where(experts < N, ranked, -float("inf"))
    ranked = tl.where(ranked == 0.0, 0.0, ranked)

    # Sort the full FP32 score, then the inverse expert id for exact ties.
    bits = ranked.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (BLOCK_N - experts).to(tl.uint64)
    if BLOCK_TOPK == 1:
        selected_keys = tl.max(keys, axis=0)[None]
    else:
        selected_keys = tl.topk(keys, BLOCK_TOPK)
    selected_ids = BLOCK_N - (selected_keys & 0xFFFFFFFF).to(tl.int32)
    selected_weights = tl.gather(scores, selected_ids, axis=0)
    slots = tl.arange(0, BLOCK_TOPK)
    selected_weights = tl.where(slots < TOPK, selected_weights, 0.0)
    scale = routed_scaling_factor
    if RENORMALIZE:
        weight_sum = tl.sum(selected_weights, axis=0)
        scale = scale / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    selected_weights = selected_weights * scale
    # A constexpr guard: merging it into the load's condition would trace a
    # load from the absent padding pointer.
    if HAS_PADDING:  # noqa: SIM102
        # Padding rows get expert -1, which the fused MoE skips.
        if tl.load(is_padding_ptr + row):
            selected_ids = tl.full(selected_ids.shape, -1, tl.int32)
            selected_weights = tl.zeros(selected_weights.shape, tl.float32)
    tl.store(
        topk_weights_ptr + row * TOPK + slots,
        selected_weights,
        mask=slots < TOPK,
    )
    tl.store(topk_ids_ptr + row * TOPK + slots, selected_ids, mask=slots < TOPK)


def can_use_rocm_fused_router_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    bias_vl: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> bool:
    """Return whether the tensors match the tuned gfx950 fused gate."""
    try:
        _validate_inputs(
            hidden_states,
            router_weight,
            correction_bias,
            topk,
            bias_vl,
            input_ids,
            is_padding,
        )
    except (RuntimeError, ValueError):
        return False
    return True


def _validate_vector(
    name: str, t: torch.Tensor, length: int, dtypes, device: torch.device
) -> None:
    if t.device != device:
        raise ValueError(f"{name} must be on the same device as hidden_states")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.dtype not in dtypes:
        raise ValueError(f"{name} must have dtype in {dtypes}")
    if t.dim() != 1 or t.shape[0] < length:
        raise ValueError(f"{name} must be 1D with at least {length} entries")


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    bias_vl: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> None:
    if not current_platform.is_rocm() or not on_gfx950():
        raise RuntimeError("rocm_fused_router_gate requires ROCm gfx950")
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("hidden_states and router_weight must be 2D tensors")
    if hidden_states.dtype != torch.bfloat16 or router_weight.dtype != torch.bfloat16:
        raise ValueError("hidden_states and router_weight must have dtype bfloat16")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError("hidden_states and router_weight must be contiguous")
    if hidden_states.device.type != "cuda":
        raise ValueError("hidden_states and router_weight must be on a GPU")
    if hidden_states.device != router_weight.device:
        raise ValueError("hidden_states and router_weight must be on the same device")
    shape = (hidden_states.shape[1], router_weight.shape[0])
    if (
        shape not in ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES
        or router_weight.shape[1] != shape[0]
    ):
        raise ValueError(
            "supported (hidden_size, num_experts) pairs are "
            f"{sorted(ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES)}"
        )
    if not 0 <= hidden_states.shape[0] <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [0, {_MAX_TOKENS}]")
    if not 0 < topk <= router_weight.shape[0]:
        raise ValueError("topk must be in (0, num_experts]")
    num_tokens, num_experts = hidden_states.shape[0], router_weight.shape[0]
    device = hidden_states.device
    for name, bias in (("correction_bias", correction_bias), ("bias_vl", bias_vl)):
        if bias is not None:
            _validate_vector(name, bias, num_experts, (torch.float32,), device)
            if bias.shape != (num_experts,):
                raise ValueError(f"{name} must have shape (num_experts,)")
    if bias_vl is not None:
        if input_ids is None:
            raise ValueError("bias_vl routing requires input_ids")
        _validate_vector(
            "input_ids", input_ids, num_tokens, (torch.int32, torch.int64), device
        )
    if is_padding is not None:
        _validate_vector("is_padding", is_padding, num_tokens, (torch.bool,), device)


def rocm_fused_router_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    indices_dtype: torch.dtype = torch.int32,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    input_ids: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score ``hidden_states @ router_weight.T`` and select the top experts.

    Args:
        hidden_states: Token rows, shape ``(num_tokens, hidden_size)``.
        router_weight: Gate weight, shape ``(num_experts, hidden_size)``.
        correction_bias: Per-expert selection bias, or None.
        topk: Number of experts to select per token.
        renormalize: Whether to normalize the selected weights to sum to one.
        routed_scaling_factor: Scale applied to the selected weights.
        indices_dtype: Dtype of the returned expert ids.
        bias_vl: Selection bias for image sentinel tokens, or None.
        image_sentinel_lo: First of the five image sentinel token ids.
        input_ids: Token ids, required with ``bias_vl``.
        is_padding: Per-token padding mask; padding rows get expert ``-1``.

    Returns:
        The routing weights and expert ids, both shaped ``(num_tokens, topk)``.

    """
    if image_sentinel_lo <= 0:
        bias_vl = None
    _validate_inputs(
        hidden_states,
        router_weight,
        correction_bias,
        topk,
        bias_vl,
        input_ids,
        is_padding,
    )
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("indices_dtype must be int32 or int64")
    num_tokens = hidden_states.shape[0]
    num_experts = router_weight.shape[0]
    topk_weights = hidden_states.new_empty((num_tokens, topk), dtype=torch.float32)
    topk_ids = hidden_states.new_empty((num_tokens, topk), dtype=indices_dtype)
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_n, block_k = 64, 128
    num_warps, num_stages, matrix_instr_nonkdim = 4, 2, 16
    target_splits = 14 if num_tokens <= 64 else 7
    if num_tokens <= 16:
        block_m = 16
    elif num_tokens <= 128:
        block_m = 32
    elif num_tokens <= 256 or 512 < num_tokens <= 768:
        block_m = 64
    else:
        block_m = 128
        num_warps, num_stages, matrix_instr_nonkdim = 8, 3, 32
    xcd_swizzle = num_tokens > 768
    if xcd_swizzle:
        block_n, block_k = 128, 64
    chunk_k = triton.cdiv(hidden_states.shape[1], target_splits * block_k) * block_k
    split_k = triton.cdiv(hidden_states.shape[1], chunk_k)
    if num_tokens <= 2:
        split_k = 1
    partial_logits = hidden_states.new_empty(
        (split_k, num_tokens, num_experts), dtype=torch.float32
    )
    if num_tokens <= 2:
        _router_gate_gemv[(num_tokens * num_experts,)](
            hidden_states,
            router_weight,
            partial_logits,
            M=num_tokens,
            K=hidden_states.shape[1],
            N=num_experts,
            BLOCK_K=triton.next_power_of_2(hidden_states.shape[1]),
            num_warps=4,
        )
    else:
        num_m = triton.cdiv(num_tokens, block_m)
        num_n = triton.cdiv(num_experts, block_n)
        grid = (
            (triton.cdiv(num_m * num_n * split_k, 8) * 8,)
            if xcd_swizzle
            else (num_m, num_n, split_k)
        )
        _router_gate_gemm[grid](
            hidden_states,
            router_weight,
            partial_logits,
            M=num_tokens,
            K=hidden_states.shape[1],
            N=num_experts,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            CHUNK_K=chunk_k,
            XCD_SWIZZLE=xcd_swizzle,
            num_warps=num_warps,
            num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
        )
    use_gluon = num_tokens >= 128 and topk in (6, 8)
    select_kernel = (
        _router_gate_reduce_topk_gluon if use_gluon else _router_gate_reduce_topk
    )
    select_kernel[(num_tokens,)](
        partial_logits,
        correction_bias,
        bias_vl,
        input_ids,
        is_padding,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
        image_sentinel_lo,
        num_tokens,
        N=num_experts,
        SPLIT_K=split_k,
        TOPK=topk,
        HAS_BIAS=correction_bias is not None,
        HAS_BIAS_VL=bias_vl is not None,
        HAS_PADDING=is_padding is not None,
        RENORMALIZE=renormalize,
        BLOCK_N=triton.next_power_of_2(num_experts),
        BLOCK_S=triton.next_power_of_2(split_k),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=1 if use_gluon else 4,
    )
    return topk_weights, topk_ids
