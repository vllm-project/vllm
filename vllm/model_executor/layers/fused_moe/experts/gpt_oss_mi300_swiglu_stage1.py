# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MI300-specialised fused MXFP4 stage-1 SwiGLU for GPT-OSS.

Specialises the openai/gpt-oss-20b first MoE matmul + SwiGLU fusion for
the small ragged decode shapes that dominate steady-state generation on a
single MI300X. Falls back to the generic ``matmul_ogs`` path whenever any
shape/dtype/router precondition is not met, so other models, hardware,
and prefill remain on the original code path.
"""

from __future__ import annotations

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_MI300_GATHER_ROWS = {
    4,
    8,
    16,
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
}
_MI300_BLOCK_M = 16
_MI300_BLOCK_N = 64
_MI300_BLOCK_K = 256
_MI300_NUM_WARPS = 8
_MI300_NUM_STAGES = 2
_MI300_MAX_EXPECTED_SLICE_SIZE = 128
_HIDDEN_DIM = 3072
_NUM_EXPERTS = 32
_ROUTES_PER_TOKEN = 4
_SWIGLU_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0


@triton.jit
def _mi300_swiglu_stage1_kernel(
    A,
    stride_am,
    stride_ak,
    GatherIndx,
    W,
    stride_we,
    stride_wk,
    stride_wn,
    WScale,
    stride_se,
    stride_sk,
    stride_sn,
    Bias,
    stride_be,
    stride_bn,
    SliceSizes,
    SliceOffs,
    BlockSchedule,
    Out,
    stride_om,
    stride_on,
    N_FINAL,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    K_DIM: tl.constexpr,
    ROUTES_PER_TOKEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mb = tl.program_id(0)
    pid_n = tl.program_id(1)

    packed_schedule = tl.load(BlockSchedule + pid_mb)
    valid_block = packed_schedule != -1
    safe_schedule = tl.where(valid_block, packed_schedule, 0)
    expert_id = safe_schedule & 65535
    block_in_slice = safe_schedule >> 16

    slice_size = tl.load(SliceSizes + expert_id, mask=valid_block, other=0)
    slice_off = tl.load(SliceOffs + expert_id, mask=valid_block, other=0)
    offs_m_local = block_in_slice * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = slice_off + offs_m_local
    mask_m = valid_block & (offs_m_local < slice_size)

    # Live vLLM passes routed-row indices: token_id * topk + route_slot. The
    # activation matrix is token-major, so reconstruct the token row in-kernel.
    token_idx = tl.load(GatherIndx + offs_m, mask=mask_m, other=0) // ROUTES_PER_TOKEN

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_pair = tl.reshape(
        offs_n[:, None] * 2 + tl.arange(0, 2)[None, :],
        (BLOCK_N * 2,),
    )
    mask_n = offs_n < N_FINAL
    mask_n_pair = tl.reshape(
        mask_n[:, None].broadcast_to(BLOCK_N, 2),
        (BLOCK_N * 2,),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        mask_k = k0 + offs_k < K_DIM
        mask_wk = k0 // 2 + tl.arange(0, BLOCK_K // 2) < tl.cdiv(K_DIM, 2)
        mask_sk = k0 // 32 + tl.arange(0, BLOCK_K // 32) < tl.cdiv(K_DIM, 32)
        a = tl.load(
            A + token_idx[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            W
            + expert_id * stride_we
            + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_wk
            + offs_n_pair[None, :] * stride_wn,
            mask=mask_wk[:, None] & mask_n_pair[None, :],
            other=0,
        )
        w_scale = tl.load(
            WScale
            + expert_id * stride_se
            + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :] * stride_sk
            + offs_n_pair[:, None] * stride_sn,
            mask=mask_n_pair[:, None] & mask_sk[None, :],
            other=0,
        )
        acc = tl.dot_scaled(
            a,
            None,
            "bf16",
            w,
            w_scale,
            "e2m1",
            acc=acc,
            rhs_k_pack=True,
            fast_math=True,
        )

    bias = tl.load(
        Bias + expert_id * stride_be + offs_n_pair * stride_bn,
        mask=mask_n_pair,
        other=0.0,
    )
    acc += bias[None, :]

    gelu, linear = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_N, 2)))
    gelu = tl.minimum(gelu.to(tl.float32), SWIGLU_LIMIT)
    linear = tl.clamp(linear.to(tl.float32), -SWIGLU_LIMIT, SWIGLU_LIMIT)
    s = gelu / (1 + tl.exp(-SWIGLU_ALPHA * gelu))
    out = tl.fma(s, linear, s).to(tl.bfloat16)

    tl.store(
        Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def run_mi300_swiglu_stage1(
    hidden_states: torch.Tensor,
    w1,
    routing_data,
    gather_indx,
    precision_config,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    apply_router_weight_on_input: bool,
    swiglu_alpha: float,
    swiglu_limit: float,
) -> bool:
    """Try the MI300 fused MXFP4 SwiGLU stage-1 fast path.

    Returns ``True`` if the call was fully handled (``out`` is populated
    with the post-SwiGLU bf16 output), ``False`` if any precondition was
    not met and the caller should fall through to the generic
    ``matmul_ogs`` + SwiGLU path.
    """
    # Hard gates: hardware, then explicit kill-switch. These must fire
    # before any tensor attribute access so the function is a cheap no-op
    # on every non-target platform (NVIDIA, MI250, CPU, etc.).
    if not current_platform.is_rocm():
        return False
    # gfx942 = MI300X / MI325X. MI355 (gfx950) and prior MI2xx archs
    # take the generic path; only the validated arch hits the fast path.
    # ``is_device_capability`` is the strict (==) form; ``has_device_capability``
    # would also let gfx950 through.
    if not current_platform.is_device_capability((9, 4)):
        return False
    if os.environ.get("VLLM_DISABLE_MI300_GPTOSS_SWIGLU", "0") == "1":
        return False

    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        return False
    if hidden_states.shape[1] != _HIDDEN_DIM:
        return False
    if swiglu_alpha != _SWIGLU_ALPHA or swiglu_limit != _SWIGLU_LIMIT:
        return False
    if apply_router_weight_on_input:
        return False
    if routing_data is None or routing_data.expt_data is None or gather_indx is None:
        return False

    gather_route_rows = gather_indx.src_indx
    if (
        not isinstance(gather_route_rows, torch.Tensor)
        or gather_route_rows.dtype != torch.int32
        or gather_route_rows.ndim != 1
    ):
        return False
    if gather_route_rows.numel() not in _MI300_GATHER_ROWS:
        return False
    if (
        hidden_states.shape[0] <= 0
        or gather_route_rows.numel() % hidden_states.shape[0] != 0
    ):
        return False
    if gather_route_rows.numel() // hidden_states.shape[0] != _ROUTES_PER_TOKEN:
        return False

    pre_act_dim = _HIDDEN_DIM * 2
    if (
        bias is None
        or bias.dtype != torch.float32
        or tuple(bias.shape) != (_NUM_EXPERTS, pre_act_dim)
    ):
        return False
    if tuple(getattr(w1, "shape", ())) != (_NUM_EXPERTS, _HIDDEN_DIM, pre_act_dim):
        return False
    if not hasattr(w1, "storage") or not hasattr(w1.storage, "data"):
        return False

    weight_scale = getattr(precision_config, "weight_scale", None)
    if weight_scale is None:
        return False
    if tuple(getattr(weight_scale, "shape", ())) != (
        _NUM_EXPERTS,
        _HIDDEN_DIM // 32,
        pre_act_dim,
    ):
        return False
    if not hasattr(weight_scale, "storage") or not hasattr(
        weight_scale.storage, "data"
    ):
        return False
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (
        gather_route_rows.numel(),
        _HIDDEN_DIM,
    ):
        return False

    metadata = routing_data.expt_data
    expected_slice_size = getattr(routing_data, "expected_tokens_per_expt", None)
    if expected_slice_size is None:
        n_slices = int(metadata.slice_sizes.numel())
        expected_slice_size = max(
            1, (gather_route_rows.numel() + n_slices - 1) // n_slices
        )
    if expected_slice_size > _MI300_MAX_EXPECTED_SLICE_SIZE:
        return False

    block_schedule = metadata.block_schedule(_MI300_BLOCK_M)
    grid = (len(block_schedule), triton.cdiv(out.shape[1], _MI300_BLOCK_N))
    value = w1.storage.data
    scale = weight_scale.storage.data
    _mi300_swiglu_stage1_kernel[grid](
        hidden_states,
        hidden_states.stride(0),
        hidden_states.stride(1),
        gather_route_rows,
        value,
        *value.stride(),
        scale,
        *scale.stride(),
        bias,
        *bias.stride(),
        metadata.slice_sizes,
        metadata.slice_offs,
        block_schedule,
        out,
        out.stride(0),
        out.stride(1),
        out.shape[1],
        SWIGLU_ALPHA=_SWIGLU_ALPHA,
        SWIGLU_LIMIT=_SWIGLU_LIMIT,
        K_DIM=_HIDDEN_DIM,
        ROUTES_PER_TOKEN=_ROUTES_PER_TOKEN,
        BLOCK_M=_MI300_BLOCK_M,
        BLOCK_N=_MI300_BLOCK_N,
        BLOCK_K=_MI300_BLOCK_K,
        num_warps=_MI300_NUM_WARPS,
        num_stages=_MI300_NUM_STAGES,
    )
    return True
