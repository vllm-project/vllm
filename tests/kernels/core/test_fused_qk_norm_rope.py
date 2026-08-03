# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [13]
PARTIAL_ROPE = [True, False]
CUDA_DEVICES = ["cuda:0"]


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    assert isinstance(q_by_head, torch.Tensor)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    assert isinstance(k_by_head, torch.Tensor)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    rotary_ratio: float,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data
    rotary_dim = int(head_dim * rotary_ratio)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    ref_result = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
        (
            qkv_fused.clone(),
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        ),
    )

    torch.ops._C.fused_qk_norm_rope(
        qkv_fused,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions.view(-1),
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


# The auto-select heuristic only picks token_heads_per_warp > 1 on SM90 for large
# batches, so the multi-head-per-warp packing kernel is never exercised by the
# num_tokens=4 test above (it always dispatches the 1-head baseline). The tests
# below force token_heads_per_warp to cover the packing kernel directly.
HEAD_DIMS = [64, 128, 256]  # dims supported by the kernel dispatch


def _build_qk_norm_rope_inputs(
    device,
    dtype,
    is_neox,
    rotary_dim,
    head_dim,
    num_heads,
    num_kv_heads,
    num_tokens,
    eps,
):
    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=16384,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)
    return qkv, positions, q_norm, k_norm, rope


def _run_fused(
    qkv,
    positions,
    q_norm,
    k_norm,
    rope,
    num_heads,
    num_kv_heads,
    head_dim,
    eps,
    is_neox,
    token_heads_per_warp,
):
    buf = qkv.clone()
    torch.ops._C.fused_qk_norm_rope(
        buf,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_norm.weight.data,
        k_norm.weight.data,
        rope.cos_sin_cache,
        is_neox,
        positions.view(-1),
        token_heads_per_warp,
    )
    return buf


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("token_heads_per_warp", [1, 2, 4, 8])
@torch.inference_mode()
def test_fused_qk_norm_rope_packing_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    rotary_ratio: float,
    head_dim: int,
    token_heads_per_warp: int,
):
    """Every token_heads_per_warp -- including the multi-head packing kernel --
    must match the fp32 Python reference.

    The auto heuristic always selects the 1-head baseline at small batches, so
    forcing token_heads_per_warp is the only way to reach
    ``fusedQKNormRopeKernelNTokenHeads``. That kernel corrupted every non-first
    head in a warp chunk under **partial** NeoX rope (rotary_dim < head_dim,
    rotary_lanes < warp size), because the RoPE section issued warp collectives
    (__syncwarp / __shfl_xor_sync) with the full mask while only rotary lanes were
    active -- UB that desynchronized the warp. Uses a small token count so bf16 /
    fp16 kernel-vs-fp32 rounding stays within the 1-head test's tolerance (the max
    abs error otherwise grows with the element count, independent of this bug).
    """
    if token_heads_per_warp > 1 and not current_platform.has_device_capability(80):
        pytest.skip("token_heads_per_warp > 1 requires SM80+")
    rotary_dim = int(head_dim * rotary_ratio)
    if rotary_dim % 2 != 0:
        pytest.skip("rotary_dim must be even")

    torch.set_default_device(device)
    set_random_seed(13)
    num_heads, num_kv_heads, num_tokens, eps = 16, 4, 4, 1e-6

    qkv, positions, q_norm, k_norm, rope = _build_qk_norm_rope_inputs(
        device,
        dtype,
        is_neox,
        rotary_dim,
        head_dim,
        num_heads,
        num_kv_heads,
        num_tokens,
        eps,
    )

    ref_result = _apply_qk_norm_rope(
        qkv=qkv,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )
    fused = _run_fused(
        qkv,
        positions,
        q_norm,
        k_norm,
        rope,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        is_neox,
        token_heads_per_warp,
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)
    torch.testing.assert_close(fused, ref_result, atol=ATOL, rtol=RTOL)

    # V is passed through untouched (no norm, no rope).
    v_off = (num_heads + num_kv_heads) * head_dim
    torch.testing.assert_close(fused[:, v_off:], qkv[:, v_off:], atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
# num_tokens straddling the SM90 auto-select thresholds (U = num_tokens*(Hq+Hk)):
# 512 -> pack4, 2048 -> pack8, so auto (-1) is checked at real dispatch boundaries.
@pytest.mark.parametrize("num_tokens", [512, 2048])
@pytest.mark.parametrize("token_heads_per_warp", [-1, 2, 4, 8])
@torch.inference_mode()
def test_fused_qk_norm_rope_packing_invariant(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    rotary_ratio: float,
    head_dim: int,
    num_tokens: int,
    token_heads_per_warp: int,
):
    """token_heads_per_warp (including auto=-1) must not change the result vs the
    1-head baseline -- it only remaps (token, head) units to warps.

    This is the tight, fp-noise-free regression for the partial-NeoX warp-desync
    bug: pre-fix the packed heads were off by ~20 (NaN at 8); the packed output
    must be *bitwise* identical to the 1-head kernel. Batch sizes straddle the
    SM90 auto-select thresholds so the auto path is validated at its real
    pack1/pack4/pack8 boundaries and at realistic prefill lengths.
    """
    if not current_platform.has_device_capability(80):
        pytest.skip("packing kernel requires SM80+")
    rotary_dim = int(head_dim * rotary_ratio)
    if rotary_dim % 2 != 0:
        pytest.skip("rotary_dim must be even")

    torch.set_default_device(device)
    set_random_seed(13)
    num_heads, num_kv_heads, eps = 16, 4, 1e-6

    qkv, positions, q_norm, k_norm, rope = _build_qk_norm_rope_inputs(
        device,
        dtype,
        is_neox,
        rotary_dim,
        head_dim,
        num_heads,
        num_kv_heads,
        num_tokens,
        eps,
    )

    baseline = _run_fused(
        qkv,
        positions,
        q_norm,
        k_norm,
        rope,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        is_neox,
        1,
    )
    packed = _run_fused(
        qkv,
        positions,
        q_norm,
        k_norm,
        rope,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        is_neox,
        token_heads_per_warp,
    )
    torch.testing.assert_close(packed, baseline, atol=0, rtol=0)
