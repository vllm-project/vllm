# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two-kernel fused decode qkv (residual-add RMSNorm, then qkv GEMV with
per-head q/k norm, RoPE and KV-cache write) must reproduce the kernels it
replaces, Flint's ``fused_add_rms_norm`` / ``rms_norm`` -> Flint GEMV ->
``qk_norm_rope_cache``, up to reduction order (rare 1-ulp differences)."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip("Flint fused qkv targets SM100", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # noqa: E402
    swizzle_blockscale,
)
from vllm.models.flint.ops import norm, nvfp4_gemv  # noqa: E402
from vllm.models.flint.ops.qk_norm_rope import qk_norm_rope_cache  # noqa: E402
from vllm.models.flint.ops.qkv_fused import flint_qkv_fused  # noqa: E402

HQ, HKV, D, K = 32, 4, 128, 2048
N = (HQ + 2 * HKV) * D
NUM_BLOCKS, BLOCK_SIZE = 64, 16


def _weights(bf16w: bool, g: torch.Generator):
    if bf16w:
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
        return w, None, None
    w = torch.randint(0, 256, (N, K // 2), device="cuda", generator=g)
    scales = torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.25
    scales = swizzle_blockscale(scales.to(torch.float8_e4m3fn))
    return w.to(torch.uint8), scales, torch.tensor([0.0037], device="cuda")


@pytest.mark.parametrize("m", [1, 2, 5, 8])
@pytest.mark.parametrize("bf16w", [False, True], ids=["nvfp4", "bf16"])
@pytest.mark.parametrize(
    "has_residual,rope",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["res-rope", "res-norope", "first-layer", "first-norope"],
)
@pytest.mark.parametrize("short_slots", [False, True], ids=["pad", "short"])
@torch.inference_mode()
def test_matches_unfused_kernels(
    m: int, bf16w: bool, has_residual: bool, rope: bool, short_slots: bool
):
    g = torch.Generator(device="cuda").manual_seed(m)
    w, ws, gs = _weights(bf16w, g)
    x = (torch.randn(m, K, device="cuda", generator=g) * 3).bfloat16()
    res = None
    if has_residual:
        res = (torch.randn(m, K, device="cuda", generator=g) * 3).bfloat16()
    norm_w = torch.empty(K, device="cuda").normal_(1, 0.2, generator=g).bfloat16()
    q_w = torch.empty(D, device="cuda").normal_(1, 0.2, generator=g).bfloat16()
    k_w = torch.empty(D, device="cuda").normal_(1, 0.2, generator=g).bfloat16()
    cos_sin = None
    if rope:
        inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
        freqs = torch.outer(torch.arange(262144, device="cuda").float(), inv)
        # FP32 cos | sin table, as get_rope(..., dtype=torch.float32) builds.
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    # Up to the end of the RoPE table.
    positions = torch.randint(262144 - 64, 262144, (m,), device="cuda", generator=g)
    slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device="cuda", generator=g)[:m]
    # Padding tokens are skipped: a -1 slot, or beyond a shorter slot mapping.
    if m > 1 and short_slots:
        slots = slots[:-1]
    elif m > 1:
        slots[-1] = -1
    kv = torch.zeros(
        NUM_BLOCKS, HKV, BLOCK_SIZE, 2 * D, device="cuda", dtype=torch.bfloat16
    )
    ref_kv = kv.clone()

    if res is not None:
        xr, ref_res = norm.fused_add_rms_norm(x.clone(), res.clone(), norm_w, 1e-6)
    else:
        xr = norm.rms_norm(x, norm_w, 1e-6)
    if bf16w:
        qkv = (xr.float() @ w.float().t()).bfloat16()
    else:
        qkv = nvfp4_gemv.nvfp4_gemv(xr, w, ws, gs)
    ref_k, ref_v = ref_kv.transpose(1, 2).split(D, dim=-1)
    ref_q = qk_norm_rope_cache(
        qkv, positions, q_w, k_w, 1e-6, cos_sin, ref_k, ref_v, slots, HQ, HKV, D
    )

    k_cache, v_cache = kv.transpose(1, 2).split(D, dim=-1)
    q, res_out = flint_qkv_fused(
        x, res, norm_w, 1e-6, w, ws, gs, q_w, k_w, 1e-6, cos_sin, positions,
        k_cache, v_cache, slots, HQ, HKV,
    )  # fmt: skip

    if has_residual:
        torch.testing.assert_close(res_out, ref_res, atol=0, rtol=0)
    else:
        assert res_out is x
    # Same rounding points, but the norms (and the BF16 GEMV's K split) sum in
    # a different fp32 order: a 1-ulp difference in a normalized activation
    # shifts every qkv output of its token slightly.
    assert (q == ref_q).float().mean() > 0.9
    assert (kv == ref_kv).float().mean() > 0.9
    torch.testing.assert_close(q, ref_q, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(kv, ref_kv, atol=1e-2, rtol=2e-2)
