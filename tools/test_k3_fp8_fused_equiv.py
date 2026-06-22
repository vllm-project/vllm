# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 numerical equivalence: fused FP8 stage1 vs bf16-workspace baseline.

Verifies that `fused_mla_tq_decode_stage1(key_fp8=True)` produces the same
output as the legacy path of (cache_fp8 -> bf16 workspace) +
`decode_attention_fwd_grouped`, modulo bf16 ULP-level rounding.
"""

from __future__ import annotations

import math

import torch

from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
    decode_attention_fwd_grouped,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_tq_decode_stage1,
)

_FP8 = torch.float8_e4m3fn
_BF16 = torch.bfloat16


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    L, R, H_q = 512, 64, 128
    B, ctx_len, page_size = 4, 4096, 64
    num_kv_splits = 4
    num_pages_per_seq = ctx_len // page_size
    n_active = B * num_pages_per_seq
    sm_scale = 1.0 / math.sqrt(L + R)

    # Build cache.
    kv_c_f = (
        torch.randn(n_active, page_size, L, device=device, dtype=torch.float32)
        .clamp(-448.0, 448.0)
        .to(_FP8)
    )
    k_pe = torch.randn(n_active, page_size, R, device=device, dtype=_BF16)
    cache = torch.empty(n_active, page_size, L + 2 * R, device=device, dtype=torch.uint8)
    cache[..., :L] = kv_c_f.view(torch.uint8)
    cache[..., L:] = k_pe.view(torch.uint8).reshape(n_active, page_size, 2 * R)
    import os as _os
    k_scale_t = torch.tensor(
        float(_os.environ.get("K_SCALE", "1.0")),
        device=device, dtype=torch.float32,
    )
    k_scale = float(k_scale_t.item())
    print(f"k_scale = {k_scale}")

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), ctx_len, device=device, dtype=torch.int32)
    q = torch.randn(B, H_q, L + R, device=device, dtype=_BF16)

    # Baseline: dequant fp8 -> bf16 workspace -> decode_attention_fwd_grouped.
    workspace = torch.empty(n_active, page_size, L + R, device=device, dtype=_BF16)
    workspace[..., :L] = (kv_c_f.to(torch.float32) * k_scale).to(_BF16)
    workspace[..., L:] = k_pe
    o_ref = torch.empty(B, H_q, L, device=device, dtype=_BF16)
    lse_ref = torch.empty(B, H_q, device=device, dtype=_BF16)
    attn_logits_ref = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    one_scale = torch.ones(1, device=device, dtype=torch.float32)
    decode_attention_fwd_grouped(
        q,
        workspace.unsqueeze(2),
        workspace.unsqueeze(2)[..., :L],
        o_ref,
        lse_ref,
        req_to_tokens,
        b_seqlen,
        attn_logits_ref,
        num_kv_splits,
        sm_scale,
        page_size,
        k_scale=one_scale,
        v_scale=one_scale,
        is_mla=True,
    )

    # Fused: stage1 reads fp8 directly; k_scale applied to y_hat before cast.
    centroids_unused = torch.empty(0, device=device, dtype=_BF16)
    o_fused = torch.empty(B, H_q, L, device=device, dtype=_BF16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=_BF16)
    v_holder = torch.empty((1, L), device=device, dtype=_BF16)
    attn_logits_f = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q,
        cache,
        centroids_unused,
        attn_logits_f,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=0,
        mse_bytes=0,
        kv_c_bytes=L,
        norm_correction=False,
        kpe_fp8=False,
        key_fp8=True,
        k_scale=k_scale_t,
        num_kv_splits=num_kv_splits,
    )
    _decode_softmax_reducev_fwd(
        attn_logits_f, q, o_fused, lse_fused, v_holder, b_seqlen, num_kv_splits,
    )

    diff = (o_fused.to(torch.float32) - o_ref.to(torch.float32)).abs()
    ref_rms = o_ref.to(torch.float32).pow(2).mean().sqrt().item()
    diff_rms = diff.pow(2).mean().sqrt().item()
    print(f"max abs diff   = {diff.max().item():.6e}")
    print(f"diff RMS       = {diff_rms:.6e}")
    print(f"ref RMS        = {ref_rms:.6e}")
    print(f"rel RMS error  = {diff_rms / max(ref_rms, 1e-9):.6e}")

    # bf16 ULP tolerance: O(1e-2) absolute is fine for this scale.
    assert diff.max().item() < 0.05, "fused FP8 vs baseline diverged > 5e-2"
    print("OK: K3 fused FP8 matches bf16-workspace baseline within bf16 tolerance.")


if __name__ == "__main__":
    main()
