# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill(chunk) == decode equivalence for the GDN ReplaySSM kernel.

The GDN analog of the Mamba2 prefill-teacher: the chunked prefill kernel
(``chunk_gated_delta_rule``, the path the model runs at prefill) and
step-by-step decode must produce the same per-position outputs and final state
over a sequence. We build q/k/v/g/beta with ``fused_post_conv_prep`` (the same
shared prep the model's prefill uses, so the teacher's inputs match what the
decode kernels compute internally from ``mixed_qkv``), then feed:

  * the chunked prefill kernel (the teacher),
  * the baseline step decode (``..._packed_decode``),
  * the ReplaySSM step decode (ring buffer, ``..._replayssm``),

and check the step decoders reproduce the teacher's per-position outputs and
final state. The chunked prefill kernel only supports bf16 *activations*
(q/k/v), so it is the (trusted) reference run with bf16 activations -- there is
no separate fp32 ground truth here (the fp32-activation decode path is covered
by the GDN standard-decode test). The recurrent *state* precision is still
swept: fp32 (the production config) and bf16, both with bf16 activations.

Prefill (a chunked scan) and decode (a step recurrence) are different code
paths, so they differ numerically; we use chunked-scan tolerances (the same
regime as the Mamba2 prefill-teacher).

``seqlen`` is a multiple of the buffer length, so the final step flushes and
ReplaySSM's stored checkpoint is the full final state.
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_replayssm,
)
from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.utils.torch_utils import set_random_seed


def _output_tolerances(act_dtype: torch.dtype) -> tuple[float, float]:
    # Chunked-scan-vs-step regime (the chunked prefill, not the decode, sets
    # these), same as the Mamba2 prefill-teacher. Keyed off the activation dtype.
    if act_dtype == torch.float32:
        return 1e-2, 3e-2
    return 6e-2, 1e-1


def _run_gdn_teacher_equivalence(
    *,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    batch: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    seqlen: int,
    max_cache_len: int,
    seed: int = 0,
) -> None:
    assert seqlen % max_cache_len == 0, "final step must be a flush"
    device = "cuda"
    H, HV, K, V = num_q_heads, num_v_heads, head_k_dim, head_v_dim
    o_rtol, o_atol = _output_tolerances(act_dtype)
    set_random_seed(seed)
    scale = K**-0.5
    qkv_dim = 2 * (H * K) + HV * V
    L = batch * seqlen

    # Step decoders use slot-indexed state (slot 0 is the null block).
    num_state_slots = batch + 1
    ssm_state_indices = torch.arange(
        1, batch + 1, device=device, dtype=torch.int32)

    A_log = torch.randn(HV, device=device, dtype=act_dtype)
    dt_bias = torch.randn(HV, device=device, dtype=act_dtype)
    mixed_qkv_seq = torch.randn(
        batch, seqlen, qkv_dim, device=device, dtype=act_dtype)
    a_seq = torch.randn(batch, seqlen, HV, device=device, dtype=act_dtype)
    b_seq = torch.randn(batch, seqlen, HV, device=device, dtype=act_dtype)

    def run_teacher(qkv_dtype: torch.dtype, st_dtype: torch.dtype):
        # Same post-conv prep as the model's prefill (GQA + l2norm), so the
        # teacher's q/k/v/g/beta match what the decode kernels derive from
        # mixed_qkv. apply_l2norm=True here -> l2norm off in the chunk kernel.
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv_seq.reshape(L, qkv_dim).to(qkv_dtype),
            a=a_seq.reshape(L, HV).to(qkv_dtype),
            b=b_seq.reshape(L, HV).to(qkv_dtype),
            A_log=A_log.to(qkv_dtype), dt_bias=dt_bias.to(qkv_dtype),
            num_k_heads=H, head_k_dim=K, head_v_dim=V,
            apply_l2norm=True, output_g_exp=False)
        q = q.reshape(batch, seqlen, *q.shape[1:])
        k = k.reshape(batch, seqlen, *k.shape[1:])
        v = v.reshape(batch, seqlen, *v.shape[1:])
        g = g.reshape(batch, seqlen, *g.shape[1:])
        beta = beta.reshape(batch, seqlen, *beta.shape[1:])
        state0 = torch.zeros(batch, HV, V, K, device=device, dtype=st_dtype)
        out, final_state = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=state0,
            output_final_state=True, cu_seqlens=None,
            use_qk_l2norm_in_kernel=False)
        return out, final_state  # (B, T, HV, V), (B, HV, V, K)

    # The chunked prefill kernel only supports bf16 activations, so it is the
    # (trusted) reference at the test (bf16-activation) precision; no separate
    # fp32 ground truth (fp32-activation decode is covered by the GDN
    # standard-decode test). The recurrent state precision is still swept.
    y_teacher, state_teacher = run_teacher(act_dtype, state_dtype)

    # Step decoders: baseline packed decode and ReplaySSM (ring buffer).
    state_base = torch.zeros(
        num_state_slots, HV, V, K, device=device, dtype=state_dtype)
    state_dec = torch.zeros(
        num_state_slots, HV, V, K, device=device, dtype=state_dtype)
    d_cache = torch.zeros(
        num_state_slots, HV, max_cache_len, V, device=device, dtype=act_dtype)
    k_cache = torch.zeros(
        num_state_slots, H, max_cache_len, K, device=device, dtype=act_dtype)
    g_cache = torch.zeros(
        num_state_slots, HV, max_cache_len, device=device, dtype=torch.float32)

    y_base = torch.empty(batch, seqlen, HV, V, device=device, dtype=act_dtype)
    y_dec = torch.empty(batch, seqlen, HV, V, device=device, dtype=act_dtype)
    for t in range(seqlen):
        mixed_qkv = mixed_qkv_seq[:, t]
        a = a_seq[:, t]
        b = b_seq[:, t]
        write_pos = torch.full(
            (batch,), t % max_cache_len, device=device, dtype=torch.int32)

        out_b = torch.empty(batch, 1, HV, V, device=device, dtype=act_dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_base, out=out_b,
            ssm_state_indices=ssm_state_indices, use_qk_l2norm_in_kernel=True)
        y_base[:, t] = out_b[:, 0]

        out_d = torch.empty(batch, 1, HV, V, device=device, dtype=act_dtype)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_dec, d_cache=d_cache,
            k_cache=k_cache, g_cache=g_cache, out=out_d,
            ssm_state_indices=ssm_state_indices, write_pos=write_pos,
            use_qk_l2norm_in_kernel=True)
        y_dec[:, t] = out_d[:, 0]

    # Baseline and ReplaySSM step decode both reproduce the chunked prefill
    # teacher's per-position outputs.
    torch.testing.assert_close(
        y_base.float(), y_teacher.float(), rtol=o_rtol, atol=o_atol)
    torch.testing.assert_close(
        y_dec.float(), y_teacher.float(), rtol=o_rtol, atol=o_atol)
    # Final state (the last step is a flush, so state_dec is the full state).
    # The step decoders' state is slot-indexed; gather active rows to match the
    # teacher's dense per-sequence state.
    torch.testing.assert_close(
        state_base[ssm_state_indices].float(), state_teacher.float(),
        rtol=o_rtol, atol=o_atol)
    torch.testing.assert_close(
        state_dec[ssm_state_indices].float(), state_teacher.float(),
        rtol=o_rtol, atol=o_atol)


# The chunked prefill teacher is bf16-activation-only, so the activation dtype is
# bf16; the recurrent-state precision is still swept: fp32 (production), bf16, and
# fp16 (a finer-mantissa state than bf16 at the same 2 bytes). Fully-fp16
# activations are out of scope here (the teacher cannot run them).
_PRECISIONS = [
    pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
    pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
    pytest.param((torch.float16, torch.bfloat16), id="sfp16_a16"),
]
_GEOMETRIES = [
    # (num_q_heads, num_v_heads, head_k_dim, head_v_dim)
    pytest.param((2, 4, 64, 64), id="small"),
    pytest.param((16, 32, 128, 128), id="qwen4b"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("max_cache_len", [4, 16])
def test_replayssm_teacher_decode_equivalence_gdn(
    precision: tuple[torch.dtype, torch.dtype],
    geometry: tuple[int, int, int, int],
    max_cache_len: int,
):
    state_dtype, act_dtype = precision
    num_q_heads, num_v_heads, head_k_dim, head_v_dim = geometry
    _run_gdn_teacher_equivalence(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        batch=4,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        seqlen=16,
        max_cache_len=max_cache_len,
    )
