# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard (autoregressive) decode correctness for the GDN ReplaySSM kernel.

GDN ReplaySSM caches the per-step delta-rule inputs (the corrected value ``u``,
the key, and the gate) in a small ring buffer and reconstructs the recurrent
state on the fly, writing it back to HBM only when the buffer flushes. This
checks the cached decode kernel ``fused_recurrent_gated_delta_rule_replayssm``
against the upstream baseline ``fused_recurrent_gated_delta_rule_packed_decode``
over a multi-step decode, across the flush boundaries and with sparse
(continuous-batching) state allocation including padding rows.

GDN uses only the state-reconstruction route (it reads the state at both k and
q, so there is no output-only variant). State precision and activation/buffer
precision are swept independently; Qwen3.5 defaults to fp32 SSM state, though
bf16 state is also supported. ``g_cache`` is always fp32.
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_replayssm,
)
from vllm.utils.torch_utils import set_random_seed


def _output_tolerances(act_dtype: torch.dtype) -> tuple[float, float]:
    # Anchored on the baseline packed-decode kernel; same regime as the existing
    # GDN test. The output has no fp32 drift (tight 1e-4); keyed off act dtype.
    if act_dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 2e-2


def _state_tolerances(act_dtype: torch.dtype) -> tuple[float, float]:
    # The reconstructed checkpoint state vs the baseline's sequential state
    # differs a bit more than the output at fp32 (~8e-4, an fp32 reconstruction
    # vs recurrence difference, not a precision bug) -- same looser regime the
    # existing GDN test uses for state.
    if act_dtype == torch.float32:
        return 1e-3, 2e-3
    return 1e-2, 2e-2


def _run_gdn_standard_decode(
    *,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    batch: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    max_cache_len: int,
    num_steps: int,
    n_pad: int,
    strided: bool = False,
    seed: int = 0,
) -> None:
    """Drive ``num_steps`` decode steps; check the ReplaySSM kernel matches the
    baseline packed decode each step (and the checkpoint state at flushes).
    State follows ``state_dtype``; activations/caches follow ``act_dtype``."""
    device = "cuda"
    o_rtol, o_atol = _output_tolerances(act_dtype)
    s_rtol, s_atol = _state_tolerances(act_dtype)
    set_random_seed(seed)
    scale = head_k_dim**-0.5
    qkv_dim = 2 * (num_q_heads * head_k_dim) + num_v_heads * head_v_dim
    padded_batch = batch + n_pad
    num_state_slots = batch + n_pad + 1  # +1 reserves a null block at slot 0

    A_log = torch.randn(num_v_heads, device=device, dtype=act_dtype)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=act_dtype)

    # Sparse state allocation: active rows map to a random permutation of slots
    # 1.., padding rows map to PAD_SLOT_ID=-1.
    state_indices = (
        torch.randperm(num_state_slots - 1, device=device)[:batch] + 1
    ).to(torch.int32)
    ssm_state_indices = torch.cat([
        state_indices,
        torch.full((n_pad,), -1, dtype=torch.int32, device=device),
    ])
    unused = torch.ones(num_state_slots, dtype=torch.bool, device=device)
    unused[state_indices] = False

    state0 = torch.randn(
        num_state_slots, num_v_heads, head_v_dim, head_k_dim,
        device=device, dtype=state_dtype)
    state_packed = state0.clone()
    state_cached = state0.clone()
    state_before = state0.clone()

    d_cache = torch.zeros(
        num_state_slots, num_v_heads, max_cache_len, head_v_dim,
        device=device, dtype=act_dtype)
    k_cache = torch.zeros(
        num_state_slots, num_q_heads, max_cache_len, head_k_dim,
        device=device, dtype=act_dtype)
    g_cache = torch.zeros(
        num_state_slots, num_v_heads, max_cache_len, device=device,
        dtype=torch.float32)

    for step in range(num_steps):
        if strided:
            proj = torch.randn(
                padded_batch, qkv_dim + 64, device=device, dtype=act_dtype)
            mixed_qkv = proj[:, :qkv_dim]
        else:
            mixed_qkv = torch.randn(
                padded_batch, qkv_dim, device=device, dtype=act_dtype)
        a = torch.randn(padded_batch, num_v_heads, device=device, dtype=act_dtype)
        b = torch.randn(padded_batch, num_v_heads, device=device, dtype=act_dtype)
        write_pos = torch.full(
            (padded_batch,), step % max_cache_len, device=device,
            dtype=torch.int32)

        out_packed = torch.empty(
            padded_batch, 1, num_v_heads, head_v_dim, device=device,
            dtype=act_dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_packed, out=out_packed,
            ssm_state_indices=ssm_state_indices, use_qk_l2norm_in_kernel=True)

        out_cached = torch.empty(
            padded_batch, 1, num_v_heads, head_v_dim, device=device,
            dtype=act_dtype)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_cached, d_cache=d_cache,
            k_cache=k_cache, g_cache=g_cache, out=out_cached,
            ssm_state_indices=ssm_state_indices, write_pos=write_pos,
            use_qk_l2norm_in_kernel=True)

        torch.testing.assert_close(
            out_cached[:batch], out_packed[:batch], rtol=o_rtol, atol=o_atol)
        if step % max_cache_len == max_cache_len - 1:
            torch.testing.assert_close(
                state_cached[state_indices], state_packed[state_indices],
                rtol=s_rtol, atol=s_atol)

    # Padding / unused state slots are never written.
    assert torch.equal(state_cached[unused], state_before[unused])


# State/activation precisions. fp32 state is the default; bf16/fp16 are the
# reduced-footprint configs. fp16 appears as an activation dtype (sfp16_afp16,
# s32_afp16) and as a state dtype under bf16 activations (sfp16_a16): fp16 has a
# finer mantissa than bf16 at the same 2 bytes, so it is a more accurate state at
# no extra footprint. We skip fp16 state under fp32 activations.
_PRECISIONS = [
    pytest.param((torch.float32, torch.float32), id="s32_a32"),
    pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
    pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
    pytest.param((torch.float32, torch.float16), id="s32_afp16"),
    pytest.param((torch.float16, torch.float16), id="sfp16_afp16"),
    pytest.param((torch.float16, torch.bfloat16), id="sfp16_a16"),
]
_GEOMETRIES = [
    # (num_q_heads, num_v_heads, head_k_dim, head_v_dim)
    pytest.param((2, 4, 64, 64), id="small"),
    pytest.param((16, 32, 128, 128), id="qwen4b"),
    pytest.param((16, 64, 128, 128), id="qwen122b"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("max_cache_len", [4, 16])
@pytest.mark.parametrize("strided", [False, True])
def test_replayssm_standard_decode_gdn_matches_packed(
    precision: tuple[torch.dtype, torch.dtype],
    geometry: tuple[int, int, int, int],
    max_cache_len: int,
    strided: bool,
):
    state_dtype, act_dtype = precision
    num_q_heads, num_v_heads, head_k_dim, head_v_dim = geometry
    _run_gdn_standard_decode(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        batch=4,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        max_cache_len=max_cache_len,
        num_steps=2 * max_cache_len + 1,
        n_pad=2,
        strided=strided,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
def test_replayssm_standard_decode_gdn_per_row_write_pos(
    precision: tuple[torch.dtype, torch.dtype],
):
    # Each row is built up to a different prefix length, so when the batch runs
    # together the rows sit at different ring positions. This validates that the
    # kernel reads a per-row write_pos (independent cache cursors per sequence --
    # the continuous-batching case the main test's uniform write_pos misses).
    state_dtype, act_dtype = precision
    device = "cuda"
    o_rtol, o_atol = _output_tolerances(act_dtype)
    set_random_seed(1)

    batch = 4
    num_q_heads, num_v_heads, head_k_dim, head_v_dim = 2, 4, 64, 64
    max_cache_len = 4
    num_state_slots = 7
    scale = head_k_dim**-0.5
    qkv_dim = 2 * (num_q_heads * head_k_dim) + num_v_heads * head_v_dim

    A_log = torch.randn(num_v_heads, device=device, dtype=act_dtype)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=act_dtype)
    ssm_state_indices = torch.tensor([4, 2, 6, 1], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

    state0 = torch.randn(
        num_state_slots, num_v_heads, head_v_dim, head_k_dim,
        device=device, dtype=state_dtype)
    state_packed = state0.clone()
    state_cached = state0.clone()
    d_cache = torch.zeros(
        num_state_slots, num_v_heads, max_cache_len, head_v_dim,
        device=device, dtype=act_dtype)
    k_cache = torch.zeros(
        num_state_slots, num_q_heads, max_cache_len, head_k_dim,
        device=device, dtype=act_dtype)
    g_cache = torch.zeros(
        num_state_slots, num_v_heads, max_cache_len, device=device,
        dtype=torch.float32)

    def decode_step(mixed_qkv, a, b, idx, write_pos):
        out_p = torch.empty(
            mixed_qkv.shape[0], 1, num_v_heads, head_v_dim, device=device,
            dtype=act_dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_packed, out=out_p,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        out_c = torch.empty_like(out_p)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=state_cached, d_cache=d_cache,
            k_cache=k_cache, g_cache=g_cache, out=out_c,
            ssm_state_indices=idx, write_pos=write_pos,
            use_qk_l2norm_in_kernel=True)
        return out_p, out_c

    # Build each row up independently to its prefix length.
    for row, prefix_len in enumerate(prefix_lens.tolist()):
        idx = ssm_state_indices[row:row + 1]
        for s in range(prefix_len):
            mixed_qkv = torch.randn(1, qkv_dim, device=device, dtype=act_dtype)
            a = torch.randn(1, num_v_heads, device=device, dtype=act_dtype)
            b = torch.randn(1, num_v_heads, device=device, dtype=act_dtype)
            decode_step(
                mixed_qkv, a, b, idx,
                torch.tensor([s], device=device, dtype=torch.int32))

    # Run the whole batch with per-row write_pos = prefix_lens.
    mixed_qkv = torch.randn(batch, qkv_dim, device=device, dtype=act_dtype)
    a = torch.randn(batch, num_v_heads, device=device, dtype=act_dtype)
    b = torch.randn(batch, num_v_heads, device=device, dtype=act_dtype)
    out_packed, out_cached = decode_step(
        mixed_qkv, a, b, ssm_state_indices, prefix_lens)
    torch.testing.assert_close(out_cached, out_packed, rtol=o_rtol, atol=o_atol)
