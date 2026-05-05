# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric accuracy parity: tokenspeed_mla_prefill vs trtllm_ragged_attention_deepseek.

Two cases mirror what the vLLM MLA prefill backend does in production:
- `test_prefill_no_context`: causal Q==KV ragged batch (run_prefill_new_tokens).
- `test_prefill_with_context`: non-causal Q ragged + KV ragged with
  per-request kv_len > q_len (run_prefill_context_chunk).
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="tokenspeed_mla / TRT-LLM ragged require Blackwell (SM100+).",
        allow_module_level=True,
    )

try:
    from flashinfer.prefill import trtllm_ragged_attention_deepseek
except ImportError:
    pytest.skip(reason="flashinfer not installed", allow_module_level=True)

try:
    from tokenspeed_mla import tokenspeed_mla_prefill, warmup_compile_prefill
except ImportError:
    pytest.skip(reason="tokenspeed_mla not installed", allow_module_level=True)


FLASHINFER_WORKSPACE_BUFFER_SIZE = 384 * 1024 * 1024


# Deepseek R1 dimensions — both kernels are shape-specialized for these.
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
SCALE = QK_HEAD_DIM**-0.5


def _make_q_kv(
    seq_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
):
    """Build ragged Q (qk_head_dim) and K (qk_head_dim) / V (v_head_dim)."""
    total_q = sum(seq_lens)
    total_kv = sum(kv_lens)

    q = torch.randn(total_q, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16).to(dtype)
    k = torch.randn(total_kv, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16).to(dtype)
    v = torch.randn(total_kv, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16).to(dtype)
    return q, k, v


def _cumsum_int32(lens: list[int]) -> torch.Tensor:
    out = torch.zeros(len(lens) + 1, dtype=torch.int32)
    out[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("bs", [1, 4, 16])
@pytest.mark.parametrize("max_q_len", [64, 256, 1024])
def test_prefill_no_context(dtype: torch.dtype, bs: int, max_q_len: int):
    """Causal Q==KV ragged: matches the run_prefill_new_tokens code path."""
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    if dtype == torch.float8_e4m3fn:
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=QK_HEAD_DIM,
            d_v=V_HEAD_DIM,
            enable_pdl=False,
        )

    seq_lens = [int(torch.randint(2, max_q_len + 1, (1,)).item()) for _ in range(bs)]
    seq_lens[-1] = max_q_len  # pin the last so max_q_len is hit

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cum_seq_lens = _cumsum_int32(seq_lens)

    q, k, v = _make_q_kv(seq_lens, seq_lens, dtype)

    # --- TRT-LLM reference ---
    workspace = torch.zeros(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
    out_ref = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=torch.bfloat16)
    ref_ret = trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=workspace,
        seq_lens=seq_lens_tensor,
        max_q_len=max_q_len,
        max_kv_len=max_q_len,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=bs,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        enable_pdl=False,
        is_causal=True,
        return_lse=False,
        out=out_ref,
    )
    out_ref = ref_ret if not isinstance(ref_ret, tuple) else ref_ret[0]

    # --- TokenSpeed candidate ---
    out_ts = tokenspeed_mla_prefill(
        query=q,
        key=k,
        value=v,
        seq_lens=seq_lens_tensor,
        cum_seq_lens=cum_seq_lens,
        max_seq_len=max_q_len,
        batch_size=bs,
        softmax_scale=SCALE,
        is_causal=True,
        return_lse=False,
        enable_pdl=False,
    )
    if isinstance(out_ts, tuple):
        out_ts = out_ts[0]

    out_ref_f = out_ref.to(torch.float32)
    out_ts_f = out_ts.to(torch.float32)
    assert out_ref_f.shape == out_ts_f.shape, (
        f"shape mismatch: trtllm={tuple(out_ref_f.shape)} "
        f"tokenspeed={tuple(out_ts_f.shape)}"
    )

    if dtype == torch.float8_e4m3fn:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(out_ts_f, out_ref_f, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("bs", [1, 4, 16])
def test_prefill_with_context(dtype: torch.dtype, bs: int):
    """Non-causal Q ragged + KV ragged: run_prefill_context_chunk path.

    Per-request KV length is independent of (and >=) Q length, mimicking the
    chunked-context call site where KV is the cache chunk and Q is the new tokens.
    """
    torch.set_default_device("cuda")
    torch.manual_seed(1)

    if dtype == torch.float8_e4m3fn:
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=QK_HEAD_DIM,
            d_v=V_HEAD_DIM,
            enable_pdl=False,
        )

    q_lens = [int(torch.randint(16, 257, (1,)).item()) for _ in range(bs)]
    kv_lens = [q_lens[i] + int(torch.randint(0, 1025, (1,)).item()) for i in range(bs)]

    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    cum_q = _cumsum_int32(q_lens)
    cum_kv = _cumsum_int32(kv_lens)
    max_q_len = max(q_lens)
    max_kv_len = max(kv_lens)

    q, k, v = _make_q_kv(q_lens, kv_lens, dtype)

    # --- TRT-LLM reference ---
    workspace = torch.zeros(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
    out_ref = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=torch.bfloat16)
    ref_ret = trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=workspace,
        seq_lens=kv_lens_t,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=bs,
        window_left=-1,
        cum_seq_lens_q=cum_q,
        cum_seq_lens_kv=cum_kv,
        enable_pdl=False,
        is_causal=False,
        return_lse=True,
        out=out_ref,
    )
    out_ref, lse_ref = ref_ret[0], ref_ret[1]

    # --- TokenSpeed candidate ---
    ts_ret = tokenspeed_mla_prefill(
        query=q,
        key=k,
        value=v,
        seq_lens=kv_lens_t,
        cum_seq_lens=cum_kv,
        max_seq_len=max_kv_len,
        batch_size=bs,
        softmax_scale=SCALE,
        is_causal=False,
        return_lse=True,
        cum_seq_lens_q=cum_q,
        max_seq_len_q=max_q_len,
        enable_pdl=False,
    )
    out_ts, lse_ts = ts_ret[0], ts_ret[1]

    if dtype == torch.float8_e4m3fn:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(
        out_ts.to(torch.float32),
        out_ref.to(torch.float32),
        atol=atol,
        rtol=rtol,
    )

    # LSE: trtllm returns (q_len, num_heads). Tokenspeed convention should
    # match shape-by-shape — if it doesn't, the LSE transpose contract that
    # merge_attn_states relies on is broken and this assert surfaces it.
    assert lse_ref.shape == lse_ts.shape, (
        f"LSE shape mismatch: trtllm={tuple(lse_ref.shape)} "
        f"tokenspeed={tuple(lse_ts.shape)}"
    )
    # Log-base normalization: trtllm returns LSE in log2, tokenspeed and
    # vLLM's merge_attn_states (triton_merge_attn_states.py:138) both use
    # natural-log. Convert trtllm's log2 LSE to natural log before
    # comparison, otherwise we'd be comparing different bases (factor ln 2).
    import math

    torch.testing.assert_close(
        lse_ts.to(torch.float32),
        lse_ref.to(torch.float32) * math.log(2),
        atol=5e-3,
        rtol=5e-3,
    )
