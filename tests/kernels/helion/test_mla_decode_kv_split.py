# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.kernels.helion.ops.mla_decode_fp8_per_token import (
    decode_grouped_att_m_fwd_baseline,
    generate_mla_decode_kv_split_inputs,
    mla_decode_kv_split,
)

LATENT_DIM = 512


def _clone_inputs(inputs):
    return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)


def _call_helion(inputs, attn_out):
    (
        q_absorbed,
        latent_kv,
        _,
        kv_dequant,
        sm_scale,
        b_seq_len,
        req_to_tokens,
        heads_per_block,
        num_kv_splits,
        page_size,
        latent_dim,
        rope_dim,
        logit_cap,
    ) = inputs
    mla_decode_kv_split(
        q_absorbed,
        latent_kv,
        attn_out,
        kv_dequant,
        sm_scale,
        b_seq_len,
        req_to_tokens,
        heads_per_block,
        num_kv_splits,
        page_size,
        latent_dim,
        rope_dim,
        logit_cap,
    )


def _call_baseline(inputs, attn_out):
    (
        q_absorbed,
        latent_kv,
        _,
        kv_dequant,
        sm_scale,
        b_seq_len,
        req_to_tokens,
        heads_per_block,
        num_kv_splits,
        page_size,
        latent_dim,
        rope_dim,
        logit_cap,
    ) = inputs

    decode_grouped_att_m_fwd_baseline(
        q_absorbed=q_absorbed,
        latent_kv=latent_kv,
        attn_out=attn_out,
        kv_dequant=kv_dequant,
        sm_scale=sm_scale,
        B_seq_len=b_seq_len,
        Req_to_Tokens=req_to_tokens,
        HEADS_PER_BLOCK=heads_per_block,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        logit_cap=logit_cap,
    )

    return attn_out


@pytest.fixture(scope="module")
def test_inputs():
    all_inputs = generate_mla_decode_kv_split_inputs()
    assert len(all_inputs) == 1, "Expected exactly one input config from generator"
    return next(iter(all_inputs.values()))


@pytest.fixture(scope="module")
def outputs(test_inputs):
    inputs = _clone_inputs(test_inputs)

    helion_out = torch.empty_like(test_inputs[2])
    _call_helion(inputs, helion_out)

    baseline_out = torch.empty_like(test_inputs[2])  # allocate separately
    _call_baseline(inputs, baseline_out)  # pass it in

    return helion_out, baseline_out


class TestMLADecodeKVSplitCorrectness:
    def test_output_shape(self, test_inputs, outputs):
        expected = test_inputs[2].shape
        helion_out, baseline_out = outputs
        assert helion_out.shape == expected
        assert baseline_out.shape == expected

    def test_output_dtype(self, outputs):
        helion_out, baseline_out = outputs
        assert helion_out.dtype == torch.float32
        assert baseline_out.dtype == torch.float32

    def test_partial_attention_values(self, outputs):
        """Compare partial weighted-sum accumulator (first latent_dim slots)."""
        helion_out, baseline_out = outputs
        acc_h = helion_out[..., :LATENT_DIM]
        acc_b = baseline_out[..., :LATENT_DIM]
        max_abs_diff = (acc_h - acc_b).abs().max().item()
        mean_abs_diff = (acc_h - acc_b).abs().mean().item()
        assert torch.allclose(acc_h, acc_b, atol=1e-2, rtol=1e-2), (
            f"Partial acc mismatch: max_abs_diff={max_abs_diff:.6f}, "
            f"mean_abs_diff={mean_abs_diff:.6f}"
        )

    def test_logsum_exp_values(self, outputs):
        """Compare log-sum-exp stored in the last slot of each split."""
        helion_out, baseline_out = outputs
        lse_h = helion_out[..., LATENT_DIM]
        lse_b = baseline_out[..., LATENT_DIM]
        valid = torch.isfinite(lse_b)
        assert valid.any(), "All baseline logsum_exp values are non-finite"
        max_abs_diff = (lse_h[valid] - lse_b[valid]).abs().max().item()
        assert torch.allclose(lse_h[valid], lse_b[valid], atol=1e-2, rtol=1e-2), (
            f"logsum_exp mismatch on valid splits: max_abs_diff={max_abs_diff:.6f}"
        )

    def test_no_nans_helion(self, outputs):
        helion_out, _ = outputs
        assert not torch.isnan(helion_out).any(), "NaNs found in helion output"

    def test_no_nans_baseline(self, outputs):
        _, baseline_out = outputs
        assert not torch.isnan(baseline_out).any(), "NaNs found in baseline output"

    def test_deterministic(self, test_inputs):
        """Two helion runs on the same input must produce bit-identical output."""
        out1 = torch.empty_like(test_inputs[2])
        out2 = torch.empty_like(test_inputs[2])
        _call_helion(_clone_inputs(test_inputs), out1)
        _call_helion(_clone_inputs(test_inputs), out2)
        assert torch.equal(out1, out2), "Helion kernel is non-deterministic"
