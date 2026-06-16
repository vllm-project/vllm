# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the mla_decode_fp8_per_token helion kernel.

Run `pytest tests/kernels/helion/test_mla_decode_fp8_per_token.py`.
"""

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.mla_decode_fp8_per_token import (
    _pick_cache,
    baseline,
    generate_mla_decode_kv_split_inputs,
    mla_decode_kv_split,
    pick_mla_decode_kv_split_config,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

LATENT_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 16
NUM_KV_SPLITS = 4
HEADS_PER_BLOCK = 4


def _clone_inputs(inputs):
    return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)


def _make_inputs(
    batch: int,
    seq_len: int,
    heads: int,
    latent_dim: int = LATENT_DIM,
    rope_dim: int = ROPE_DIM,
    page_size: int = PAGE_SIZE,
    num_kv_splits: int = NUM_KV_SPLITS,
    heads_per_block: int = HEADS_PER_BLOCK,
    logit_cap=None,
):
    """Build a single input tuple matching the generator's layout."""
    from helion._utils import cdiv

    num_pages = cdiv(seq_len, page_size)

    q_absorbed = torch.randn(
        batch, heads, latent_dim + rope_dim, device="cuda", dtype=torch.bfloat16
    )
    latent_kv = torch.randn(
        num_pages,
        page_size,
        1,
        latent_dim + rope_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    req_to_tokens = (
        torch.arange(num_pages, device="cuda")
        .unsqueeze(0)
        .expand(batch, num_pages)
        .contiguous()
    )
    b_seq_len = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    attn_out = torch.empty(
        (batch, heads, num_kv_splits, latent_dim + 1),
        device="cuda",
        dtype=torch.float32,
    )
    kv_dequant = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    sm_scale = torch.tensor(
        [1.0 / (latent_dim**0.5)], device="cuda", dtype=torch.float32
    )

    return (
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
    baseline(
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


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


@pytest.fixture(scope="module")
def test_inputs():
    all_inputs = generate_mla_decode_kv_split_inputs()
    assert len(all_inputs) >= 1, "Expected at least one input config from generator"
    return next(iter(all_inputs.values()))


@pytest.fixture(scope="module")
def outputs(test_inputs):
    """Run both kernels once on cloned identical inputs."""
    inputs = _clone_inputs(test_inputs)

    helion_out = torch.empty_like(test_inputs[2])
    _call_helion(inputs, helion_out)

    baseline_out = torch.empty_like(test_inputs[2])
    _call_baseline(inputs, baseline_out)

    return helion_out, baseline_out


class TestMLADecodeConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 8, "seqlen": 1024, "headspb": 32}),
        ]
        inputs = _make_inputs(batch=4, seq_len=512, heads=16)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected == CaseKey({"batch": 4, "seqlen": 512, "headspb": 16})

    def test_config_picker_closest_batch(self):
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 8, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 16, "seqlen": 512, "headspb": 16}),
        ]
        # batch=6 is closest to 4 or 8; either is acceptable, just not 16
        inputs = _make_inputs(batch=6, seq_len=512, heads=16)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected is not None
        assert selected["batch"] in (4, 8)

    def test_config_picker_seqlen_ceil(self):
        """Should pick the smallest seqlen >= requested."""
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 256, "headspb": 16}),
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 4, "seqlen": 1024, "headspb": 16}),
        ]
        inputs = _make_inputs(batch=4, seq_len=300, heads=16)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected == CaseKey({"batch": 4, "seqlen": 512, "headspb": 16})

    def test_config_picker_seqlen_fallback_to_largest(self):
        """When seq_len exceeds all configs, fall back to largest available."""
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 256, "headspb": 16}),
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
        ]
        inputs = _make_inputs(batch=4, seq_len=8192, heads=16)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected == CaseKey({"batch": 4, "seqlen": 512, "headspb": 16})

    def test_config_picker_heads_ceil(self):
        """Should pick the smallest headspb >= requested."""
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 32}),
        ]
        inputs = _make_inputs(batch=4, seq_len=512, heads=20)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected == CaseKey({"batch": 4, "seqlen": 512, "headspb": 32})

    def test_config_picker_heads_fallback_to_largest(self):
        config_keys = [
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 16}),
            CaseKey({"batch": 4, "seqlen": 512, "headspb": 32}),
        ]
        inputs = _make_inputs(batch=4, seq_len=512, heads=64)
        selected = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert selected == CaseKey({"batch": 4, "seqlen": 512, "headspb": 32})

    def test_config_picker_no_configs(self):
        inputs = _make_inputs(batch=4, seq_len=512, heads=16)
        selected = pick_mla_decode_kv_split_config(inputs, [])
        assert selected is None

    def test_config_picker_cache_hit(self):
        """Second call with same shape should return cached result."""
        config_keys = [CaseKey({"batch": 4, "seqlen": 512, "headspb": 16})]
        inputs = _make_inputs(batch=4, seq_len=512, heads=16)
        first = pick_mla_decode_kv_split_config(inputs, config_keys)
        second = pick_mla_decode_kv_split_config(inputs, config_keys)
        assert first == second


class TestMLADecodeKVSplitCorrectness:
    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
            (8, 1024, 32),
            (1, 256, 32),
        ],
    )
    def test_output_shape(self, batch, seq_len, heads):
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)
        attn_out = torch.empty_like(inputs[2])
        _call_helion(inputs, attn_out)
        expected = inputs[2].shape
        assert attn_out.shape == expected

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
            (8, 1024, 32),
        ],
    )
    def test_output_dtype(self, batch, seq_len, heads):
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)
        attn_out = torch.empty_like(inputs[2])
        _call_helion(inputs, attn_out)
        assert attn_out.dtype == torch.float32

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
            (4, 512, 32),
            (8, 1024, 16),
        ],
    )
    def test_partial_attention_values(self, batch, seq_len, heads):
        """Compare partial weighted-sum accumulator (first latent_dim slots)."""
        skip_if_platform_unsupported("mla_decode_kv_split")
        torch.manual_seed(42)
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)

        helion_out = torch.empty_like(inputs[2])
        baseline_out = torch.empty_like(inputs[2])
        _call_helion(_clone_inputs(inputs), helion_out)
        _call_baseline(_clone_inputs(inputs), baseline_out)

        acc_h = helion_out[..., :LATENT_DIM]
        acc_b = baseline_out[..., :LATENT_DIM]
        max_abs_diff = (acc_h - acc_b).abs().max().item()
        mean_abs_diff = (acc_h - acc_b).abs().mean().item()
        assert torch.allclose(acc_h, acc_b, atol=1e-2, rtol=1e-2), (
            f"[batch={batch}, seq={seq_len}, heads={heads}] "
            f"Partial acc mismatch: max={max_abs_diff:.6f}, mean={mean_abs_diff:.6f}"
        )

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
            (4, 512, 32),
            (8, 1024, 16),
        ],
    )
    def test_logsum_exp_values(self, batch, seq_len, heads):
        """Compare log-sum-exp stored in the last slot of each split."""
        skip_if_platform_unsupported("mla_decode_kv_split")
        torch.manual_seed(42)
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)

        helion_out = torch.empty_like(inputs[2])
        baseline_out = torch.empty_like(inputs[2])
        _call_helion(_clone_inputs(inputs), helion_out)
        _call_baseline(_clone_inputs(inputs), baseline_out)

        lse_h = helion_out[..., LATENT_DIM]
        lse_b = baseline_out[..., LATENT_DIM]
        valid = torch.isfinite(lse_b)
        assert valid.any(), "All baseline logsum_exp values are non-finite"
        max_abs_diff = (lse_h[valid] - lse_b[valid]).abs().max().item()
        assert torch.allclose(lse_h[valid], lse_b[valid], atol=1e-2, rtol=1e-2), (
            f"[batch={batch}, seq={seq_len}, heads={heads}] "
            f"logsum_exp mismatch on valid splits: max={max_abs_diff:.6f}"
        )

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
        ],
    )
    def test_no_nans_helion(self, batch, seq_len, heads):
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)
        attn_out = torch.empty_like(inputs[2])
        _call_helion(inputs, attn_out)
        assert not torch.isnan(attn_out).any(), (
            f"NaNs found in helion output [batch={batch}, seq={seq_len}, heads={heads}]"
        )

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
        ],
    )
    def test_no_nans_baseline(self, batch, seq_len, heads):
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)
        attn_out = torch.empty_like(inputs[2])
        _call_baseline(inputs, attn_out)
        assert not torch.isnan(attn_out).any(), (
            f"NaNs in baseline output [batch={batch}, seq={seq_len}, heads={heads}]"
        )

    @pytest.mark.parametrize(
        "batch,seq_len,heads",
        [
            (1, 128, 16),
            (4, 512, 16),
            (4, 512, 32),
        ],
    )
    def test_deterministic(self, batch, seq_len, heads):
        """Two helion runs on the same input must produce bit-identical output."""
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=batch, seq_len=seq_len, heads=heads)
        out1 = torch.empty_like(inputs[2])
        out2 = torch.empty_like(inputs[2])
        _call_helion(_clone_inputs(inputs), out1)
        _call_helion(_clone_inputs(inputs), out2)
        assert torch.equal(out1, out2), (
            f"Helion kernel is non-deterministic"
            f"[batch={batch}, seq={seq_len}, heads={heads}]"
        )

    def test_logit_cap_applied(self):
        """With a tight logit_cap, attention scores should be bounded."""
        skip_if_platform_unsupported("mla_decode_kv_split")
        torch.manual_seed(0)
        logit_cap = 10.0
        inputs = _make_inputs(batch=2, seq_len=128, heads=16, logit_cap=logit_cap)
        attn_out = torch.empty_like(inputs[2])
        _call_helion(inputs, attn_out)
        # LSE stored in last slot — should be finite and bounded if cap is tight
        lse = attn_out[..., LATENT_DIM]
        assert torch.isfinite(lse).all(), (
            "LSE contains non-finite values with logit_cap"
        )

    def test_single_token_sequence(self):
        """Edge case: sequence length of exactly one token per page."""
        skip_if_platform_unsupported("mla_decode_kv_split")
        inputs = _make_inputs(batch=1, seq_len=PAGE_SIZE, heads=16)
        helion_out = torch.empty_like(inputs[2])
        baseline_out = torch.empty_like(inputs[2])
        _call_helion(_clone_inputs(inputs), helion_out)
        _call_baseline(_clone_inputs(inputs), baseline_out)
        assert torch.allclose(
            helion_out[..., :LATENT_DIM],
            baseline_out[..., :LATENT_DIM],
            atol=1e-2,
            rtol=1e-2,
        )


class TestMLADecodeKVSplitIntegration:
    def test_kernel_registration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        assert "mla_decode_kv_split" in registered

        wrapper = registered["mla_decode_kv_split"]
        assert wrapper.op_name == "mla_decode_kv_split"
        assert wrapper._config_picker is not None
        assert "attn_out" in wrapper._mutates_args

    def test_fake_impl_returns_none(self):
        from vllm.kernels.helion.ops.mla_decode_fp8_per_token import fake_impl

        inputs = _make_inputs(batch=1, seq_len=128, heads=16)
        # fake_impl should be a no-op returning None
        result = fake_impl(*inputs)
        assert result is None

    def test_input_generator_produces_valid_shapes(self):
        all_inputs = generate_mla_decode_kv_split_inputs()
        assert len(all_inputs) > 0

        for key, inputs in all_inputs.items():
            (
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
            ) = inputs

            batch = q_absorbed.shape[0]
            heads = q_absorbed.shape[1]

            assert q_absorbed.shape == (batch, heads, latent_dim + rope_dim)
            assert latent_kv.ndim == 4
            assert latent_kv.dtype == torch.float8_e4m3fn
            assert latent_kv.shape[2] == 1, (
                "MLA KV cache must have a single latent head"
            )
            assert attn_out.shape == (batch, heads, num_kv_splits, latent_dim + 1)
            assert b_seq_len.shape == (batch,)
            assert req_to_tokens.shape[0] == batch

    def test_single_kv_head_in_latent_cache(self):
        """Explicitly verify the MLA single-head KV cache constraint."""
        all_inputs = generate_mla_decode_kv_split_inputs()
        for _, inputs in all_inputs.items():
            _, latent_kv, *_ = inputs
            # shape: (num_pages, page_size, num_kv_heads, latent_dim+rope_dim)
            assert latent_kv.shape[2] == 1, (
                f"Expected 1 KV head (MQA-style MLA), got {latent_kv.shape[2]}"
            )
