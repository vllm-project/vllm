# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.mla_decode_fp8_per_token import (
    _softmax_reduce_pick_cache,
    fake_impl_softmax_reduce,
    mla_decode_softmax_reduce,
    pick_softmax_reduce_config,
)


def skip_if_platform_unsupported():
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        platform = get_canonical_gpu_name()

        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs(
            "mla_decode_softmax_reduce", platform
        )
        if len(configs) == 0:
            pytest.skip(
                "Current GPU platform not supported formla_decode_softmax_reduce kernel"
            )

    except (ImportError, RuntimeError, KeyError):
        pytest.skip(
            "Error detecting platform support for mla_decode_softmax_reduce kernel"
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


@pytest.fixture(autouse=True)
def clear_pick_cache():
    """Clear the global config-picker cache before and after every test.

    Without this, a test that writes a bogus key (e.g. to verify cache
    reuse) would poison subsequent tests in other classes, causing the
    kernel to receive an invalid CaseKey and raise ValueError.
    """
    _softmax_reduce_pick_cache.clear()
    yield
    _softmax_reduce_pick_cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    batch: int,
    heads: int,
    num_kv_splits: int,
    latent_dim: int,
    seq_len: int,
    device: str = "cuda",
):
    """Return (partial_attn, attn_out, lse_out, b_seq_len)."""
    partial_attn = torch.randn(
        batch,
        heads,
        num_kv_splits,
        latent_dim + 1,
        device=device,
        dtype=torch.float32,
    )
    attn_out = torch.empty(batch, heads, latent_dim, device=device, dtype=torch.float32)
    lse_out = torch.empty(batch, heads, device=device, dtype=torch.float32)
    b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return partial_attn, attn_out, lse_out, b_seq_len


def _pytorch_softmax_reduce(
    partial_attn: torch.Tensor,  # (batch, heads, num_kv_splits, latent_dim + 1)
    attn_out: torch.Tensor,  # (batch, heads, latent_dim)
    lse_out: torch.Tensor,  # (batch, heads)
    b_seq_len: torch.Tensor,  # (batch,)
    num_kv_splits: int,
    latent_dim: int,
) -> None:
    """Pure-PyTorch reference for the online-softmax reduce over KV splits."""
    batch, heads = partial_attn.shape[:2]

    for b in range(batch):
        seq_len = int(b_seq_len[b].item())
        for h in range(heads):
            e_sum = torch.zeros([], dtype=torch.float32, device=partial_attn.device)
            e_max = torch.full(
                [], -float("inf"), dtype=torch.float32, device=partial_attn.device
            )
            acc = torch.zeros(
                latent_dim, dtype=torch.float32, device=partial_attn.device
            )

            kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits

            for s in range(num_kv_splits):
                split_start = s * kv_len_per_split
                split_end = min(split_start + kv_len_per_split, seq_len)

                if split_end <= split_start:
                    continue

                tv = partial_attn[b, h, s, :latent_dim]
                tlogic = partial_attn[b, h, s, latent_dim]

                n_e_max = torch.maximum(tlogic, e_max)
                old_scale = torch.exp(e_max - n_e_max)
                acc = acc * old_scale + torch.exp(tlogic - n_e_max) * tv
                e_sum = e_sum * old_scale + torch.exp(tlogic - n_e_max)
                e_max = n_e_max

            attn_out[b, h] = acc / e_sum
            lse_out[b, h] = e_max + torch.log(e_sum)


# ---------------------------------------------------------------------------
# Config picker tests
# ---------------------------------------------------------------------------


class TestSoftmaxReduceConfigPicker:
    def test_exact_match(self):
        config_keys = [
            CaseKey({"batch": 1, "heads": 16}),
            CaseKey({"batch": 4, "heads": 16}),
        ]
        partial_attn = torch.randn(1, 16, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 1, "heads": 16})

    def test_closest_batch(self):
        config_keys = [
            CaseKey({"batch": 1, "heads": 16}),
            CaseKey({"batch": 16, "heads": 16}),
        ]
        # batch=5 is closer to 4 than to 16 — but we only have 1 and 16;
        # abs(1-5)=4, abs(16-5)=11 → should pick batch=1
        partial_attn = torch.randn(5, 16, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 1, "heads": 16})

    def test_closest_batch_upper(self):
        config_keys = [
            CaseKey({"batch": 1, "heads": 16}),
            CaseKey({"batch": 16, "heads": 16}),
        ]
        # batch=12 is closer to 16 than to 1
        partial_attn = torch.randn(12, 16, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 16, "heads": 16})

    def test_smallest_heads_gte_input(self):
        config_keys = [
            CaseKey({"batch": 4, "heads": 8}),
            CaseKey({"batch": 4, "heads": 16}),
            CaseKey({"batch": 4, "heads": 32}),
        ]
        # heads=10 → smallest available >= 10 is 16
        partial_attn = torch.randn(4, 10, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 4, "heads": 16})

    def test_heads_exact_match(self):
        config_keys = [
            CaseKey({"batch": 4, "heads": 8}),
            CaseKey({"batch": 4, "heads": 16}),
            CaseKey({"batch": 4, "heads": 32}),
        ]
        partial_attn = torch.randn(4, 16, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 4, "heads": 16})

    def test_heads_fallback_to_largest(self):
        config_keys = [
            CaseKey({"batch": 4, "heads": 8}),
            CaseKey({"batch": 4, "heads": 16}),
        ]
        # heads=64 exceeds all available — fall back to largest (16)
        partial_attn = torch.randn(4, 64, 4, 513, device="cuda")
        selected = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert selected == CaseKey({"batch": 4, "heads": 16})

    def test_no_configs(self):
        selected = pick_softmax_reduce_config(
            (torch.randn(1, 16, 4, 513, device="cuda"),), []
        )
        assert selected is None

    def test_cache_is_populated(self):
        config_keys = [CaseKey({"batch": 4, "heads": 16})]
        partial_attn = torch.randn(4, 16, 4, 513, device="cuda")
        pick_softmax_reduce_config((partial_attn,), config_keys)
        assert (4, 16) in _softmax_reduce_pick_cache

    def test_cache_is_reused(self):
        """Second call with same (batch, heads) must return the cached result
        without re-running the selection logic, even when config_keys changes."""
        config_keys = [CaseKey({"batch": 4, "heads": 16})]
        partial_attn = torch.randn(4, 16, 4, 513, device="cuda")

        result1 = pick_softmax_reduce_config((partial_attn,), config_keys)
        assert result1 == CaseKey({"batch": 4, "heads": 16})
        assert (4, 16) in _softmax_reduce_pick_cache

        # Pass a *different* config_keys list — if the cache is consulted the
        # result must still be the first answer (not the new list's only entry).
        alt_config_keys = [CaseKey({"batch": 1, "heads": 8})]
        result2 = pick_softmax_reduce_config((partial_attn,), alt_config_keys)
        assert result2 == result1  # cache hit; alt_config_keys was ignored


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestSoftmaxReduceCorrectness:
    @pytest.mark.parametrize("batch", [1, 4, 16])
    @pytest.mark.parametrize("heads", [8, 16])
    @pytest.mark.parametrize("num_kv_splits", [2, 4])
    @pytest.mark.parametrize("latent_dim", [128, 512])
    @pytest.mark.parametrize("seq_len", [64, 512])
    def test_correctness(self, batch, heads, num_kv_splits, latent_dim, seq_len):
        skip_if_platform_unsupported()

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )

        ref_attn_out = torch.empty_like(attn_out)
        ref_lse_out = torch.empty_like(lse_out)
        _pytorch_softmax_reduce(
            partial_attn,
            ref_attn_out,
            ref_lse_out,
            b_seq_len,
            num_kv_splits,
            latent_dim,
        )

        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )

        torch.testing.assert_close(
            attn_out,
            ref_attn_out,
            atol=1e-4,
            rtol=1e-4,
            msg=f"attn_out mismatch: batch={batch}, heads={heads}, "
            f"num_kv_splits={num_kv_splits}, latent_dim={latent_dim}",
        )
        torch.testing.assert_close(
            lse_out,
            ref_lse_out,
            atol=1e-4,
            rtol=1e-4,
            msg=f"lse_out mismatch: batch={batch}, heads={heads}, "
            f"num_kv_splits={num_kv_splits}, latent_dim={latent_dim}",
        )

    def test_single_kv_split(self):
        """With a single split the result should equal a simple softmax over values."""
        skip_if_platform_unsupported()
        batch, heads, latent_dim, seq_len = 2, 4, 128, 64
        num_kv_splits = 1

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )
        ref_attn_out = torch.empty_like(attn_out)
        ref_lse_out = torch.empty_like(lse_out)
        _pytorch_softmax_reduce(
            partial_attn,
            ref_attn_out,
            ref_lse_out,
            b_seq_len,
            num_kv_splits,
            latent_dim,
        )
        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )
        torch.testing.assert_close(attn_out, ref_attn_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_out, ref_lse_out, atol=1e-4, rtol=1e-4)

    def test_seq_len_not_divisible_by_num_kv_splits(self):
        """Sequence lengths that don't divide evenly across splits."""
        skip_if_platform_unsupported()
        batch, heads, num_kv_splits, latent_dim = 2, 4, 4, 128
        seq_len = 100  # not divisible by 4

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )
        ref_attn_out = torch.empty_like(attn_out)
        ref_lse_out = torch.empty_like(lse_out)
        _pytorch_softmax_reduce(
            partial_attn,
            ref_attn_out,
            ref_lse_out,
            b_seq_len,
            num_kv_splits,
            latent_dim,
        )
        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )
        torch.testing.assert_close(attn_out, ref_attn_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_out, ref_lse_out, atol=1e-4, rtol=1e-4)

    def test_variable_seq_lens_in_batch(self):
        """Each item in the batch has a different sequence length."""
        skip_if_platform_unsupported()
        batch, heads, num_kv_splits, latent_dim = 4, 4, 4, 128

        partial_attn = torch.randn(
            batch,
            heads,
            num_kv_splits,
            latent_dim + 1,
            device="cuda",
            dtype=torch.float32,
        )
        attn_out = torch.empty(
            batch, heads, latent_dim, device="cuda", dtype=torch.float32
        )
        lse_out = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
        b_seq_len = torch.tensor([32, 64, 100, 512], device="cuda", dtype=torch.int32)

        ref_attn_out = torch.empty_like(attn_out)
        ref_lse_out = torch.empty_like(lse_out)
        _pytorch_softmax_reduce(
            partial_attn,
            ref_attn_out,
            ref_lse_out,
            b_seq_len,
            num_kv_splits,
            latent_dim,
        )
        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )
        torch.testing.assert_close(attn_out, ref_attn_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_out, ref_lse_out, atol=1e-4, rtol=1e-4)

    def test_output_shapes(self):
        skip_if_platform_unsupported()
        batch, heads, num_kv_splits, latent_dim, seq_len = 4, 16, 4, 512, 512

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )
        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )

        assert attn_out.shape == (batch, heads, latent_dim)
        assert lse_out.shape == (batch, heads)

    def test_numerically_stable_with_large_logits(self):
        """Kernel should not produce NaN/Inf when logits are large."""
        skip_if_platform_unsupported()
        batch, heads, num_kv_splits, latent_dim, seq_len = 2, 4, 4, 128, 128

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )
        # Inject extreme logit values
        partial_attn[..., latent_dim] = 1e6

        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )
        assert not torch.isnan(attn_out).any(), "attn_out contains NaN"
        assert not torch.isinf(attn_out).any(), "attn_out contains Inf"
        assert not torch.isnan(lse_out).any(), "lse_out contains NaN"

    def test_numerically_stable_with_negative_large_logits(self):
        """Kernel should remain stable when logits are very negative."""
        skip_if_platform_unsupported()
        batch, heads, num_kv_splits, latent_dim, seq_len = 2, 4, 4, 128, 128

        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len
        )
        partial_attn[..., latent_dim] = -1e6

        mla_decode_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )
        assert not torch.isnan(attn_out).any()
        assert not torch.isinf(attn_out).any()


# ---------------------------------------------------------------------------
# Fake-impl / integration tests
# ---------------------------------------------------------------------------


class TestSoftmaxReduceFakeImpl:
    def test_fake_impl_is_no_op(self):
        """fake_impl_softmax_reduce must return None without mutating outputs."""
        batch, heads, num_kv_splits, latent_dim = 2, 4, 4, 128
        partial_attn, attn_out, lse_out, b_seq_len = _make_inputs(
            batch, heads, num_kv_splits, latent_dim, seq_len=64
        )
        sentinel_attn = attn_out.clone()
        sentinel_lse = lse_out.clone()

        result = fake_impl_softmax_reduce(
            partial_attn, attn_out, lse_out, b_seq_len, num_kv_splits, latent_dim
        )

        assert result is None
        # Outputs must be untouched
        torch.testing.assert_close(attn_out, sentinel_attn)
        torch.testing.assert_close(lse_out, sentinel_lse)


class TestSoftmaxReduceIntegration:
    def test_kernel_registration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        assert "mla_decode_softmax_reduce" in registered

        wrapper = registered["mla_decode_softmax_reduce"]
        assert wrapper.op_name == "mla_decode_softmax_reduce"
        assert wrapper._config_picker is not None

    def test_fake_impl_registered(self):
        from vllm.kernels.helion.register import get_registered_kernels

        wrapper = get_registered_kernels()["mla_decode_softmax_reduce"]
        assert wrapper._fake_impl is not None
