# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AutoAWQConfig behavior after unification.

These tests verify the bug fixes for:
1. CPU platform override conflict (auto_awq should not override on CPU)
2. MoE fallback compatibility (full_config["quant_method"] should be "awq")
3. Config attribute consistency
4. End-to-end quantization method loading (auto_awq loads and runs correctly)

Note: Tests that require importing the full auto_awq module (which has GPU-dependent
imports) should use subprocess or be run in a GPU environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform


def _get_auto_awq_config_source() -> str:
    """Read the AutoAWQConfig class source code for isolated testing."""
    import inspect

    import vllm.model_executor.layers.quantization.auto_awq as auto_awq_module

    return inspect.getsource(auto_awq_module.AutoAWQConfig)


class TestAutoAWQConfigFromConfig:
    """Tests for AutoAWQConfig.from_config behavior.

    These tests require GPU environment to import the full module.
    They are skipped on non-GPU platforms.
    """

    def test_full_config_quant_method_is_awq_for_moe_fallback(self):
        """full_config should have quant_method='awq' for MoE fallback compatibility.

        MoeWNA16Config only accepts 'gptq' or 'awq' as linear_quant_method.
        If full_config has 'auto_awq', the MoE fallback will fail.
        """
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
        awq_config = AutoAWQConfig.from_config(config)

        # Verify quant_method is 'awq' for MoE fallback
        assert awq_config.full_config["quant_method"] == "awq", (
            f"Expected quant_method='awq', got {awq_config.full_config['quant_method']}"
        )

    def test_full_config_preserves_other_fields(self):
        """full_config should preserve all original config fields."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
            "custom_field": "custom_value",
        }
        awq_config = AutoAWQConfig.from_config(config)

        assert awq_config.full_config["w_bit"] == 4
        assert awq_config.full_config["q_group_size"] == 128
        assert awq_config.full_config["zero_point"] is True
        assert awq_config.full_config["lm_head"] is False
        assert awq_config.full_config["custom_field"] == "custom_value"

    def test_full_config_is_copy_not_original(self):
        """full_config should be a copy, not the original dict."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
        original_quant_method = config.get("quant_method")

        AutoAWQConfig.from_config(config)

        # Original config should not be modified
        assert config.get("quant_method") == original_quant_method


class TestAutoAWQConfigAttributes:
    """Tests for AutoAWQConfig attribute consistency.

    These tests require GPU environment to import the full module.
    They are skipped on non-GPU platforms.
    """

    def test_config_attributes_match_input(self):
        """Config attributes should match input values."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_config = AutoAWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=["lm_head"],
        )

        assert awq_config.weight_bits == 4
        assert awq_config.group_size == 128
        assert awq_config.zero_point is True
        assert awq_config.lm_head_quantized is False
        assert awq_config.modules_to_not_convert == ["lm_head"]

    def test_pack_factor_for_4bit(self):
        """Pack factor should be 8 for 4-bit quantization."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_config = AutoAWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            lm_head_quantized=False,
        )

        assert awq_config.pack_factor == 8  # 32 // 4


class TestAutoAWQConfigOverrideLogic:
    """Tests for override logic by parsing source code (no GPU import required)."""

    def _get_auto_awq_source(self) -> str:
        """Read the auto_awq.py source file."""
        import inspect
        import pathlib

        import vllm.model_executor.layers.quantization.auto_awq as auto_awq_module

        source_path = inspect.getfile(auto_awq_module)
        return pathlib.Path(source_path).read_text()

    def test_cpu_check_in_override_method(self):
        """override_quantization_method should check current_platform.is_cpu()."""
        source = self._get_auto_awq_source()

        # Verify the CPU check exists in override method
        assert "current_platform.is_cpu()" in source, (
            "override_quantization_method should check is_cpu()"
        )
        assert "return None" in source, (
            "override_quantization_method should return None on CPU"
        )

    def test_quant_method_normalization_in_from_config(self):
        """from_config should normalize quant_method to 'awq' for MoE fallback."""
        source = self._get_auto_awq_source()

        # Verify the normalization exists
        assert (
            '"quant_method"] = "awq"' in source or "'quant_method'] = 'awq'" in source
        ), "from_config should set quant_method='awq' in full_config"


# =============================================================================
# End-to-end integration tests (require GPU environment)
# =============================================================================

PROMPT = "On the surface of Mars, we found"

# Small AWQ model for testing - using Qwen2 1.5B which has official AWQ checkpoint
AWQ_MODELS = [
    "Qwen/Qwen2-1.5B-Instruct-AWQ",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_awq"),
    reason="auto_awq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", AWQ_MODELS)
def test_auto_awq_quantization_method(
    model_id: str, monkeypatch, dist_init, workspace_init
):
    """Test that quantization='auto_awq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model, vllm_config = load_model_without_vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_awq",
        model_config_kwargs={"max_model_len": 2048},
    )
    target_device = torch.device(current_platform.device_type)

    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQLinearMethod,
        AutoAWQMarlinLinearMethod,
    )

    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    assert isinstance(
        qkv_proj.quant_method,
        (AutoAWQLinearMethod, AutoAWQMarlinLinearMethod),
    ), (
        "Expected AutoAWQLinearMethod or AutoAWQMarlinLinearMethod, "
        f"got {type(qkv_proj.quant_method)}"
    )

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=target_device)
    positions = torch.arange(input_ids.numel(), device=target_device)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


def test_auto_awq_config_get_name():
    """Test that AutoAWQConfig.get_name() returns 'auto_awq'."""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    assert AutoAWQConfig.get_name() == "auto_awq"


class _FakeQuantConfig:
    def __init__(self, pack_factor: int = 8):
        self.pack_factor = pack_factor


def _make_fake_awq_layer_and_method(
    k: int, n: int, group_size: int, device: torch.device
):
    """Build a minimal (layer, method) pair that exercises
    AutoAWQLinearMethod.apply's dispatch logic without a full model load."""
    import vllm.model_executor.layers.quantization.auto_awq as auto_awq_module

    layer = SimpleNamespace()
    layer.qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (k, n // 8), dtype=torch.int32, device=device
    )
    layer.scales = torch.rand((k // group_size, n), dtype=torch.float16, device=device)
    layer.qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // group_size, n // 8),
        dtype=torch.int32,
        device=device,
    )

    method = auto_awq_module.AutoAWQLinearMethod.__new__(
        auto_awq_module.AutoAWQLinearMethod
    )
    method.quant_config = _FakeQuantConfig()
    return layer, method


def _count_fused_calls(monkeypatch) -> list[int]:
    """Patch awq_gemm_fused_fp32 with a call counter; returns a 1-item list
    holding the running count (mutable cell, since the closure needs to be
    read after the patched calls happen)."""
    import vllm.model_executor.layers.quantization.auto_awq as auto_awq_module

    counter = [0]
    original_fused_gemm = auto_awq_module.awq_gemm_fused_fp32

    def _counting_fused_gemm(*args, **kwargs):
        counter[0] += 1
        return original_fused_gemm(*args, **kwargs)

    monkeypatch.setattr(auto_awq_module, "awq_gemm_fused_fp32", _counting_fused_gemm)
    return counter


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(89)),
    reason="The fused AWQ BI GEMM path only dispatches on SM89.",
)
def test_auto_awq_batch_invariant_dispatch_ignores_input_contiguity(monkeypatch):
    """Fused and legacy paths aren't numerically identical, so dispatch must
    not depend on incidental input contiguity (regression test for the bug
    fixed in this PR)."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    fused_calls = _count_fused_calls(monkeypatch)

    device = torch.device(current_platform.device_type)
    k, n, group_size, m = 3584, 512, 128, 16
    layer, method = _make_fake_awq_layer_and_method(k, n, group_size, device)

    x_contig = torch.rand((m, k), dtype=torch.float16, device=device)
    padded = torch.zeros((m, 2 * k), dtype=torch.float16, device=device)
    padded[:, :k] = x_contig
    x_noncontig = padded[:, :k]
    assert not x_noncontig.is_contiguous()
    assert torch.equal(x_contig, x_noncontig)

    out_contig = method.apply(layer, x_contig)
    out_noncontig = method.apply(layer, x_noncontig)

    assert fused_calls[0] == 2, (
        "Expected both the contiguous and non-contiguous inputs to dispatch "
        f"to the fused kernel; got {fused_calls[0]} fused call(s)."
    )
    # assert_close(atol=0, rtol=0) still treats +0.0 and -0.0 as equal;
    # compare raw bytes to catch that and any other bit-level divergence.
    assert torch.equal(
        out_contig.contiguous().view(torch.uint8),
        out_noncontig.contiguous().view(torch.uint8),
    )
