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


@pytest.mark.parametrize(
    "allow_fallback,input_size,error",
    [
        (True, 128, None),
        (False, 128, "WNA16 linear layer"),
        (True, 127, "input size is not aligned"),
    ],
)
def test_auto_awq_fallback_weight_creation(
    dist_init,
    monkeypatch: pytest.MonkeyPatch,
    allow_fallback: bool,
    input_size: int,
    error: str | None,
) -> None:
    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
        AutoAWQLinearMethod,
        AutoAWQMarlinLinearMethod,
    )

    monkeypatch.setenv(
        "VLLM_DISABLED_KERNELS",
        ",".join(k.__name__ for k in _POSSIBLE_KERNELS[current_platform._enum]),
    )
    config = AutoAWQConfig(4, 128, True, False)
    layer = torch.nn.Module()
    method = AutoAWQMarlinLinearMethod(config, allow_legacy_fallback=allow_fallback)
    layer.quant_method = method

    def create_weights() -> None:
        method.create_weights(layer, input_size, [64], input_size, 64, torch.float16)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            create_weights()
        assert layer.quant_method is method
        assert list(layer.parameters()) == []
    else:
        create_weights()
        assert isinstance(layer.quant_method, AutoAWQLinearMethod)
        assert layer.qweight.shape == (128, 8)
        assert layer.qweight.packed_dim == 1
        assert layer.qzeros.shape == (1, 8)
        assert layer.scales.shape == (1, 64)


@pytest.mark.parametrize(
    "kernel_options",
    [
        {"linear_backend": "triton"},
        {"linear_backend_per_quant": {"mixed_precision": "triton"}},
    ],
)
def test_auto_awq_explicit_backend_does_not_fallback(
    dist_init, monkeypatch: pytest.MonkeyPatch, kernel_options: dict
) -> None:
    from vllm.config import VllmConfig
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_KERNELS,
        NoCompatibleMPLinearKernelError,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    monkeypatch.setenv(
        "VLLM_DISABLED_KERNELS",
        ",".join(k.__name__ for k in _POSSIBLE_KERNELS[current_platform._enum]),
    )
    config = VllmConfig(kernel_config=KernelConfig(**kernel_options))
    with (
        set_current_vllm_config(config),
        pytest.raises(NoCompatibleMPLinearKernelError, match="WNA16 linear layer"),
    ):
        ReplicatedLinear(
            128,
            64,
            bias=False,
            params_dtype=torch.float16,
            quant_config=AutoAWQConfig(4, 128, True, False),
        )


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

    # Import inside the guard: importing vllm.platforms.rocm reads gcnArchName,
    # which must not happen on non-ROCm hosts.
    is_rdna = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        is_rdna = on_gfx1x()

    if is_rdna:
        # Regression guard for issue #63455: on RDNA, AutoAWQ must reach the
        # mixed-precision W4A16 kernel instead of the generic AWQ kernels.
        assert isinstance(qkv_proj.quant_method, AutoAWQMarlinLinearMethod), (
            "Expected AutoAWQMarlinLinearMethod on RDNA, "
            f"got {type(qkv_proj.quant_method)}"
        )
    else:
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
