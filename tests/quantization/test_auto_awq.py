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
from torch import nn

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.config.scheduler import SchedulerConfig
from vllm.config.utils import Range
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


# =============================================================================
# Decode/prefill dispatch under torch.compile
# =============================================================================

AWQ_COMPILE_K = 256
AWQ_COMPILE_N = 512
AWQ_COMPILE_GROUP_SIZE = 128
AWQ_COMPILE_MAX_TOKENS = 512


@support_torch_compile(dynamic_arg_dims={"x": 0})
class _AutoAWQLinearModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()
        from vllm.model_executor.layers.quantization.auto_awq import (
            AutoAWQConfig,
            AutoAWQLinearMethod,
        )

        quant_config = AutoAWQConfig(
            weight_bits=4,
            group_size=AWQ_COMPILE_GROUP_SIZE,
            zero_point=True,
            lm_head_quantized=False,
        )
        self.quant_method = AutoAWQLinearMethod(quant_config)
        pack_factor = quant_config.pack_factor
        num_groups = AWQ_COMPILE_K // AWQ_COMPILE_GROUP_SIZE
        self.linear = nn.Module()
        for name, shape, dtype in (
            ("qweight", (AWQ_COMPILE_K, AWQ_COMPILE_N // pack_factor), torch.int32),
            ("qzeros", (num_groups, AWQ_COMPILE_N // pack_factor), torch.int32),
            ("scales", (num_groups, AWQ_COMPILE_N), torch.float16),
        ):
            self.linear.register_parameter(
                name, nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self.linear, x)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="AutoAWQLinearMethod's awq_gemm path runs on CUDA and ROCm.",
)
def test_auto_awq_linear_dispatch_under_torch_compile(
    monkeypatch, disable_vllm_compile_cache
):
    """With VLLM_COMPILE, one compiled graph serves every batch size in a
    compile range, so the awq_gemm vs. dequantize + matmul choice must be
    made per call: M=1 has to reach awq_gemm, and M>=256 has to keep
    awq_dequantize + matmul."""
    from vllm.model_executor.layers.quantization import auto_awq

    calls: list[str] = []

    def fake_awq_gemm(x, qweight, scales, qzeros, split_k_iters):
        calls.append("awq_gemm")
        return x.new_zeros(x.shape[0], qweight.shape[1] * 8)

    def fake_awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy):
        calls.append("awq_dequantize")
        return scales.new_zeros(qweight.shape[0], qweight.shape[1] * 8)

    monkeypatch.setattr(auto_awq.ops, "awq_gemm", fake_awq_gemm)
    monkeypatch.setattr(auto_awq.ops, "awq_dequantize", fake_awq_dequantize)

    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=AWQ_COMPILE_MAX_TOKENS,
            max_model_len=AWQ_COMPILE_MAX_TOKENS,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    # Decode and prefill sizes must share one compile range (and so one
    # graph), otherwise this test passes even with a compile-time branch.
    assert vllm_config.compilation_config.get_compile_ranges() == [
        Range(start=1, end=AWQ_COMPILE_MAX_TOKENS)
    ]

    device = torch.device(current_platform.device_type)

    def run(model: nn.Module, num_tokens: int) -> list[str]:
        calls.clear()
        x = torch.randn(num_tokens, AWQ_COMPILE_K, dtype=torch.float16, device=device)
        with set_forward_context({}, vllm_config=vllm_config):
            out = model(x)
        assert out.shape == (num_tokens, AWQ_COMPILE_N)
        return list(calls)

    with (
        set_current_vllm_config(vllm_config),
        torch.device(device),
        torch.inference_mode(),
    ):
        model = _AutoAWQLinearModel(vllm_config=vllm_config, prefix="").eval()
        with compilation_counter.expect(
            num_graphs_seen=1,
            num_backend_compilations=1,
        ):
            # The first call compiles the graph with a prefill-sized input,
            # like the profile run does.
            assert run(model, AWQ_COMPILE_MAX_TOKENS) == ["awq_dequantize"]
            assert run(model, 1) == ["awq_gemm"]
            assert run(model, 256) == ["awq_dequantize"]
