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

from tests.quantization.utils import is_quant_method_supported

_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _sign_extend_int4_nibbles(t: torch.Tensor) -> torch.Tensor:
    mask = (t & 0x8).bool()
    t = t.clone()
    t[mask] = t[mask] | 0xF0
    return t


def _dequantize_quark_signed_awq_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    *,
    pack_reorder: bool = True,
) -> torch.Tensor:
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    iweights = (qweight[:, :, None] >> shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(qweight.shape[0], -1)
    zeros = (qzeros[:, :, None] >> shifts[None, None, :]).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)

    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, device=qweight.device)
    else:
        order = torch.arange(8, device=qweight.device)
    iweights = iweights.view(qweight.shape[0], -1, 8)[:, :, order].reshape(
        qweight.shape[0], -1
    )
    zeros = zeros.view(qzeros.shape[0], -1, 8)[:, :, order].reshape(
        qzeros.shape[0], -1
    )
    iweights = _sign_extend_int4_nibbles(iweights & 0xF)
    zeros = _sign_extend_int4_nibbles(zeros & 0xF)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


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
def test_auto_awq_quantization_method(vllm_runner, model_id: str, monkeypatch):
    """Test that quantization='auto_awq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_awq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            from vllm.model_executor.layers.quantization.auto_awq import (
                AutoAWQLinearMethod,
                AutoAWQMarlinLinearMethod,
            )

            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    # Should use either AutoAWQLinearMethod (Triton) or
                    # AutoAWQMarlinLinearMethod (Marlin) depending on hardware
                    assert isinstance(
                        submodule.quant_method,
                        (AutoAWQLinearMethod, AutoAWQMarlinLinearMethod),
                    ), (
                        f"Expected AutoAWQLinearMethod or AutoAWQMarlinLinearMethod "
                        f"for {name}, got {type(submodule.quant_method)}"
                    )
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_awq_config_get_name():
    """Test that AutoAWQConfig.get_name() returns 'auto_awq'."""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    assert AutoAWQConfig.get_name() == "auto_awq"


# =============================================================================
# Quark AWQ format support tests
# =============================================================================


class TestQuarkAWQFormat:
    """Tests for Quark AWQ export format compatibility.

    Quark exports AWQ checkpoints with pack_method="reorder" and uses
    "qscales"/"qqzeros" tensor name suffixes instead of the standard
    AutoAWQ "scales"/"qzeros".  AutoAWQConfig must detect and remap these.
    """

    def test_standard_awq_is_not_detected_as_quark(self):
        """Standard AutoAWQ config without pack_method should not set is_quark_format."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "gemm",
        }
        awq_config = AutoAWQConfig.from_config(config)
        assert not awq_config.is_quark_format

    def test_quark_awq_format_detected_by_pack_method(self):
        """pack_method='reorder' in config should trigger Quark format detection."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "quant_method": "awq",
            "version": "gemm",
            "pack_method": "reorder",
        }
        awq_config = AutoAWQConfig.from_config(config)
        assert awq_config.is_quark_format

    def test_quark_order_pack_method_sets_pack_reorder_false(self):
        """pack_method='order' (Quark f2f export) must not apply AWQ nibble reorder."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "quant_method": "quark",
            "export": {"pack_method": "order"},
            "global_quant_config": {
                "weight": {
                    "dtype": "int4",
                    "group_size": 128,
                    "symmetric": True,
                }
            },
            "exclude": ["model.visual"],
        }
        awq_config = AutoAWQConfig.from_config(config)
        assert awq_config.is_quark_format
        assert not awq_config.quark_pack_reorder
        assert awq_config.lm_head_quantized

    def test_quark_excluded_lm_head_is_not_remapped_to_qweight(self):
        """Quark exports may keep lm_head as an unquantized FP16 weight."""
        import torch
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "quant_method": "quark",
            "export": {"pack_method": "reorder"},
            "global_quant_config": {
                "weight": {
                    "dtype": "int4",
                    "group_size": 128,
                    "symmetric": True,
                }
            },
            "exclude": ["lm_head"],
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        output_names = {
            name for name, _ in mapper.apply([("lm_head.weight", torch.zeros(1))])
        }

        assert not awq_config.lm_head_quantized
        assert "lm_head.weight" in output_names
        assert "lm_head.qweight" not in output_names

    def test_quark_excluded_linear_is_not_remapped_to_qweight(self):
        """Quark real_quantized exports may exclude specific linear modules."""
        import torch
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "quant_method": "quark",
            "export": {"pack_method": "reorder"},
            "global_quant_config": {
                "weight": {
                    "dtype": "int4",
                    "group_size": 128,
                    "symmetric": True,
                }
            },
            "exclude": ["model.language_model.layers.0.mlp.gate.linear"],
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        output_names = {
            name
            for name, _ in mapper.apply(
                [
                    (
                        "model.language_model.layers.0.mlp.gate.linear.weight",
                        torch.zeros(1),
                    ),
                    (
                        "model.language_model.layers.1.mlp.gate.linear.weight",
                        torch.zeros(1),
                    ),
                    ("layers.0.mlp.gate.weight", torch.zeros(1)),
                    ("layers.0.mlp.gate.qweight", torch.zeros(1)),
                    ("layers.1.mlp.gate.weight", torch.zeros(1)),
                ]
            )
        }

        assert (
            "model.language_model.layers.0.mlp.gate.linear.weight" in output_names
        )
        assert (
            "model.language_model.layers.0.mlp.gate.linear.qweight"
            not in output_names
        )
        assert (
            "model.language_model.layers.1.mlp.gate.linear.qweight"
            in output_names
        )
        assert "layers.0.mlp.gate.weight" in output_names
        assert "layers.0.mlp.gate.qweight" not in output_names
        assert "layers.1.mlp.gate.qweight" in output_names

    def test_quark_excluded_projection_qweight_is_mapped_back_to_weight(self):
        """Excluded projections stay unquantized even if a suffix remap fired."""
        import torch
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "quant_method": "quark",
            "export": {"pack_method": "reorder"},
            "global_quant_config": {
                "weight": {
                    "dtype": "int4",
                    "group_size": 128,
                    "symmetric": True,
                }
            },
            "exclude": ["layers.0.mlp.shared_expert.down_proj"],
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        output_names = {
            name
            for name, _ in mapper.apply(
                [
                    (
                        "layers.0.mlp.shared_expert.down_proj.weight",
                        torch.zeros(1),
                    ),
                    (
                        "layers.1.mlp.shared_expert.down_proj.weight",
                        torch.zeros(1),
                    ),
                ]
            )
        }

        assert "layers.0.mlp.shared_expert.down_proj.weight" in output_names
        assert (
            "layers.0.mlp.shared_expert.down_proj.qweight" not in output_names
        )
        assert "layers.1.mlp.shared_expert.down_proj.qweight" in output_names

    def test_quark_mapper_adds_suffix_remappings(self):
        """get_cache_scale_mapper should add qscales/qqzeros suffix remappings
        for Quark format."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "quant_method": "awq",
            "pack_method": "reorder",
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        assert ".qscales" in mapper.orig_to_new_suffix
        assert mapper.orig_to_new_suffix[".qscales"] == ".scales"
        assert ".qqzeros" in mapper.orig_to_new_suffix
        assert mapper.orig_to_new_suffix[".qqzeros"] == ".qzeros"

    def test_standard_mapper_has_no_quark_suffix_remappings(self):
        """Standard AWQ get_cache_scale_mapper should not add Quark suffix remappings."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "quant_method": "awq",
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        assert ".qscales" not in mapper.orig_to_new_suffix
        assert ".qqzeros" not in mapper.orig_to_new_suffix

    def test_quark_mapper_renames_tensor_names(self):
        """The mapper returned for Quark format should rename qscales/qqzeros
        in a simulated weight stream."""
        import torch
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "quant_method": "awq",
            "pack_method": "reorder",
        }
        awq_config = AutoAWQConfig.from_config(config)
        mapper = awq_config.get_cache_scale_mapper()

        input_weights = [
            ("model.layers.0.mlp.down_proj.qweight", torch.zeros(1)),
            ("model.layers.0.mlp.down_proj.qscales", torch.zeros(1)),
            ("model.layers.0.mlp.down_proj.qqzeros", torch.zeros(1)),
        ]
        output_names = {name for name, _ in mapper.apply(input_weights)}

        assert "model.layers.0.mlp.down_proj.qweight" in output_names
        assert "model.layers.0.mlp.down_proj.scales" in output_names
        assert "model.layers.0.mlp.down_proj.qzeros" in output_names
        assert "model.layers.0.mlp.down_proj.qscales" not in output_names
        assert "model.layers.0.mlp.down_proj.qqzeros" not in output_names


@pytest.mark.skipif(
    not __import__("vllm.platforms", fromlist=["current_platform"]).current_platform.is_cuda_alike(),
    reason="Quark signed INT4 Triton test requires CUDA/ROCm.",
)
def test_quark_signed_int4_triton_matches_torch():
    """Signed INT4 path must only affect Quark exports, not standard AWQ."""
    import os

    import torch
    from safetensors.torch import load_file

    from vllm.model_executor.layers.quantization.awq_triton import (
        awq_dequantize_triton,
    )

    model_dir = os.environ.get("QUARK_AWQ_INT4_TEST_MODEL")
    if model_dir is None:
        pytest.skip("Set QUARK_AWQ_INT4_TEST_MODEL to run this checkpoint test")
    weights_path = f"{model_dir}/model.safetensors"
    if not os.path.isfile(weights_path):
        pytest.skip(f"Quark AWQ INT4 checkpoint not found: {weights_path}")
    weights = load_file(weights_path)
    prefix = "model.layers.0.mlp.up_proj"
    qweight = weights[f"{prefix}.weight"].cuda()
    scales = weights[f"{prefix}.weight_scale"].cuda()
    qzeros = weights[f"{prefix}.weight_zero_point"].cuda()

    group_size = 128
    expected = _dequantize_quark_signed_awq_torch(qweight, scales, qzeros, group_size)
    unsigned = awq_dequantize_triton(qweight, scales, qzeros, signed_int4=False)
    signed = awq_dequantize_triton(qweight, scales, qzeros, signed_int4=True)

    cosine_unsigned = torch.nn.functional.cosine_similarity(
        expected.flatten().float(), unsigned.flatten().float(), dim=0
    )
    assert cosine_unsigned < 0.5
    torch.testing.assert_close(signed, expected, rtol=1e-2, atol=1e-2)
