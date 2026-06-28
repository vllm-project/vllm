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

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.rms_norm_fp8_block_quant import (
    pick_rms_norm_fp8_block_quant_config,
    rms_norm_fp8_block_quant,
    rms_norm_fp8_block_quant_baseline,
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
            "rms_norm_fp8_block_quant", platform
        )
        if len(configs) == 0:
            pytest.skip(
                "Current GPU platform not supported for "
                "rms_norm_fp8_block_quant kernel"
            )

    except (ImportError, RuntimeError, KeyError):
        pytest.skip(
            "Error detecting platform support for rms_norm_fp8_block_quant kernel"
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestRmsNormFp8BlockQuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "hidden_4096_numtokens_32_groupsize_128",
            "hidden_8192_numtokens_32_groupsize_128",
        ]
        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected == "hidden_4096_numtokens_32_groupsize_128"

    def test_config_picker_closest_hidden_size(self):
        config_keys = [
            "hidden_4096_numtokens_32_groupsize_128",
            "hidden_8192_numtokens_32_groupsize_128",
        ]
        # hidden_size=7000, closer to 8192 (diff=1192) than 4096 (diff=2904)
        input_tensor = torch.randn(32, 7000, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(7000, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected == "hidden_8192_numtokens_32_groupsize_128"

    def test_config_picker_numtokens_ceiling(self):
        config_keys = [
            "hidden_4096_numtokens_8_groupsize_128",
            "hidden_4096_numtokens_32_groupsize_128",
            "hidden_4096_numtokens_128_groupsize_128",
        ]
        # 20 tokens -> should pick 32 (smallest >= 20)
        input_tensor = torch.randn(20, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected == "hidden_4096_numtokens_32_groupsize_128"

    def test_config_picker_numtokens_fallback_to_largest(self):
        config_keys = [
            "hidden_4096_numtokens_8_groupsize_128",
            "hidden_4096_numtokens_32_groupsize_128",
            "hidden_4096_numtokens_128_groupsize_128",
        ]
        # 512 tokens -> exceeds all, pick largest (128)
        input_tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected == "hidden_4096_numtokens_128_groupsize_128"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]
        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []
        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        selected = pick_rms_norm_fp8_block_quant_config(args, config_keys)
        assert selected is None

    def test_config_picker_malformed_key_raises(self):
        config_keys = ["hidden_4096_badformat_128"]
        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cpu")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cpu")
        args = (input_tensor, weight, 128, 1e-6)

        with pytest.raises(ValueError, match="Malformed config key"):
            pick_rms_norm_fp8_block_quant_config(args, config_keys)


class TestRmsNormFp8BlockQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
    @pytest.mark.parametrize("hidden_size", [2048, 4096, 8192])
    @pytest.mark.parametrize("group_size", [128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_correctness(self, num_tokens, hidden_size, group_size, dtype):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(
            num_tokens, hidden_size, dtype=dtype, device="cuda"
        )
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")

        ref_out, ref_scales = rms_norm_fp8_block_quant_baseline(
            input_tensor, weight, group_size, 1e-6
        )
        helion_out, helion_scales = rms_norm_fp8_block_quant(
            input_tensor, weight, group_size, 1e-6
        )

        assert helion_out.shape == ref_out.shape
        assert helion_out.dtype == torch.float8_e4m3fn
        assert helion_scales.shape == ref_scales.shape

        torch.testing.assert_close(
            helion_scales, ref_scales, atol=1e-4, rtol=1e-4,
            msg=f"Scale mismatch at tokens={num_tokens}, hidden={hidden_size}",
        )
        torch.testing.assert_close(
            helion_out.to(torch.float32),
            ref_out.to(torch.float32),
            atol=0.05, rtol=0.05,
            msg=f"Output mismatch at tokens={num_tokens}, hidden={hidden_size}",
        )

    def test_output_shape(self):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")

        out, scales = rms_norm_fp8_block_quant(input_tensor, weight, 128, 1e-6)

        assert out.shape == (32, 4096)
        assert scales.shape == (32, 32)  # 4096 // 128 = 32 groups
        assert out.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.float32

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 4096),
            (16, 4096),
            (128, 4096),
            (1, 8192),
            (16, 8192),
        ],
    )
    def test_various_shapes(self, shape):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        weight = torch.ones(shape[-1], dtype=torch.bfloat16, device="cuda")

        out, scales = rms_norm_fp8_block_quant(input_tensor, weight, 128, 1e-6)

        assert out.shape == shape
        assert scales.shape == (shape[0], shape[1] // 128)


class TestRmsNormFp8BlockQuantIntegration:
    def test_kernel_registration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        assert "rms_norm_fp8_block_quant" in registered

        wrapper = registered["rms_norm_fp8_block_quant"]
        assert wrapper.op_name == "rms_norm_fp8_block_quant"
        assert wrapper._config_picker is not None

    def test_input_generator(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        wrapper = registered["rms_norm_fp8_block_quant"]
        inputs = wrapper.get_inputs()

        assert len(inputs) > 0
        for key, args in inputs.items():
            assert len(args) == 4  # input, weight, group_size, epsilon