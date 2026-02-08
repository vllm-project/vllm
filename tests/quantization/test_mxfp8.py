# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 (Microscaling FP8) online quantization.

Run `pytest tests/quantization/test_mxfp8.py --forked`.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.mxfp8 import Mxfp8LinearMethod


def compute_sqnr(output: torch.Tensor, ref_output: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB.

    SQNR = 10 * log10(signal_power / noise_power)

    Args:
        output: Quantized output tensor.
        ref_output: Reference (full precision) output tensor.

    Returns:
        SQNR in dB.
    """
    noise = output - ref_output
    signal_power = (ref_output**2).mean()
    noise_power = (noise**2).mean()
    return 10 * torch.log10(signal_power / noise_power).item()


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="MXFP8 is not supported on this GPU type (requires SM100+/Blackwell).",
)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_load_bf16_model(vllm_runner, monkeypatch, enforce_eager: bool) -> None:
    """Test loading a BF16 model with MXFP8 online quantization."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_COMPILE_CACHE", "1")

    with vllm_runner(
        "facebook/opt-125m",
        quantization="mxfp8",
        enforce_eager=enforce_eager,
        dtype="bfloat16",
    ) as llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, Mxfp8LinearMethod)
            # Weights should be quantized to FP8
            assert fc1.weight.dtype == torch.float8_e4m3fn

        llm.apply_model(check_model)

        # Basic generation test
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="MXFP8 is not supported on this GPU type (requires SM100+/Blackwell).",
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_mxfp8_linear_method_apply(dist_init, use_bias: bool) -> None:
    """Test Mxfp8LinearMethod.apply produces numerically reasonable results."""
    from vllm.model_executor.layers.quantization.mxfp8 import Mxfp8Config
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    config = Mxfp8Config()
    method = Mxfp8LinearMethod(config)

    # Create layer and weights
    # Use sizes that are multiples of 128 for proper blocking
    layer = torch.nn.Module()
    input_size = 256  # Must be multiple of 32 (block size)
    output_size = 128

    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            input_size_per_partition=input_size,
            output_partition_sizes=[output_size],
            input_size=input_size,
            output_size=output_size,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )

        # Initialize weights
        weight_ref = torch.randn(output_size, input_size, dtype=torch.bfloat16)
        layer.weight.data.copy_(weight_ref)

        # Process weights (quantize)
        method.process_weights_after_loading(layer)

        # Create input and optional bias
        x = torch.randn(4, input_size, dtype=torch.bfloat16, device="cuda")
        bias = (
            torch.randn(output_size, dtype=torch.bfloat16, device="cuda")
            if use_bias
            else None
        )

        # Run MXFP8 forward pass
        output = method.apply(layer, x, bias=bias)

        # Reference: BF16 linear
        ref_output = torch.nn.functional.linear(x, weight_ref.cuda(), bias)

        # Check outputs using SQNR (Signal-to-Quantization-Noise Ratio)
        sqnr = compute_sqnr(output, ref_output)
        assert sqnr > 15, f"SQNR {sqnr:.2f} dB is too low (expected > 15 dB)"
