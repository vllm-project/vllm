# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for zero-overhead NaN/Inf detection in RMSNorm kernels."""

import pytest
import torch

from vllm.model_executor.layers.nan_detector import NaNDetector


@pytest.fixture(autouse=True)
def reset_nan_detector():
    """Reset the singleton between tests."""
    NaNDetector.reset()
    yield
    NaNDetector.reset()


@pytest.fixture
def device():
    return "cuda:0"


@pytest.mark.parametrize("hidden_size", [64, 128, 256])
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@torch.inference_mode()
def test_nan_detection_rms_norm(default_vllm_config, device, hidden_size, num_tokens):
    """NaN in input should be detected at the correct token position."""
    from vllm import _custom_ops as ops

    num_layers = 3
    max_num_tokens = 32

    nan_flags = torch.zeros(num_layers, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    # Clean input — no flags should be set.
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags.sum().item() == 0, "False positive on clean input"

    # Inject NaN at token 1, layer 0.
    nan_flags.zero_()
    x_nan = x.clone()
    if num_tokens > 1:
        x_nan[1, 0] = float("nan")
        ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 0, max_num_tokens)
        assert nan_flags[0, 1].item() == 1, "NaN not detected at token 1"
        assert nan_flags[0, 0].item() == 0, "False positive at token 0"
    else:
        x_nan[0, 0] = float("nan")
        ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 0, max_num_tokens)
        assert nan_flags[0, 0].item() == 1, "NaN not detected at token 0"

    # Inject NaN at a different layer index.
    nan_flags.zero_()
    ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 2, max_num_tokens)
    assert nan_flags[0].sum().item() == 0, "Wrong layer got the flag"
    assert nan_flags[2].any().item(), "NaN not detected at layer 2"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_inf_detection_rms_norm(default_vllm_config, device, hidden_size):
    """Inf in input should be detected."""
    from vllm import _custom_ops as ops

    num_tokens = 4
    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    x[2, 0] = float("inf")
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags[0, 2].item() == 1, "Inf not detected at token 2"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_nan_detection_fused_add_rms_norm(default_vllm_config, device, hidden_size):
    """NaN detection works with the fused add+norm path."""
    from vllm import _custom_ops as ops

    num_tokens = 4
    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn_like(x)

    # Clean — no flags.
    ops.fused_add_rms_norm(
        x.clone(), residual.clone(), weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags.sum().item() == 0

    # Inject NaN in the input (not residual).
    nan_flags.zero_()
    x_nan = x.clone()
    x_nan[3, 0] = float("nan")
    ops.fused_add_rms_norm(
        x_nan, residual.clone(), weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 3].item() == 1, "NaN not detected at token 3"

    # Inject NaN in the residual.
    nan_flags.zero_()
    res_nan = residual.clone()
    res_nan[0, 0] = float("nan")
    ops.fused_add_rms_norm(
        x.clone(), res_nan, weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 0].item() == 1, "NaN in residual not detected"


@torch.inference_mode()
def test_no_detection_when_disabled(default_vllm_config, device):
    """When nan_flags is None, no detection occurs (null pointer path)."""
    from vllm import _custom_ops as ops

    hidden_size = 64
    num_tokens = 4
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    x[0, 0] = float("nan")
    out = torch.empty_like(x)

    # Should not crash — nan_flags=None means no detection.
    ops.rms_norm(out, x, weight, 1e-6)


@torch.inference_mode()
def test_nan_detector_class(default_vllm_config, device):
    """Test the NaNDetector singleton lifecycle."""
    detector = NaNDetector.get()

    # Register layers.
    idx0 = detector.register("layer_0")
    idx1 = detector.register("layer_1")
    assert idx0 == 0
    assert idx1 == 1

    # Finalize.
    max_tokens = 8
    detector.finalize(torch.device(device), max_tokens)
    assert detector.nan_flags is not None
    assert detector.nan_flags.shape == (2, max_tokens)
    assert detector.max_num_tokens == max_tokens

    # Clear + check with no NaN — should log nothing.
    detector.clear()
    detector.check(4)  # 4 real tokens

    # Manually set a flag and check.
    detector.nan_flags[0, 2] = 1
    detector.check(4)  # Should log ERROR for layer_0, token 2

    # Set a flag in padding region.
    detector.clear()
    detector.nan_flags[1, 6] = 1
    detector.check(4)  # Should log WARNING for layer_1 (padding)
