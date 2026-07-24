# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the online MXFP4 weight quantization backends.

Each backend (Triton, aiter, FlashInfer, XPU) quantizes a synthetic bf16/fp16
tensor to MXFP4, the result is dequantized back to bf16/fp16, and compared
against the pure-torch reference in `reference_mxfp4.py`.
"""

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    quark_quantize_weight_to_mxfp4,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch


def fix_negative_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize FP4 e2m1 negative-zero codewords (0b1000) to positive zero
    (0b0000) in a packed uint8 tensor (two e2m1 values per byte, low nibble
    first).

    -0.0 and +0.0 dequantize to the same float value, but different MXFP4
    quantization backends disagree on which one they emit for values that
    round to zero magnitude. This normalizes both packed representations so
    tensors from different backends can be compared with `torch.equal`.
    """
    assert tensor.dtype == torch.uint8, (
        f"Expected a torch.uint8 tensor, got {tensor.dtype}"
    )

    low_nibble = tensor & 0x0F
    high_nibble = tensor & 0xF0

    low_nibble = torch.where(
        low_nibble == 0x08, torch.zeros_like(low_nibble), low_nibble
    )
    high_nibble = torch.where(
        high_nibble == 0x80, torch.zeros_like(high_nibble), high_nibble
    )

    return low_nibble | high_nibble


def _skip_reason_if_unavailable(backend: str, dtype: torch.dtype) -> str | None:
    """Return a skip reason if `backend` cannot run on the current host."""
    if backend == "triton":
        if not (current_platform.is_cuda() or current_platform.is_rocm()):
            return "Triton MXFP4 kernel requires a CUDA or ROCm GPU."
        return None
    if backend == "aiter":
        from vllm._aiter_ops import is_aiter_found_and_supported

        if not is_aiter_found_and_supported():
            return "aiter is not available/supported on this platform."
        if dtype != torch.bfloat16:
            return "aiter's dynamic_mxfp4_quant only supports bfloat16 input."
        return None
    if backend == "flashinfer":
        from vllm.utils.flashinfer import has_flashinfer

        if not (current_platform.is_cuda() and has_flashinfer()):
            return "FlashInfer is not available on this platform."
        return None
    if backend == "xpu":
        if not current_platform.is_xpu():
            return "not on XPU platform."
        return None
    if backend == "quark":
        try:
            import quark.torch.kernel.mx  # noqa: F401
        except ImportError:
            return "amd-quark is not installed."
        return None
    raise ValueError(f"Unknown backend {backend}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["triton", "aiter", "flashinfer", "xpu", "quark"])
def test_mxfp4_quantization_correctness(backend: str, dtype: torch.dtype):
    """
    Tests that the different implementations of mxfp4_quantize
    in mxfp4_utils.py all match.
    """
    skip_reason = _skip_reason_if_unavailable(backend, dtype)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(3)

    num_rows = 64
    hidden_size = 32 * 32  # multiple 32-element MXFP4 blocks
    device = current_platform.device_type

    x = (torch.rand(num_rows, hidden_size, dtype=dtype, device=device) - 0.5) * 2
    # Vary the magnitude block-to-block so several scale exponents are
    # exercised, rather than a single one for the whole tensor.
    scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
    for i in range(hidden_size // 32):
        x[:, i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]

    if backend == "triton":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            downcast_to_mxfp,
        )

        x_fp4, x_scale, _ = downcast_to_mxfp(x, axis=-1)
    elif backend == "aiter":
        from vllm.model_executor.layers.quantization.quark.utils import (
            quark_quantize_weight_to_mxfp4,
        )

        x_fp4, x_scale = quark_quantize_weight_to_mxfp4(x)
    elif backend == "flashinfer":
        # TODO: enable this test with flashinfer
        pytest.skip("flashinfer mxfp4 quantization match to reference is untested")
        # from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

        # x_fp4, x_scale = flashinfer_mxfp4_quantize(x, backend="cute-dsl")
    elif backend == "xpu":
        # TODO: enable this test on XPU
        pytest.skip("xpu mxfp4 quantization match to reference is untested")
        # from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        #     xpu_mxfp4_quantize,
        # )

        # x_fp4, x_scale = xpu_mxfp4_quantize(x)
    elif backend == "quark":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            quant_dequant_mxfp4,
        )

        result = quant_dequant_mxfp4(x, scale_calculation_mode="even")
    else:
        raise ValueError(f"Unknown backend {backend}")

    if backend != "quark":
        result = dq_mxfp4_torch(x_fp4, x_scale, x.dtype)
    reference = qdq_mxfp4_torch(x, scale_calculation_mode="even")

    assert torch.equal(result, reference)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only compared against a ROCm/AITER build."
)
@pytest.mark.parametrize("moe_backend", ["aiter", "emulation"])
def test_online_mxfp4_moe_matches_quark(
    moe_backend: str, default_vllm_config, dist_init, monkeypatch
):
    """
    Ensures `Mxfp4OnlineMoEMethod` (online quantization)
    and `QuarkOCP_MX_MoEMethod` (AMD Quark checkpoints) produce the same weights,
    with same MOE backend used.
    """
    if moe_backend == "aiter":
        from vllm._aiter_ops import rocm_aiter_ops

        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()

    default_vllm_config.model_config = ModelConfig()

    num_experts = 4
    hidden_size = 256
    intermediate_size = 256
    device = current_platform.device_type

    def make_layer(prefix: str) -> RoutedExperts:
        runner = FusedMoE(
            num_experts=num_experts,
            top_k=2,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=prefix,
        )
        layer = runner.routed_experts
        layer.moe_config.moe_backend = moe_backend
        return layer

    # `create_weights` implementations use plain `torch.zeros`/`torch.randn`
    # with no explicit device, so without this context they would default
    # to CPU and diverge from the (GPU) tensors produced by
    # `mxfp4_quantize`/`replace_parameter`.
    with torch.device(device):
        checkpoint_layer = make_layer("checkpoint_layer")
        online_layer = make_layer("online_layer")

        w13_weight = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        w2_weight = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
        )

        scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
        for i in range(hidden_size // 32):
            w13_weight[..., i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]
            w2_weight[..., i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]

        # Checkpoint path: pre-quantize the source weights to MXFP4 with the
        # ~same RTN recipe a real Quark checkpoint, then feed the
        # already-packed tensors into `QuarkOCP_MX_MoEMethod`.
        checkpoint_w13, checkpoint_w13_scale = quark_quantize_weight_to_mxfp4(
            w13_weight
        )
        checkpoint_w2, checkpoint_w2_scale = quark_quantize_weight_to_mxfp4(w2_weight)

        checkpoint_method = QuarkOCP_MX_MoEMethod(
            weight_config={"qscheme": "per_group", "dtype": "fp4"},
            input_config={"dtype": "fp4", "is_dynamic": True},
            moe=checkpoint_layer.moe_config,
        )
        checkpoint_method.create_weights(
            layer=checkpoint_layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )
        checkpoint_layer.w13_weight.data.copy_(checkpoint_w13)
        checkpoint_layer.w2_weight.data.copy_(checkpoint_w2)
        checkpoint_layer.w13_weight_scale.data.copy_(checkpoint_w13_scale)
        checkpoint_layer.w2_weight_scale.data.copy_(checkpoint_w2_scale)
        checkpoint_method.process_weights_after_loading(checkpoint_layer)

        # Online path: feed the *same* raw bf16 source weights and let
        # `Mxfp4OnlineMoEMethod` quantize them during
        # `process_weights_after_loading`.
        online_method = Mxfp4OnlineMoEMethod(layer=online_layer)
        online_method.create_weights(
            layer=online_layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )
        replace_parameter(online_layer, "w13_weight", w13_weight.clone())
        replace_parameter(online_layer, "w2_weight", w2_weight.clone())
        online_method.process_weights_after_loading(online_layer)

    assert checkpoint_method.mxfp4_backend == online_method.mxfp4_backend

    for key in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        checkpoint_tensor = getattr(checkpoint_layer, key)
        online_tensor = getattr(online_layer, key)

        checkpoint_tensor = checkpoint_tensor.view(torch.uint8)
        online_tensor = online_tensor.view(torch.uint8)

        # NOTE: AMD Quark checkpoints use exclusively **positive** zeros,
        # while other mxfp4_quantize implementations from mxfp4_utils.py may not.
        if key in ("w13_weight", "w2_weight"):
            checkpoint_tensor = fix_negative_zeros(checkpoint_tensor)
            online_tensor = fix_negative_zeros(online_tensor)

        assert checkpoint_tensor.shape == online_tensor.shape
        assert checkpoint_tensor.dtype == online_tensor.dtype
        num_mismatched = (checkpoint_tensor != online_tensor).sum().item()
        total = checkpoint_tensor.numel()
        print(
            f"{key}: {num_mismatched}/{total} "
            f"({100 * num_mismatched / total:.4f}%) mismatched bytes"
        )

        assert torch.equal(checkpoint_tensor, online_tensor)
