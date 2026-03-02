# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Cutlass W4A16 (Machete) kernel on Hopper.

Verifies that W4A16 quantized models loaded through vllm select the
MacheteLinearKernel on sm_90 GPUs, that weights are correctly repacked,
and that inference produces valid output.

Run `pytest tests/quantization/test_cutlass_w4a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.has_device_capability(90):
    pytest.skip(
        "Machete W4A16 requires Hopper (sm_90).",
        allow_module_level=True,
    )

from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.machete import (  # noqa: E501
    MacheteLinearKernel,
)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
    CompressedTensorsWNA16,
)
from vllm.scalar_type import scalar_types


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.parametrize(
    "act_type,weight_type,group_size,zero_points",
    [
        (torch.float16, scalar_types.uint4b8, 128, False),
        (torch.bfloat16, scalar_types.uint4b8, 128, False),
        (torch.float16, scalar_types.uint4, 128, True),
        (torch.float16, scalar_types.uint4b8, -1, False),
    ],
    ids=[
        "fp16-gptq-g128",
        "bf16-gptq-g128",
        "fp16-awq-g128",
        "fp16-channelwise",
    ],
)
def test_machete_kernel_selected(act_type, weight_type, group_size, zero_points):
    """Verify choose_mp_linear_kernel picks MacheteLinearKernel."""
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        act_type=act_type,
        weight_type=weight_type,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=False,
    )
    kernel = choose_mp_linear_kernel(config)
    assert kernel is MacheteLinearKernel, (
        f"Expected MacheteLinearKernel, got {kernel.__name__}"
    )


def test_machete_rejects_partitioned_g_idx():
    """Verify Machete rejects act reordering with partitioned weights."""
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(2048, 4096),
        act_type=torch.float16,
        weight_type=scalar_types.uint4b8,
        group_size=128,
        zero_points=False,
        has_g_idx=True,
    )
    can_impl, reason = MacheteLinearKernel.can_implement(config)
    assert not can_impl, "Machete should reject partitioned g_idx"
    assert "Act reordering" in reason


def test_machete_rejects_unsupported_quant_type():
    """Verify Machete rejects unsupported weight quantization types."""
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        act_type=torch.float16,
        weight_type=scalar_types.float6_e3m2f,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    can_impl, reason = MacheteLinearKernel.can_implement(config)
    assert not can_impl, "Machete should reject unsupported quant type"
    assert "Quant type" in reason


def test_machete_rejects_unsupported_group_size():
    """Verify Machete rejects unsupported group sizes."""
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        act_type=torch.float16,
        weight_type=scalar_types.uint4b8,
        group_size=32,
        zero_points=False,
        has_g_idx=False,
    )
    can_impl, reason = MacheteLinearKernel.can_implement(config)
    assert not can_impl, "Machete should reject unsupported group size"
    assert "Group size" in reason


def test_kernel_selection_with_disabled_machete(monkeypatch):
    """Verify kernel selection falls back when Machete is disabled."""
    monkeypatch.setattr("vllm.envs.VLLM_DISABLED_KERNELS", ["MacheteLinearKernel"])

    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        act_type=torch.float16,
        weight_type=scalar_types.uint4b8,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    kernel = choose_mp_linear_kernel(config)
    assert kernel is not MacheteLinearKernel, "MacheteLinearKernel should be disabled"


@pytest.mark.parametrize(
    "model_name",
    [
        "nm-testing/tinyllama-oneshot-w4a16-channel-v2",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated-ActOrder",
    ],
)
def test_w4a16_machete_e2e(vllm_runner, model_name):
    """Load a W4A16 model, verify Machete kernel is used, and generate."""
    with vllm_runner(model_name, enforce_eager=True, gpu_memory_utilization=0.5) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensorsWNA16)
            assert isinstance(qkv_proj.scheme.kernel, MacheteLinearKernel), (
                f"Expected MacheteLinearKernel on Hopper, "
                f"got {type(qkv_proj.scheme.kernel).__name__}"
            )

            assert hasattr(qkv_proj, "weight_packed")
            assert hasattr(qkv_proj, "weight_scale")
            assert qkv_proj.weight_packed.dtype == torch.int32

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=10)
        assert output
        assert len(output[0][1]) > 0


def test_w4a16_machete_bfloat16(vllm_runner):
    """Verify Machete works with bf16 activations."""
    model_name = "nm-testing/tinyllama-oneshot-w4a16-channel-v2"
    with vllm_runner(
        model_name,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
    ) as llm:

        def check_kernel_type(model):
            layer = model.model.layers[0]
            scheme = layer.self_attn.qkv_proj.scheme
            assert isinstance(scheme.kernel, MacheteLinearKernel), (
                f"Expected MacheteLinearKernel with bf16, "
                f"got {type(scheme.kernel).__name__}"
            )

        llm.apply_model(check_kernel_type)

        output = llm.generate_greedy("1 2 3 4 5", max_tokens=5)
        assert output


def test_w4a16_machete_deterministic(vllm_runner):
    """Verify Machete produces deterministic output across runs."""
    model_name = "nm-testing/tinyllama-oneshot-w4a16-channel-v2"
    prompt = "The capital of France is"

    with vllm_runner(model_name, enforce_eager=True, gpu_memory_utilization=0.5) as llm:
        out1 = llm.generate_greedy(prompt, max_tokens=10)
        out2 = llm.generate_greedy(prompt, max_tokens=10)
        assert out1[0][1] == out2[0][1], (
            f"Non-deterministic: '{out1[0][1]}' vs '{out2[0][1]}'"
        )
