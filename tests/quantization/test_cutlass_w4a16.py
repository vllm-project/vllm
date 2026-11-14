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

if not current_platform.has_device_capability(90) or current_platform.is_rocm():
    pytest.skip(
        "Machete W4A16 requires Hopper (sm_90).",
        allow_module_level=True,
    )

from vllm.model_executor.kernels.linear import (
    HummingLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision import (
    MacheteLinearKernel,
)
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
    CompressedTensorsWNA16,
)
from vllm.scalar_type import scalar_types
from vllm.utils.import_utils import has_humming


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
    )
    kernel = choose_mp_linear_kernel(config)
    assert kernel is MacheteLinearKernel, (
        f"Expected MacheteLinearKernel, got {kernel.__name__}"
    )


@pytest.mark.parametrize(
    "full_shape,part_shape,weight_type,group_size,expected_reason",
    [
        (
            (4096, 4096),
            (4096, 4096),
            scalar_types.float6_e3m2f,
            128,
            "Quant type",
        ),
        ((4096, 4096), (4096, 4096), scalar_types.uint4b8, 32, "Group size"),
    ],
    ids=["unsupported-quant-type", "unsupported-group-size"],
)
def test_machete_rejects_invalid_config(
    full_shape, part_shape, weight_type, group_size, expected_reason
):
    """Verify Machete rejects unsupported configurations."""
    config = MPLinearLayerConfig(
        full_weight_shape=full_shape,
        partition_weight_shape=part_shape,
        act_type=torch.float16,
        weight_type=weight_type,
        group_size=group_size,
        zero_points=False,
    )
    can_impl, reason = MacheteLinearKernel.can_implement(config)
    assert not can_impl
    assert reason is not None
    assert expected_reason in reason


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
    )
    kernel = choose_mp_linear_kernel(config)
    assert kernel is not MacheteLinearKernel, "MacheteLinearKernel should be disabled"


def _make_batch_invariant_gptq_method() -> AutoGPTQLinearMethod:
    method = AutoGPTQLinearMethod(AutoGPTQConfig(4, 128, False, True, False, {}, {}))
    method.create_weights(
        torch.nn.Module(),
        input_size_per_partition=4096,
        output_partition_sizes=[4096],
        input_size=4096,
        output_size=4096,
        params_dtype=torch.float16,
        weight_loader=lambda *args, **kwargs: None,
    )
    return method


@pytest.mark.skipif(not has_humming(), reason="requires humming-kernels")
@pytest.mark.parametrize(
    ("disabled_kernels", "expected_kernel"),
    [
        # Selection resolves to Machete, which is not batch-invariant: keep
        # the dequantize + batch-invariant matmul fallback.
        (["HummingLinearKernel"], None),
        # Selection resolves to Humming, which honors VLLM_BATCH_INVARIANT:
        # run the normal Humming path.
        (["MacheteLinearKernel"], HummingLinearKernel),
    ],
    ids=["non-humming-selection-keeps-fallback", "humming-selection-is-used"],
)
def test_batch_invariant_gptq_follows_humming_selection(
    monkeypatch, dist_init, disabled_kernels, expected_kernel
):
    """Under VLLM_BATCH_INVARIANT, GPTQ linear runs the Humming kernel only
    when kernel selection resolves to it."""
    monkeypatch.setattr("vllm.envs.VLLM_DISABLED_KERNELS", disabled_kernels)
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)

    method = _make_batch_invariant_gptq_method()
    if expected_kernel is None:
        assert method.kernel is None
    else:
        assert isinstance(method.kernel, expected_kernel)


@pytest.mark.skipif(not has_humming(), reason="requires humming-kernels")
def test_batch_invariant_int8_gptq_keeps_fallback(monkeypatch, dist_init):
    """W8A16 GPTQ is outside the validated scope: even when kernel selection
    resolves to Humming, batch-invariant mode must keep the dequant fallback."""
    monkeypatch.setattr("vllm.envs.VLLM_DISABLED_KERNELS", ["MacheteLinearKernel"])
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)

    method = AutoGPTQLinearMethod(AutoGPTQConfig(8, 128, False, True, False, {}, {}))
    method.create_weights(
        torch.nn.Module(),
        input_size_per_partition=4096,
        output_partition_sizes=[4096],
        input_size=4096,
        output_size=4096,
        params_dtype=torch.float16,
        weight_loader=lambda *args, **kwargs: None,
    )
    assert method.kernel is None


@pytest.mark.parametrize(
    "model_name",
    [
        "nm-testing/tinyllama-oneshot-w4a16-channel-v2",
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


def test_w4a16_machete_bfloat16_deterministic(vllm_runner):
    """Verify Machete works with bf16 activations and is deterministic."""
    model_name = "nm-testing/tinyllama-oneshot-w4a16-channel-v2"
    prompt = "The capital of France is"

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

        out1 = llm.generate_greedy(prompt, max_tokens=10)
        out2 = llm.generate_greedy(prompt, max_tokens=10)
        assert out1[0][1] == out2[0][1], (
            f"Non-deterministic: '{out1[0][1]}' vs '{out2[0][1]}'"
        )
