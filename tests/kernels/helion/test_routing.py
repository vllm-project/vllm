# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.kernels.helion.ops import import_all_kernels
from vllm.kernels.helion.register import get_registered_kernels
from vllm.kernels.helion.routing import (
    _HELION_TO_NATIVE_OP,
    _HELION_TO_ROCM_AITER_OP,
    _schema_tail,
    build_compiled_helion_op_map,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip("Helion is not installed", allow_module_level=True)

ROUTED_OPS = (
    _HELION_TO_ROCM_AITER_OP if current_platform.is_rocm() else _HELION_TO_NATIVE_OP
)


@pytest.mark.parametrize("name", list(ROUTED_OPS))
def test_rocm_uses_separate_kernel_body(name: str):
    import_all_kernels()
    kernel = get_registered_kernels()[name]
    module_suffix = f"rocm.{name}" if current_platform.is_rocm() else name
    assert kernel.raw_kernel_func.__module__.endswith(module_suffix)


@pytest.mark.parametrize("name", list(ROUTED_OPS))
def test_rocm_uses_aiter_autotune_baseline(name: str):
    import_all_kernels()
    kernel = get_registered_kernels()[name]
    baseline = kernel.helion_settings.autotune_baseline_fn
    if current_platform.is_rocm():
        expected = getattr(torch.ops.vllm, _HELION_TO_ROCM_AITER_OP[name]).default
        assert baseline is expected
    else:
        assert baseline.__name__ == "baseline"


@pytest.mark.parametrize("helion_name,fallback_name", list(ROUTED_OPS.items()))
def test_routed_op_schema_matches_fallback(helion_name: str, fallback_name: str):
    """A routed Helion op must match the platform fallback op exactly."""
    import_all_kernels()
    namespace = torch.ops.vllm if current_platform.is_rocm() else torch.ops._C
    native_packet = getattr(namespace, fallback_name, None)
    helion_packet = getattr(torch.ops.vllm_helion, helion_name, None)
    if native_packet is None or helion_packet is None:
        pytest.skip(f"{fallback_name}/{helion_name} not registered in this build")
    assert native_packet is not None and helion_packet is not None

    assert _schema_tail(native_packet.default) == _schema_tail(helion_packet.default), (
        f"schema mismatch for '{helion_name}': "
        f"fallback={native_packet.default._schema} "
        f"helion={helion_packet.default._schema}"
    )


@pytest.mark.parametrize(
    "name",
    [
        "per_token_group_fp8_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
    ],
)
def test_compiled_route_uses_fallback_then_captures_helion(name: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import_all_kernels()
    op_map = build_compiled_helion_op_map()
    fallback_name = ROUTED_OPS[name]
    namespace = torch.ops.vllm if current_platform.is_rocm() else torch.ops._C
    native_op = getattr(namespace, fallback_name).default
    if native_op not in op_map:
        pytest.skip(f"{name} is not supported on this platform")

    args = list(next(iter(get_registered_kernels()[name].get_inputs().values())))
    if not current_platform.is_rocm() and name == "silu_and_mul_per_block_quant":
        # This is the path emitted by ActivationQuantFusionPass.
        args[4] = None

    routed_op = op_map[native_op]
    if current_platform.is_rocm():
        expected = native_op(*args)
        fallback = routed_op(*args)
        torch.testing.assert_close(fallback[1], expected[1], rtol=0.02, atol=1e-4)
        assert (
            fallback[0].view(torch.uint8).to(torch.int16)
            - expected[0].view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = routed_op(*args)
        graph.replay()
        torch.accelerator.synchronize()

        torch.testing.assert_close(captured[1], expected[1], rtol=0.02, atol=1e-4)
        assert (
            captured[0].view(torch.uint8).to(torch.int16)
            - expected[0].view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1
        return

    expected_args = copy.deepcopy(args)
    fallback_args = copy.deepcopy(args)
    captured_args = copy.deepcopy(args)

    native_op(*expected_args)
    routed_op(*fallback_args)
    for index, schema_arg in enumerate(native_op._schema.arguments):
        if schema_arg.alias_info and schema_arg.alias_info.is_write:
            torch.testing.assert_close(fallback_args[index], expected_args[index])

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed_op(*captured_args)
    graph.replay()
    torch.accelerator.synchronize()

    for index, schema_arg in enumerate(native_op._schema.arguments):
        if schema_arg.alias_info and schema_arg.alias_info.is_write:
            captured = captured_args[index]
            expected = expected_args[index]
            if expected.dtype == current_platform.fp8_dtype():
                max_ulp = (
                    (
                        captured.view(torch.uint8).to(torch.int16)
                        - expected.view(torch.uint8).to(torch.int16)
                    )
                    .abs()
                    .max()
                )
                assert max_ulp <= 1
            else:
                torch.testing.assert_close(
                    captured.float(), expected.float(), rtol=0.1, atol=0.1
                )
