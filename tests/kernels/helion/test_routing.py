# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.kernels.helion.ops import import_all_kernels
from vllm.kernels.helion.register import get_registered_kernels
from vllm.kernels.helion.routing import (
    _HELION_TO_NATIVE_OP,
    build_compiled_helion_op_map,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip("Helion is not installed", allow_module_level=True)


def _mutation_signature(op: torch._ops.OpOverload) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
        for arg in op._schema.arguments
    )


@pytest.mark.parametrize("helion_name,native_name", list(_HELION_TO_NATIVE_OP.items()))
def test_routed_op_schema_matches_native(helion_name: str, native_name: str):
    """Every routed op must share the native op's argument layout and in-place
    mutation semantics, so HelionFusionRoutingPass can retarget the graph node
    with a plain target swap. This enforces the contract at dev time; the same
    check runs at load time to guard against runtime helion-version skew.
    """
    import_all_kernels()
    native_packet = getattr(torch.ops._C, native_name, None)
    helion_packet = getattr(torch.ops.vllm_helion, helion_name, None)
    if native_packet is None or helion_packet is None:
        pytest.skip(f"{native_name}/{helion_name} not registered in this build")
    assert native_packet is not None and helion_packet is not None

    assert _mutation_signature(native_packet.default) == _mutation_signature(
        helion_packet.default
    ), (
        f"schema mismatch for '{helion_name}': "
        f"native={native_packet.default._schema} "
        f"helion={helion_packet.default._schema}"
    )


@pytest.mark.parametrize(
    "name",
    [
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "fused_qk_norm_rope",
    ],
)
def test_compiled_route_uses_native_then_captures_helion(name: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import_all_kernels()
    op_map = build_compiled_helion_op_map()
    native_op = getattr(torch.ops._C, name).default
    if native_op not in op_map:
        pytest.skip(f"{name} is not supported on this platform")

    args = list(next(iter(get_registered_kernels()[name].get_inputs().values())))
    if name == "silu_and_mul_per_block_quant":
        # This is the path emitted by ActivationQuantFusionPass.
        args[4] = None

    expected_args = copy.deepcopy(args)
    fallback_args = copy.deepcopy(args)
    captured_args = copy.deepcopy(args)
    routed_op = op_map[native_op]

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
            torch.testing.assert_close(
                captured_args[index].float(),
                expected_args[index].float(),
                rtol=0.1,
                atol=0.1,
            )
