# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Shared test utilities for vLLM IR op tests.

This module provides:
1. Common test parameters (NUM_TOKENS, COMMON_HIDDEN_SIZES)
2. Generic helpers (clone_args, supported_providers, assert_close)
3. Lowering test helpers (assert_supports_args_returns_bool, assert_op_lowered_to_provider)
"""

import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir.op import IrOp
from vllm.platforms import current_platform

from tests.compile.backend import TestBackend


# ============================================================
# Common test parameters
# ============================================================

NUM_TOKENS = [1, 8, 17, 32, 512, 2048]

COMMON_HIDDEN_SIZES = [
    2048,  # Llama 3.2 1B, Qwen 3 MoE 30B-A3B, Gemma 3n
    4096,  # Llama 3 8B, Qwen 3 8B
    5120,  # Llama 4 Scout 17B-16E
    7168,  # DeepSeek V3
    8192,  # Llama 3 70B
]


# ============================================================
# Generic helpers
# ============================================================


def clone_args(args: tuple) -> tuple:
    """Deep copy args to avoid tensor mutation between test cases."""
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def supported_providers(op: IrOp) -> list[str]:
    """Get list of non-native supported provider names."""
    return [
        name for name, impl in op.impls.items() if name != "native" and impl.supported
    ]


def assert_close(op: IrOp, actual, expected):
    """Assert tensors are close, using op-defined tolerances."""
    if isinstance(actual, torch.Tensor):
        tol = op.get_tolerance(actual.dtype)
        try:
            torch.testing.assert_close(
                actual, expected, atol=tol["atol"], rtol=tol["rtol"]
            )
        except AssertionError as e:
            raise AssertionError(
                f"{e}\n\nTo adjust tolerance, use:\n"
                f"  ir.ops.{op.name}.override_tolerance("
                f"{actual.dtype}, atol=..., rtol=...)"
            ) from None
    elif isinstance(actual, (tuple, list)):
        for a, ex in zip(actual, expected):
            assert_close(op, a, ex)
    else:
        assert actual == expected


# ============================================================
# Lowering test helpers
# ============================================================


def _make_simple_model(op: IrOp, real_args: tuple) -> nn.Module:
    """Create a simple model that calls the op with given arguments."""

    class SimpleModel(nn.Module):
        def forward(self, x):
            return op(*real_args)

    return SimpleModel()


def assert_supports_args_returns_bool(
    op: IrOp, provider: str, **generate_kwargs
) -> None:
    """
    Assert that supports_args returns a proper bool (not SymBool) with unbacked SymInts.

    This is a necessary condition for Dynamo compatibility. If supports_args
    returns a SymBool (e.g., from comparing a dimension to a concrete number),
    it indicates the implementation may specialize on batch size.

    Args:
        op: The IrOp to test.
        provider: The provider/implementation to test.
        **generate_kwargs: Keyword arguments passed to op.generate_symbolic_inputs().

    Raises:
        AssertionError: If supports_args returns a non-bool (SymBool).
    """
    impl = op.impls[provider]
    if not impl.supported:
        return  # Skip unsupported implementations

    fake_args = op.generate_symbolic_inputs(**generate_kwargs)
    result = impl.supports_args(*fake_args)
    assert isinstance(result, bool), (
        f"supports_args for {op.name}/{provider} returned {type(result).__name__}, "
        f"expected bool. This likely means the implementation specializes on "
        f"batch size (e.g., x.size(0) == 8)."
    )


def assert_op_lowered_to_provider(
    op: IrOp,
    provider: str,
    **generate_kwargs,
) -> None:
    """
    Assert that lowering selects the correct implementation provider.

    Args:
        op: The IrOp to test.
        provider: The expected provider after lowering.
        **generate_kwargs: Keyword arguments passed to op.generate_inputs().

    Raises:
        AssertionError: If lowering fails or selected a different provider.
    """
    if not op.impls[provider].supported:
        return  # Skip unsupported implementations

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    torch.set_default_device(current_platform.device_type)

    real_args = op.generate_inputs(**generate_kwargs)

    with (
        op.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        model = _make_simple_model(op, real_args)
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        output = compiled_model(x)

    assert op.name in lowering_pass.selected_impls, (
        f"Op {op.name} was not lowered"
    )
    selected = lowering_pass.selected_impls[op.name]
    assert len(selected) > 0, f"No instances of {op.name} were lowered"
    assert isinstance(output, torch.Tensor)

    # Verify the correct provider was selected
    for node_name, selected_provider in selected.items():
        assert selected_provider == provider, (
            f"Expected {op.name} to be lowered to {provider}, "
            f"but got {selected_provider} for node {node_name}"
        )