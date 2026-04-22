# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared test utilities for vLLM IR op correctness tests.
"""

import torch

from vllm.ir.op import IrOp

NUM_TOKENS = [1, 8, 17, 32, 512, 2048]
COMMON_HIDDEN_SIZES = [
    2048,  # Llama 3.2 1B, Qwen 3 MoE 30B-A3B, Gemma 3n
    4096,  # Llama 3 8B, Qwen 3 8B
    5120,  # Llama 4 Scout 17B-16E
    7168,  # DeepSeek V3
    8192,  # Llama 3 70B
]


def clone_args(args: tuple) -> tuple:
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def supported_providers(op: IrOp) -> list[str]:
    return [
        name for name, impl in op.impls.items() if name != "native" and impl.supported
    ]


def assert_close(op: IrOp, actual, expected):
    if isinstance(actual, torch.Tensor):
        tol = op.get_tolerance(actual.dtype)
        try:
            torch.testing.assert_close(actual, expected, **tol)
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
