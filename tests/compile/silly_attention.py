# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared PyTorch custom silly attention for compilation tests.
Centralizes custom operation definitions to avoid duplicate registrations.
"""

import torch
from torch.library import Library

from vllm.utils import direct_register_custom_op

# Shared library for all compilation test operations
# Using "silly" namespace to match existing test expectations
# import this file will automatically register
# torch ops for testing (like silly.attention)
silly_lib = Library("silly", "FRAGMENT")

# Global counter that counts the number of times attention is invoked
_global_counter = 0


def get_global_counter():
    """Get the current global counter value"""
    return _global_counter


def reset_global_counter():
    """Reset the global counter to 0"""
    global _global_counter
    _global_counter = 0


def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    """
    Unified attention implementation that depends on
    all inputs and affects the output.
    Always increments a global counter that tests can use or ignore.
    """
    global _global_counter

    # Always increment the global counter
    _global_counter += 1

    # Unified implementation that depends on all inputs
    out.copy_(q + k + v)


def silly_attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         out: torch.Tensor) -> None:
    """Fake implementation for testing"""
    return


# Register the unified attention operation
direct_register_custom_op(
    op_name="attention",
    op_func=silly_attention,
    mutates_args=["out"],
    fake_impl=silly_attention_fake,
    target_lib=silly_lib,
    tags=(torch._C.Tag.cudagraph_unsafe, ),
)
