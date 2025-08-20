# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared PyTorch custom operations for compilation tests.

This module provides a centralized place to define and register custom
PyTorch operations used across multiple compilation tests. This avoids
duplicate operation registrations that would cause RuntimeErrors when
running tests together.

The main "attention" operation is automatically registered when this module
is imported. Individual test files can access additional functionality
through helper functions.
"""

import torch
from torch.library import Library

from vllm.utils import direct_register_custom_op

# Shared library for all compilation test operations
# Using "silly" namespace to match existing test expectations
silly_lib = Library("silly", "FRAGMENT")


# Global state for test_simple.py compatibility
_global_counter = 0
_use_counting_mode = False


def get_global_counter():
    """Get the current global counter value (for test_simple.py)"""
    return _global_counter


def reset_global_counter():
    """Reset the global counter to 0 (for test_simple.py)"""
    global _global_counter
    _global_counter = 0


def enable_counting_mode():
    """Enable counting mode for test_simple.py"""
    global _use_counting_mode
    _use_counting_mode = True
    reset_global_counter()


def disable_counting_mode():
    """Disable counting mode"""
    global _use_counting_mode
    _use_counting_mode = False


def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   out: torch.Tensor) -> None:
    """
    Unified attention implementation that can handle both standard and counting modes.
    """
    global _global_counter, _use_counting_mode
    
    if _use_counting_mode:
        # Counting mode for test_simple.py
        _global_counter += 1
        print(f"global_counter={_global_counter}")
        out.copy_(q)
        out[0] += 1
    else:
        # Standard mode for test_multiple_graphs.py and test_toy_llama.py
        out.copy_(q)
        out += k
        out += v


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
)