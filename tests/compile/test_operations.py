# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared PyTorch custom operations for compilation tests.

This module provides a centralized place to define and register custom
PyTorch operations used across multiple compilation tests. This avoids
duplicate operation registrations that would cause RuntimeErrors when
running tests together.

The main "attention" operation is automatically registered when this module
is imported. Individual test files can access the global counter functionality
through helper functions.
"""

import torch
from torch.library import Library

from vllm.utils import direct_register_custom_op

# Shared library for all compilation test operations
# Using "silly" namespace to match existing test expectations
silly_lib = Library("silly", "FRAGMENT")


# Global counter that all tests can use or ignore
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
    Unified attention implementation that depends on all inputs and affects the output.
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
)