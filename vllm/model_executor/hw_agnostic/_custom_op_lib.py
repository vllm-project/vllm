# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared torch.library for ops registered from hw-agnostic code.

All ``direct_register_custom_op`` calls under ``model_executor/hw_agnostic``
or ``models/<model>/hw_agnostic`` should pass ``target_lib=vllm_hw_agnostic_lib``.
The ops then live under ``torch.ops.vllm_hw_agnostic.*`` rather than the
generic ``torch.ops.vllm.*`` namespace, making their origin in the
hw-agnostic tree visible at every call site.
"""

import torch

vllm_hw_agnostic_lib = torch.library.Library(  # noqa: TOR901
    "vllm_hw_agnostic", "FRAGMENT"
)

__all__ = ["vllm_hw_agnostic_lib"]
