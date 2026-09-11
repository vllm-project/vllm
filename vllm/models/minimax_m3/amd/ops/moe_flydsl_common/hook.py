# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared by the install hooks of the MiniMax-M3 FlyDSL MoE packages."""

from __future__ import annotations


def maybe_run_shared_experts(shared_experts, shared_experts_input) -> None:
    """What the modular kernel does with the layer's unfused shared experts
    (vLLM does not fuse them into the routed experts for ModelOpt MXFP8): run
    them here if the runner left them to the kernel
    (``SharedExpertsOrder.MK_INTERNAL_OVERLAPPED``); in every other order the
    runner has computed them before calling ``apply`` and adds their output
    afterwards, and this call is a no-op."""
    if shared_experts is None:
        return
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
