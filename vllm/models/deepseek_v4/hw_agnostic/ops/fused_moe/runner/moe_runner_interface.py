# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoERunnerInterface — re-exported from upstream.

The upstream-registered ``torch.ops.vllm.moe_forward`` and
``torch.ops.vllm.moe_forward_shared`` ops (which we deliberately do
NOT re-register from our vendored copy — see the comment in the
vendored ``moe_runner.py``) call into upstream's
``get_layer_from_name``, which asserts ``isinstance(layer,
MoERunnerInterface)`` against the **upstream** ABC. The vendored
``MoERunner`` inherits from this re-exported (= identity-shared)
``MoERunnerInterface`` so the assert passes.

Same justification as the
``MoEActivation`` / ``FusedMoEMethodBase`` /
``modular_kernel`` carve-outs: pure ABC; identity-matching
cross-boundary is required.
"""

from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)

__all__ = ["MoERunnerInterface"]
