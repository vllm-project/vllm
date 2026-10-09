# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-specific environment variables.

Merged into :mod:`vllm.envs` at import time, so this module must only depend on
the standard library.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION: bool = False

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Emergency rollback for the NVIDIA MegaMoE path. By default, DeepGEMM
    # computes replicated FP8 shared experts in the same persistent SM100 kernel
    # as the routed FP4 experts.
    "VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION": lambda: bool(
        int(os.getenv("VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION", "0"))
    ),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # Delegate so reads share vllm.envs' cache.
    if name in environment_variables:
        import vllm.envs

        return getattr(vllm.envs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
