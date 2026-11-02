# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment switches of the mono decode path (read once at import)."""

import os


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True")


class _Envs:
    # route supported decode steps to the mono kernels
    MONO_ENABLE = _flag("VLLM_ROCM_MINIMAX_M3_MONO")
    # run every sparse layer twice (kernels, then the original modules) and log the
    # per-token differences; the layer-parity self-check, meant for --enforce-eager
    MONO_CHECK = _flag("VLLM_ROCM_MINIMAX_M3_MONO_CHECK")
    # run supported DSpark draft steps (7 speculative tokens) on the drafter kernels;
    # on by default, =0 keeps the stock draft layers
    MONO_DRAFTER = _flag("VLLM_ROCM_MINIMAX_M3_MONO_DRAFTER", "1")


envs = _Envs()
