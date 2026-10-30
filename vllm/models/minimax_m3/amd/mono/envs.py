# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment switches of the mono decode path (read once at import)."""

import os


def _flag(name: str) -> bool:
    return os.getenv(name, "0") in ("1", "true", "True")


class _Envs:
    # route supported decode steps to the mono kernels
    MONO_ENABLE = _flag("VLLM_ROCM_MINIMAX_M3_MONO")
    # run every sparse layer twice (kernels, then the original modules) and log the
    # per-token differences; the layer-parity self-check, meant for --enforce-eager
    MONO_CHECK = _flag("VLLM_ROCM_MINIMAX_M3_MONO_CHECK")


envs = _Envs()
