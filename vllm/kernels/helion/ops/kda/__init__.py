# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion attention kernels for Kimi Delta-Attention (KDA).

This package provides Helion-based KDA decode and prefill kernels ported
from SGLang PR #32593.  The kernels are JIT-compiled at first call and
require ``helion >= 1.4.0`` (``pip install vllm[helion]``).

Only the shared constant is importable without helion installed.
Kernel sub-modules (``kda_decode``, ``kda_prefill``, ``kda_replayssm``)
import helion at module level and must NOT be imported eagerly.
"""

# K3 exposes 12 local value heads at TP=8. Lower value-head counts share the
# same small-head decode regime.
KDA_SMALL_VALUE_HEAD_THRESHOLD = 12
