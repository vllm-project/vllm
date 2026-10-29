# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-fused eager Triton kernels for Mistral models on ROCm.

Same kernels and ``VLLM_ML4_FUSIONS`` selection as the NVIDIA package: a
comma-separated subset of :data:`FUSIONS`, ``all`` (default) or ``none``.
"""

import os

from vllm.platforms import current_platform

FUSIONS = (
    "ssmax",
    "mla_rope_ssmax",
    "indexer_q",
    "indexer_k",
    "indexer_q_norm",
    "indexer_k_cache",
)


def fusion_enabled(name: str) -> bool:
    assert name in FUSIONS, name
    if not current_platform.is_rocm():
        return False
    value = os.environ.get("VLLM_ML4_FUSIONS", "all").strip()
    if value == "all":
        return True
    return name in {v.strip() for v in value.split(",")}
