# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend priority: Humming outranks Marlin on SM120.

GPU-free. On 99 KB shared-memory GPUs (SM120) the 8-bit Marlin MoE kernel has
no fitting tile for expert shapes whose intermediate size is not a multiple of
128 and fails at warm-up, so MXFP8 MoE backend selection widens the
Humming-preferred set to include SM120 without touching the global default
used by other formats.
"""

import pytest

from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _MXFP8_HUMMING_PREFERRED_CAPABILITIES,
    _SUPPORTED_BACKENDS,
)
from vllm.model_executor.layers.quantization.utils.humming.priority import (
    _HUMMING_PREFERRED_CAPABILITIES,
    prefers_humming,
    prioritize_humming,
)


def _order(compute_capability: int, capabilities: frozenset[int]) -> list[str]:
    return [
        b.name
        for b in prioritize_humming(
            list(_SUPPORTED_BACKENDS),
            compute_capability=compute_capability,
            capabilities=capabilities,
        )
    ]


def test_mxfp8_set_is_a_superset_of_the_global_set():
    assert _HUMMING_PREFERRED_CAPABILITIES <= _MXFP8_HUMMING_PREFERRED_CAPABILITIES
    assert 120 in _MXFP8_HUMMING_PREFERRED_CAPABILITIES


def test_global_default_unchanged_on_sm120():
    """Other kernel families keep their Marlin-first order on SM120."""
    assert not prefers_humming(120)
    assert prioritize_humming(
        list(_SUPPORTED_BACKENDS), compute_capability=120
    ) == list(_SUPPORTED_BACKENDS)


@pytest.mark.parametrize(
    "compute_capability", sorted(_MXFP8_HUMMING_PREFERRED_CAPABILITIES)
)
def test_mxfp8_prefers_humming_over_marlin(compute_capability):
    names = _order(compute_capability, _MXFP8_HUMMING_PREFERRED_CAPABILITIES)
    humming = names.index(Fp8MoeBackend.HUMMING.name)
    assert names[humming + 1] == Fp8MoeBackend.MARLIN.name
    # Everything else keeps its relative order.
    del names[humming]
    assert names == [
        b.name for b in _SUPPORTED_BACKENDS if b is not Fp8MoeBackend.HUMMING
    ]


@pytest.mark.parametrize("compute_capability", [80, 86, 89, 100, 103])
def test_mxfp8_keeps_marlin_first_elsewhere(compute_capability):
    names = _order(compute_capability, _MXFP8_HUMMING_PREFERRED_CAPABILITIES)
    assert names == [b.name for b in _SUPPORTED_BACKENDS]
    assert names.index(Fp8MoeBackend.MARLIN.name) < names.index(
        Fp8MoeBackend.HUMMING.name
    )
