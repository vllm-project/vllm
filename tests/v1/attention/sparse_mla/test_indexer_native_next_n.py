# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

import pytest

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _paged_mqa_logits_schedule_slots
from vllm.v1.attention.backends.mla import indexer

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)


@pytest.mark.parametrize(
    "family,expected_native",
    [
        # SM90 gained next_n=4 (MTP=3) via 2-CTA multicast, but never 3.
        (9, {1, 2, 4}),
        # SM100 schedules any next_n with multi-atom tiles.
        (10, {1, 2, 3, 4, 5, 8}),
        # SM120 advertises multi-atom too but is unvalidated on hardware, so
        # it stays on the conservative gate. Loosen it only with measurements.
        (12, {1, 2}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, expected_native):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize(
    "cuda,deep_gemm", [(False, True), (True, False), (False, False)]
)
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS
