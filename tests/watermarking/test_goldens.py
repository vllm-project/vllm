# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the watermarking goldens still reproduce."""

import json
from pathlib import Path

import pytest

from tests.watermarking.golden_candidates import (
    DETECTOR_FACTORIES,
    GOLDEN_FLOAT_RTOL,
    REGENERATE_COMMAND,
    WATERMARKING_CANDIDATES,
    GoldenCandidatePayload,
    compare_golden,
    configured_algorithms,
    configured_prfs,
    load_goldens,
)

GOLDENS_PATH = Path(__file__).with_name("watermarking_goldens.json")


@pytest.fixture(scope="module")
def goldens() -> dict[str, GoldenCandidatePayload]:
    return load_goldens(json.loads(GOLDENS_PATH.read_text(encoding="utf-8")))


def test_goldens_cover_every_candidate(
    goldens: dict[str, GoldenCandidatePayload],
):
    candidate_ids = {candidate.id for candidate in WATERMARKING_CANDIDATES}

    assert len(candidate_ids) == len(WATERMARKING_CANDIDATES)
    assert set(goldens) == candidate_ids, f"Run `{REGENERATE_COMMAND}`"
    assert {candidate.scheme for candidate in WATERMARKING_CANDIDATES} == (
        configured_algorithms()
    )
    assert set(DETECTOR_FACTORIES) == configured_algorithms()
    assert {candidate.prf for candidate in WATERMARKING_CANDIDATES} == configured_prfs()


@pytest.mark.parametrize(
    "candidate",
    WATERMARKING_CANDIDATES,
    ids=lambda candidate: candidate.id,
)
def test_watermarking_candidate_golden(
    candidate,
    goldens: dict[str, GoldenCandidatePayload],
):
    differences = compare_golden(candidate, goldens[candidate.id])

    assert not differences, "\n".join(
        [
            f"{candidate.id}: golden drift "
            f"(tolerance {GOLDEN_FLOAT_RTOL:g} on score and p_value)",
            *differences,
            f"Regenerate with `{REGENERATE_COMMAND}`",
        ]
    )
