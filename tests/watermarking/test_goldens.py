# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from tests.watermarking.golden_candidates import (
    DETECTOR_FACTORIES,
    WATERMARKING_CANDIDATES,
    GoldenCandidatePayload,
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
    assert set(goldens) == candidate_ids, (
        "Run `python tests/watermarking/generate_goldens.py`"
    )
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
    golden = goldens[candidate.id]

    assert candidate.configuration() == golden["configuration"]
    generated = candidate.generate()
    assert generated == golden["generation"]

    detection = candidate.detect(golden["generation"])
    expected = golden["detection"]
    assert detection.score.hex() == expected["score"]
    assert detection.p_value.hex() == expected["p_value"]
    assert detection.num_scored_tokens == expected["num_scored_tokens"]
    assert detection.is_watermarked == expected["is_watermarked"]
