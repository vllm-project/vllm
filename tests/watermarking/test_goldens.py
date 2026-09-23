# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the watermarking goldens still reproduce; see README.md."""

import dataclasses
from pathlib import Path

import pytest

from tests.watermarking.golden_candidates import (
    DETECTOR_FACTORIES,
    GOLDEN_FLOAT_RTOL,
    WATERMARK_CONFIG_FIELDS,
    WATERMARKING_CANDIDATES,
    compare_golden,
    configured_algorithm_prf_combinations,
    configured_algorithms,
    read_goldens,
    routing_boundary,
    validate_golden_guards,
)
from vllm.config.watermarking import WatermarkConfig

GOLDENS_PATH = Path(__file__).with_name("watermarking_goldens.json")

CANDIDATES_BY_ID = {candidate.id: candidate for candidate in WATERMARKING_CANDIDATES}

NEGATIVE_CANDIDATE_ID = "gumbel-philox-key42-cw4-wrong-key"


@pytest.fixture(scope="module")
def goldens() -> dict[str, dict]:
    return read_goldens(GOLDENS_PATH)


def test_goldens_cover_every_candidate(goldens: dict[str, dict]):
    candidate_ids = set(CANDIDATES_BY_ID)

    assert len(candidate_ids) == len(WATERMARKING_CANDIDATES)
    assert set(goldens) == candidate_ids, "golden contract does not match candidates"
    candidate_combinations = {
        (candidate.scheme, candidate.prf) for candidate in WATERMARKING_CANDIDATES
    }
    assert candidate_combinations == configured_algorithm_prf_combinations()
    assert set(DETECTOR_FACTORIES) == configured_algorithms()


def test_golden_fixture_guards(goldens: dict[str, dict]):
    validate_golden_guards(goldens)


def test_resolved_records_every_watermark_config_field():
    assert {field.name for field in dataclasses.fields(WatermarkConfig)} == (
        set(WATERMARK_CONFIG_FIELDS) | {"key"}
    )


@pytest.mark.parametrize(
    "candidate",
    WATERMARKING_CANDIDATES,
    ids=lambda candidate: candidate.id,
)
def test_watermarking_candidate_golden(candidate, goldens: dict[str, dict]):
    differences = compare_golden(candidate, goldens[candidate.id])

    assert not differences, "\n".join(
        [
            f"{candidate.id}: golden drift "
            f"(tolerance {GOLDEN_FLOAT_RTOL:g} on score and p_value)",
            *differences,
        ]
    )


def test_negative_row_is_not_watermarked(goldens: dict[str, dict]):
    candidate = CANDIDATES_BY_ID[NEGATIVE_CANDIDATE_ID]

    assert candidate.detection_key != candidate.key
    assert goldens[NEGATIVE_CANDIDATE_ID]["detection"]["is_watermarked"] is False


@pytest.mark.parametrize(
    "candidate",
    [
        candidate
        for candidate in WATERMARKING_CANDIDATES
        if candidate.scheme == "dual_key_gumbel"
        and 0 < candidate.scheme_config.generation_alpha < 1
    ],
    ids=lambda candidate: candidate.id,
)
def test_routing_sequence_resolves_alpha(candidate):
    alpha = candidate.scheme_config.generation_alpha
    uniforms = candidate.fixture.routing_uniforms
    boundary = routing_boundary(alpha)
    share = sum(1 for uniform in uniforms if boundary <= uniform) / len(uniforms)

    assert abs(share - alpha) <= 0.03
