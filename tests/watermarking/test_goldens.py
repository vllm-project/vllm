# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the watermarking goldens still reproduce; see README.md."""

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from tests.watermarking.golden_candidates import (
    DEDUPLICATION_TWINS,
    DETECTOR_FACTORIES,
    GOLDEN_FLOAT_RTOL,
    MAX_HISTORY_TWINS,
    PROMPT_TWINS,
    SKIP_PARTIAL_TWINS,
    WATERMARK_CONFIG_SPECS,
    WATERMARKING_CANDIDATES,
    GoldenCandidatePayload,
    GoldenFormatError,
    compare_golden,
    configured_algorithm_prf_combinations,
    configured_algorithms,
    load_goldens,
    read_goldens,
    routing_boundary,
    validate_golden_guards,
)
from vllm.config.watermarking import WatermarkConfig

GOLDENS_PATH = Path(__file__).with_name("watermarking_goldens.json")

CANDIDATES_BY_ID = {candidate.id: candidate for candidate in WATERMARKING_CANDIDATES}

NEGATIVE_CANDIDATE_ID = "gumbel-philox-key42-cw4-wrong-key"


@pytest.fixture(scope="module")
def goldens() -> dict[str, GoldenCandidatePayload]:
    return read_goldens(GOLDENS_PATH)


@pytest.fixture(scope="module")
def raw_payload() -> dict:
    return json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))


def test_goldens_cover_every_candidate(
    goldens: dict[str, GoldenCandidatePayload],
):
    candidate_ids = set(CANDIDATES_BY_ID)

    assert len(candidate_ids) == len(WATERMARKING_CANDIDATES)
    assert set(goldens) == candidate_ids, "golden contract does not match candidates"
    candidate_combinations = {
        (candidate.scheme, candidate.prf) for candidate in WATERMARKING_CANDIDATES
    }
    assert candidate_combinations == configured_algorithm_prf_combinations()
    assert set(DETECTOR_FACTORIES) == configured_algorithms()

    twin_ids = {
        candidate_id
        for pairs in (
            MAX_HISTORY_TWINS,
            PROMPT_TWINS,
            SKIP_PARTIAL_TWINS,
            DEDUPLICATION_TWINS,
        )
        for pair in pairs
        for candidate_id in pair
    }
    assert twin_ids <= candidate_ids


def test_golden_fixture_guards(
    goldens: dict[str, GoldenCandidatePayload],
):
    validate_golden_guards(goldens)


def test_resolved_records_every_watermark_config_field():
    assert {field.name for field in dataclasses.fields(WatermarkConfig)} == (
        set(WATERMARK_CONFIG_SPECS) | {"key"}
    )


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
        ]
    )


def test_negative_row_is_not_watermarked(
    goldens: dict[str, GoldenCandidatePayload],
):
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


def _set_schema_version(payload: dict) -> None:
    payload["schema_version"] = 1


def _schema_1_payload(payload: dict) -> None:
    # The shape schema 1 actually had: no environment block.
    payload["schema_version"] = 1
    del payload["environment"]


def _drop_field(payload: dict) -> None:
    del payload["candidates"][NEGATIVE_CANDIDATE_ID]["detection"]["p_value_ratio"]


def _add_field(payload: dict) -> None:
    payload["candidates"][NEGATIVE_CANDIDATE_ID]["trace"]["extra"] = 0


def _bool_as_int(payload: dict) -> None:
    payload["candidates"][NEGATIVE_CANDIDATE_ID]["trace"]["dedup_skips"] = True


def _int_as_float(payload: dict) -> None:
    payload["candidates"][NEGATIVE_CANDIDATE_ID]["detection"]["p_value_ratio"] = 1


def _unparsable_score(payload: dict) -> None:
    payload["candidates"][NEGATIVE_CANDIDATE_ID]["detection"]["score"] = "0xzz"


@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        (_set_schema_version, ["schema_version", "expected 2"]),
        (_schema_1_payload, ["schema_version", "expected 2, got 1"]),
        (_drop_field, ["detection", "missing", "p_value_ratio"]),
        (_add_field, ["trace", "unexpected", "extra"]),
        (_bool_as_int, ["trace.dedup_skips", "expected int, got bool"]),
        (_int_as_float, ["p_value_ratio", "expected float, got int"]),
        (_unparsable_score, ["detection.score", "hexadecimal"]),
    ],
    ids=[
        "schema-version",
        "schema-1-file",
        "missing-field",
        "unexpected-field",
        "bool-as-int",
        "int-as-float",
        "unparsable-score",
    ],
)
def test_loader_rejects_bad_payloads(raw_payload, corrupt, expected):
    payload = copy.deepcopy(raw_payload)
    corrupt(payload)

    with pytest.raises(GoldenFormatError) as error:
        load_goldens(payload, source="goldens")

    message = str(error.value)
    for fragment in expected:
        assert fragment in message
    if corrupt not in (_set_schema_version, _schema_1_payload):
        assert NEGATIVE_CANDIDATE_ID in message


def test_loader_rejects_duplicate_keys(tmp_path):
    duplicated = (
        '{"schema_version": 2, "schema_version": 2, '
        '"environment": {}, "candidates": {}}'
    )
    path = tmp_path / "watermarking_goldens.json"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(GoldenFormatError) as error:
        read_goldens(path)

    message = str(error.value)
    assert str(path) in message
    assert "schema_version" in message
    assert "more than once" in message
