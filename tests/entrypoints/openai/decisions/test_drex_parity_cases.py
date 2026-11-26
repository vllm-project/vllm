"""Checks the shipped Drex parity cases keep the shape the harness needs."""

import json
from pathlib import Path

CASES = Path(__file__).resolve().parents[4] / "benchmarks" / "drex_parity_cases.jsonl"


def _cases():
    with CASES.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_cases_are_jsonl_requests_with_unique_question_ids():
    cases = _cases()
    assert len(cases) >= 10
    for case in cases:
        assert "state" in case
        assert case["questions"]
        ids = list(case["questions"])
        assert len(ids) == len(set(ids))


def test_cases_cover_every_question_type():
    types = {q["type"] for case in _cases() for q in case["questions"].values()}
    assert {"choice", "noul", "score"} <= types


def test_choice_cases_cover_the_observed_option_counts():
    counts = set()
    for case in _cases():
        for question in case["questions"].values():
            if question["type"] != "choice":
                continue
            n = len(question["criteria"])
            assert 1 <= n <= 255
            counts.add(n)
    assert {2, 10, 11, 77, 129, 151} <= counts


def test_score_cases_have_at_least_two_levels_and_noul_has_none():
    for case in _cases():
        for question in case["questions"].values():
            if question["type"] == "score":
                assert len(question["criteria"]) >= 2
            elif question["type"] == "noul":
                assert "criteria" not in question


def test_option_descriptions_stay_strings_or_none():
    # The reference renders structured descriptions; vLLM stringifies them.
    # The parity set keeps descriptions flat so both sides agree.
    for case in _cases():
        for question in case["questions"].values():
            criteria = question.get("criteria", {})
            descriptions = criteria.values() if isinstance(criteria, dict) else criteria
            for description in descriptions:
                assert description is None or isinstance(description, str)


def test_a_case_exercises_unicode_and_nested_state():
    states = [case["state"] for case in _cases()]
    assert any(
        isinstance(s, dict)
        and any(ord(c) > 127 for c in json.dumps(s, ensure_ascii=False))
        for s in states
    )
    assert any(
        isinstance(s, dict) and any(isinstance(v, dict) for v in s.values())
        for s in states
    )
