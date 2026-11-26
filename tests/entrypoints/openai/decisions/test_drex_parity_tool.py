"""Unit tests for the pure parts of the Drex parity harness."""

import pytest

from benchmarks.benchmark_drex_parity import (
    _ece,
    _flip_body,
    _flipped_answer,
    _normalize_vllm,
    _reference_vector,
)


def test_noul_answer_maps_yes_no_onto_true_false():
    answer = {"probabilities": {"yes": 0.7, "no": 0.3}}
    assert _normalize_vllm("noul", answer) == {"false": 0.3, "true": 0.7}


def test_choice_answer_keeps_named_keys():
    answer = {"probabilities": {"a": 0.25, "b": 0.75}}
    assert _normalize_vllm("choice", answer) == {"a": 0.25, "b": 0.75}


def test_reference_vector_follows_meta_keys():
    import torch

    vector = _reference_vector({"keys": ["false", "true"]}, torch.tensor([0.2, 0.8]))
    assert vector == pytest.approx({"false": 0.2, "true": 0.8})


def test_ece_counts_the_gap_between_confidence_and_accuracy():
    rows = [(1.0, True), (1.0, False), (1.0, True), (1.0, False)]
    assert _ece(rows) == pytest.approx(0.5)


def test_ece_of_a_perfect_calibration_is_zero():
    assert _ece([(1.0, True), (1.0, True)]) == 0.0


def test_flip_reverses_choice_and_score_options():
    body = {
        "state": "s",
        "questions": {
            "c": {"type": "choice", "instructions": "", "criteria": {"a": 1, "b": 2}},
            "s": {
                "type": "score",
                "instructions": "",
                "criteria": ["bad", "ok", "good"],
            },
            "n": {"type": "noul", "instructions": ""},
        },
    }
    flipped = _flip_body(body)
    assert list(flipped["questions"]["c"]["criteria"]) == ["b", "a"]
    assert flipped["questions"]["s"]["criteria"] == ["good", "ok", "bad"]
    assert "criteria" not in flipped["questions"]["n"]


def test_flipped_score_answer_maps_back_to_the_original_level():
    answer = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}
    assert _flipped_answer("score", ["bad", "ok", "good"], answer) == "0"


def test_flipped_choice_answer_names_the_same_option():
    answer = {"probabilities": {"a": 0.9, "b": 0.1}}
    assert _flipped_answer("choice", {"a": 1, "b": 2}, answer) == "a"
