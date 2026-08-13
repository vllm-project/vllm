# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import lm_eval
import pytest

from .gsm8k_eval import (
    STRICT_MATCH,
    GSM8KResult,
    _score_gsm8k,
    assert_min_accuracy,
    call_vllm_api,
    evaluate_gsm8k_lm_eval,
    result_from_lm_eval,
)


def test_score_gsm8k_returns_normalized_isolated_result():
    result = _score_gsm8k(
        states=["The answer is 12", "no numeric answer"],
        output_tokens=[5, 3],
        labels=[12, 9],
        num_shots=5,
        max_tokens=256,
        latency=2.0,
    )

    assert result == GSM8KResult(
        accuracy=0.5,
        profile="isolated-v1",
        metric="last-number-exact-match",
        num_questions=2,
        num_shots=5,
        max_tokens=256,
        invalid_rate=0.5,
        latency=2.0,
        questions_per_second=1.0,
        total_output_tokens=8,
        tokens_per_second=4.0,
        timestamp=result.timestamp,
    )
    assert result.to_dict()["accuracy"] == 0.5


def test_result_from_lm_eval_identifies_profile_and_metric():
    result = result_from_lm_eval({"results": {"gsm8k": {STRICT_MATCH: 0.75}}})

    assert result.accuracy == 0.75
    assert result.profile == "lm-eval-v3"
    assert result.metric == STRICT_MATCH


def test_result_from_lm_eval_rejects_missing_metric():
    with pytest.raises(ValueError, match="did not return GSM8K metric"):
        result_from_lm_eval({"results": {"gsm8k": {}}})


def test_lm_eval_adapter_selects_gsm8k(monkeypatch: pytest.MonkeyPatch):
    received = {}

    def fake_evaluate(**kwargs):
        received.update(kwargs)
        return {"results": {"gsm8k": {STRICT_MATCH: 0.8}}}

    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_evaluate)

    result = evaluate_gsm8k_lm_eval(
        model="vllm",
        model_args="pretrained=test",
        num_fewshot=5,
    )

    assert result.accuracy == 0.8
    assert received == {
        "model": "vllm",
        "model_args": "pretrained=test",
        "tasks": "gsm8k",
        "num_fewshot": 5,
    }


def test_min_accuracy_is_one_sided():
    improved = GSM8KResult(accuracy=0.9, profile="lm-eval-v3", metric=STRICT_MATCH)
    at_floor = GSM8KResult(accuracy=0.72, profile="lm-eval-v3", metric=STRICT_MATCH)
    regressed = GSM8KResult(accuracy=0.71, profile="lm-eval-v3", metric=STRICT_MATCH)

    assert_min_accuracy(improved, 0.8, tolerance=0.08)
    assert_min_accuracy(at_floor, 0.8, tolerance=0.08)
    with pytest.raises(AssertionError, match="lm-eval-v3/exact_match,strict-match"):
        assert_min_accuracy(regressed, 0.8, tolerance=0.08)


def test_http_errors_fail_evaluation():
    class FailingRequest:
        async def __aenter__(self):
            raise RuntimeError("server unavailable")

        async def __aexit__(self, *args):
            return None

    class FailingSession:
        def post(self, *args, **kwargs):
            return FailingRequest()

    with pytest.raises(RuntimeError, match="server unavailable"):
        asyncio.run(
            call_vllm_api(
                FailingSession(),
                prompt="Question: 1 + 1?\nAnswer:",
                temperature=0.0,
                max_tokens=16,
                url="http://localhost:8000",
            )
        )
