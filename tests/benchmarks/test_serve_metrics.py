# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import (
    _build_generation_result,
    _format_mean_chars_per_token,
    calculate_metrics,
)


class _Tokenizer:
    def __init__(self, token_counts: dict[str, int]):
        self.token_counts = token_counts

    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return SimpleNamespace(input_ids=[0] * self.token_counts.get(text, 0))


def _calculate(outputs: list[RequestFuncOutput], use_tokenizer: bool = True):
    return calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=2.0,
        tokenizer=(
            _Tokenizer(
                {output.generated_text: output.output_tokens for output in outputs}
            )
            if use_tokenizer
            else None
        ),
        selected_percentiles=[50.0],
        goodput_config_dict={},
    )[0]


def test_calculate_metrics_counts_unicode_output_characters():
    multilingual_text = "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u4e16\u754c"
    metrics = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="hello",
                output_tokens=5,
                latency=1.0,
                ttft=0.1,
            ),
            RequestFuncOutput(
                success=True,
                generated_text=multilingual_text,
                output_tokens=4,
                latency=1.0,
                ttft=0.1,
            ),
            RequestFuncOutput(
                success=True,
                generated_text="",
                output_tokens=0,
                latency=1.0,
                ttft=0.1,
            ),
            RequestFuncOutput(
                success=False,
                generated_text="failed response",
                output_tokens=10,
            ),
        ]
    )

    # Python's character iteration counts Unicode code points, not UTF-8 bytes.
    assert metrics.total_output_chars == len("hello") + len(multilingual_text)
    assert metrics.output_char_throughput == pytest.approx(
        metrics.total_output_chars / 2.0
    )
    assert metrics.mean_chars_per_token == pytest.approx(metrics.total_output_chars / 9)
    assert metrics.total_output == 9
    assert metrics.output_throughput == pytest.approx(9 / 2.0)


def test_calculate_metrics_handles_zero_tokens_and_whitespace():
    metrics = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="   ",
                output_tokens=0,
                latency=1.0,
            ),
            RequestFuncOutput(
                success=True,
                generated_text="",
                output_tokens=0,
                latency=1.0,
            ),
        ]
    )

    assert metrics.total_output_chars == 0
    assert metrics.output_char_throughput == 0.0
    assert metrics.mean_chars_per_token == 0.0


def test_measured_zero_tokens_are_distinct_from_unmeasured_tokens():
    measured_zero = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="",
                output_tokens=0,
                latency=1.0,
            )
        ]
    )
    unmeasured = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="hello",
                output_tokens=0,
                latency=1.0,
            )
        ],
        use_tokenizer=False,
    )

    assert measured_zero.total_output == 0
    assert measured_zero.mean_chars_per_token == 0.0
    assert unmeasured.total_output == 1
    assert unmeasured.mean_chars_per_token is None
    assert _format_mean_chars_per_token(measured_zero.mean_chars_per_token) == "0.00"


def test_calculate_metrics_does_not_synthesize_token_ratio_without_usage():
    metrics = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="hello",
                output_tokens=0,
                latency=1.0,
            )
        ],
        use_tokenizer=False,
    )

    assert metrics.total_output_chars == 5
    assert metrics.total_output == 1
    assert metrics.mean_chars_per_token is None
    assert _format_mean_chars_per_token(metrics.mean_chars_per_token) == "N/A"

    result = _build_generation_result(
        metrics,
        benchmark_duration=2.0,
        outputs=[
            RequestFuncOutput(
                success=True,
                generated_text="hello",
                output_tokens=0,
                latency=1.0,
            )
        ],
        actual_output_lens=[1],
        goodput_config_dict={},
    )
    assert result["mean_chars_per_token"] is None
    serialized = json.dumps(result)
    assert '"mean_chars_per_token": null' in serialized


def test_character_metrics_are_json_serializable():
    metrics = _calculate(
        [
            RequestFuncOutput(
                success=True,
                generated_text="\u4f60\u597d",
                output_tokens=2,
                latency=1.0,
            )
        ]
    )

    result = _build_generation_result(
        metrics,
        benchmark_duration=2.0,
        outputs=[
            RequestFuncOutput(
                success=True,
                generated_text="\u4f60\u597d",
                output_tokens=2,
                latency=1.0,
            )
        ],
        actual_output_lens=[2],
        goodput_config_dict={},
    )
    assert result["total_output_chars"] == 2
    assert result["output_char_throughput"] == pytest.approx(1.0)
    assert result["mean_chars_per_token"] == pytest.approx(1.0)
    serialized = json.dumps(result, ensure_ascii=False)
    assert json.loads(serialized) == result
