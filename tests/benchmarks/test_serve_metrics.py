# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation SLO accounting and reporting, without a model server."""

import json
from typing import Any

import pytest
from aiohttp import web
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import TokenizersBackend

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import add_cli_args, calculate_metrics, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("thresholds", "passed", "tokens"),
    [
        ({"ttft": 250}, 3, 7),
        ({"tpot": 125}, 3, 6),
        ({"e2el": 375}, 2, 4),
        ({"e2el": 375, "ttft": 250, "tpot": 125}, 2, 4),
    ],
)
def test_slo_attainment_includes_failures_and_intersects_objectives(
    thresholds, passed, tokens
):
    outputs = [
        RequestFuncOutput(success=True, output_tokens=3, ttft=0.125, latency=0.375),
        RequestFuncOutput(success=False, output_tokens=999),
        RequestFuncOutput(success=True, output_tokens=2, ttft=0.5, latency=0.625),
        RequestFuncOutput(success=True, output_tokens=3, ttft=0.125, latency=0.625),
        RequestFuncOutput(success=True, output_tokens=1, ttft=0.25, latency=0.25),
    ]
    metrics, lengths = calculate_metrics([], outputs, 2.0, None, [], thresholds)

    assert lengths == [3, 0, 2, 3, 1]
    assert (metrics.attempted, metrics.completed, metrics.failed) == (5, 4, 1)
    assert metrics.slo_passed == passed
    assert metrics.slo_attainment_pct == 100 * passed / 5
    assert metrics.request_goodput == passed / 2  # Existing request-goodput values.
    assert metrics.output_token_goodput == tokens / 2
    expected_counts = {"ttft": 3, "tpot": 3, "e2el": 2}
    assert metrics.slo_attainment_by_metric == {
        name: {
            "passed": expected_counts[name],
            "attainment_pct": expected_counts[name] * 20,
        }
        for name in thresholds
    }


@pytest.mark.parametrize("output_tokens", [None, 0, 1, 4])
@pytest.mark.parametrize("qualifies", [False, True])
def test_output_goodput_requires_counts_only_for_qualifying_requests(
    output_tokens, qualifies
):
    outputs = [
        RequestFuncOutput(success=True, output_tokens=2, ttft=0.125, latency=0.25),
        RequestFuncOutput(
            success=True,
            output_tokens=output_tokens,
            ttft=0.125 if qualifies else 0.5,
            latency=0.5,
        ),
    ]
    metrics, _ = calculate_metrics([], outputs, 2, None, [], {"ttft": 250})
    assert metrics.request_goodput == (2 if qualifies else 1) / 2
    if qualifies and not output_tokens:
        assert metrics.output_token_goodput is None
    else:
        assert (
            metrics.output_token_goodput
            == (2 + (output_tokens if qualifies else 0)) / 2
        )


@pytest.mark.parametrize(("text", "count"), [("", 0), ("hello", 1), ("hello world", 2)])
def test_output_goodput_uses_tokenizer_fallback(text, count):
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(tokenizer_object=tokenizer)
    output = RequestFuncOutput(
        success=True, generated_text=text, ttft=0.125, latency=0.125
    )
    metrics, lengths = calculate_metrics([], [output], 2, tokenizer, [], {"tpot": 0})
    assert lengths == [count]
    assert metrics.slo_passed == 1
    assert metrics.output_token_goodput == count / 2


def test_unknown_output_length_preserves_zero_tpot_convention():
    output = RequestFuncOutput(success=True, ttft=0.125, latency=1.0)
    metrics, lengths = calculate_metrics([], [output], 2, None, [], {"tpot": 0})
    assert lengths == [1]
    assert metrics.request_goodput == 0.5
    assert metrics.slo_passed == 1
    assert metrics.output_token_goodput is None


@pytest.mark.parametrize("attempted", [0, 2])
def test_no_successful_requests_have_zero_goodput(attempted):
    with pytest.warns(UserWarning, match="All requests failed"):
        metrics, _ = calculate_metrics(
            [], [RequestFuncOutput()] * attempted, 2, None, [], {"ttft": 100}
        )
    pct = 0.0 if attempted else None
    assert metrics.attempted == attempted
    assert metrics.completed == metrics.slo_passed == 0
    assert metrics.failed == attempted
    assert metrics.slo_attainment_pct == pct
    assert metrics.slo_attainment_by_metric == {
        "ttft": {"passed": 0, "attainment_pct": pct}
    }
    assert metrics.request_goodput == metrics.output_token_goodput == 0


@pytest.mark.parametrize("duration", [0, -1])
def test_goodput_requires_positive_duration(duration):
    with pytest.raises(ValueError, match="duration must be greater than zero"):
        calculate_metrics([], [], duration, None, [], {"ttft": 100})


@pytest.mark.parametrize(
    ("goodput", "usage", "threshold"),
    [(True, True, 60000), (True, False, 60000), (True, False, 0), (False, True, 60000)],
)
@pytest.mark.asyncio
async def test_goodput_console_and_saved_json(
    tmp_path, capsys, unused_tcp_port, goodput, usage, threshold
):
    """Exercise argument parsing, HTTP streaming, warmups, and result serialization."""
    received = []

    async def completions(request):
        prompt = (await request.json())["prompt"]
        received.append(prompt)
        if prompt == "fail":
            return web.Response(status=500, text="synthetic failure")
        chunks: list[dict[str, Any]] = [{"choices": [{"text": "hello world"}]}]
        if usage:
            chunks.append({"choices": [], "usage": {"completion_tokens": 4}})
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return web.Response(
            text=body + "data: [DONE]\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        dataset = tmp_path / "requests.jsonl"
        dataset.write_text("\n".join(json.dumps({"prompt": p}) for p in ["ok", "fail"]))
        parser = FlexibleArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model",
                "test-model",
                "--base-url",
                f"http://127.0.0.1:{unused_tcp_port}",
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(dataset),
                "--disable-shuffle",
                "--skip-tokenizer-init",
                "--num-prompts",
                "2",
                "--num-warmups",
                "2",
                "--disable-tqdm",
                "--save-result",
                "--result-dir",
                str(tmp_path),
                "--result-filename",
                "result.json",
            ]
            + (
                ["--goodput", f"ttft:{threshold}", f"tpot:{threshold}"]
                if goodput
                else []
            )
        )
        result = await main_async(args)
    finally:
        await runner.cleanup()

    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved == result
    assert received.count("ok") == 3  # Two warmups and one measured success.
    assert received.count("fail") == 1
    assert (saved["completed"], saved["failed"]) == (1, 1)
    assert "output_lens" not in saved  # Summary fields survive --save-detailed=False.
    console = capsys.readouterr().out
    if not goodput:
        assert saved["request_goodput"] is None
        assert "output_token_goodput" not in saved
        assert "goodput_thresholds_ms" not in saved
        assert "SLO attainment" not in console
        return

    passed = int(threshold > 0)
    assert saved["attempted"] == 2
    assert saved["goodput_thresholds_ms"] == {"ttft": threshold, "tpot": threshold}
    assert saved["slo_passed"] == passed
    assert saved["slo_attainment_pct"] == passed * 50
    assert saved["slo_attainment_by_metric"]["ttft"] == {
        "passed": passed,
        "attainment_pct": passed * 50,
    }
    assert saved["request_goodput"] == pytest.approx(passed / saved["duration"])
    if usage or not passed:
        assert saved["output_token_goodput"] == pytest.approx(
            4 * passed / saved["duration"]
        )
    else:
        assert saved["output_token_goodput"] is None
        assert "N/A (output token counts unavailable)" in console
    console_values = dict(
        line.split(":", 1) for line in console.splitlines() if ":" in line
    )
    assert console_values["Attempted requests"].strip() == "2"
    assert console_values["TTFT SLO passing requests"].strip() == str(passed)
    assert console_values["Combined SLO attainment (%)"].strip() == f"{passed * 50:.2f}"
    assert "Output token goodput (tok/s):" in console
