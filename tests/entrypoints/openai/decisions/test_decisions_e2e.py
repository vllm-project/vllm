# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.benchmarks.lib.endpoint_request_func import (
    DECISIONS_QUESTIONS,
    SYSTEMONE_QUESTIONS,
)
from vllm.benchmarks.serve import add_cli_args
from vllm.benchmarks.serve import main as benchmark
from vllm.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--enable-prefix-caching",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_openai_decisions_text_questions(server):
    response = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": "The package arrived with a broken screen.",
            "questions": [
                {"type": "predicate", "instructions": "Is the screen broken?"},
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which department should handle this?",
                    "choices": [{"value": "returns"}, {"value": "billing"}],
                },
                {
                    "type": "score",
                    "name": "damage",
                    "instructions": "Rate the damage.",
                    "levels": [{"label": "undamaged"}, {"label": "broken"}],
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    predicate, choice_answer, score = body["answers"]
    assert predicate["type"] == "predicate"
    assert predicate["name"] is None
    assert 0 <= predicate["probability"] <= 1
    assert choice_answer["type"] == "choice"
    assert choice_answer["name"] == "department"
    assert [p["value"] for p in choice_answer["probabilities"]] == [
        "returns",
        "billing",
    ]
    assert (
        choice_answer["choice"]
        == max(choice_answer["probabilities"], key=lambda p: p["probability"])["value"]
    )
    assert sum(
        p["probability"] for p in choice_answer["probabilities"]
    ) == pytest.approx(1)
    assert score["type"] == "score"
    assert score["name"] == "damage"
    assert [p["label"] for p in score["probabilities"]] == ["undamaged", "broken"]
    assert score["score"] == pytest.approx(
        sum(p["value"] * p["probability"] for p in score["probabilities"])
    )
    usage = body["usage"]
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_openai_decisions_rejects_images(server):
    response = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                        }
                    ],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this red?"}],
        },
    )
    assert response.status_code == 400
    assert "text input only" in response.json()["error"]["message"]


def test_openai_decisions_predicate_matches_systemone_noul(server):
    text = "The package arrived with a broken screen."
    question = "Is the screen broken?"
    decision = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "input": text,
            "questions": [{"type": "predicate", "instructions": question}],
        },
    )
    noul = requests.post(
        server.url_for("v1/systemone"),
        json={
            "model": MODEL_NAME,
            "state": text,
            "questions": {"0": {"type": "noul", "instructions": question}},
        },
    )
    assert decision.status_code == 200, decision.text
    assert noul.status_code == 200, noul.text
    assert decision.json()["answers"][0]["probability"] == pytest.approx(
        noul.json()["answers"]["0"]["noul"], abs=1e-3
    )


def test_openai_decisions_score_matches_systemone_score(server):
    text = "The data copy finished. Validation and cutover are still outstanding."
    question = {"type": "score", "instructions": "Rate migration progress."}
    levels = [
        {"label": "not started"},
        {"label": "in progress", "description": "work remains"},
        {"label": "complete"},
    ]
    criteria = ["not started", "in progress - work remains", "complete"]
    decision = requests.post(
        server.url_for("v1/decisions"),
        json={
            "model": MODEL_NAME,
            "cache_salt": "decisions-score",
            "input": text,
            "questions": [{**question, "levels": levels}],
        },
    )
    native = requests.post(
        server.url_for("v1/systemone"),
        json={
            "model": MODEL_NAME,
            "cache_salt": "systemone-score",
            "state": text,
            "questions": {"0": {**question, "criteria": criteria}},
        },
    )
    assert decision.status_code == 200, decision.text
    assert native.status_code == 200, native.text
    answer = decision.json()["answers"][0]
    reference = native.json()["answers"]["0"]
    assert [p["label"] for p in answer["probabilities"]] == [v["label"] for v in levels]
    assert [p["probability"] for p in answer["probabilities"]] == pytest.approx(
        [reference["probabilities"][str(i)] for i in range(len(levels))], abs=1e-3
    )


@pytest.mark.parametrize(
    ("backend", "endpoint"),
    [("vllm-systemone", "/v1/systemone"), ("openai-decisions", "/v1/decisions")],
)
@pytest.mark.parametrize(
    ("use_endpoint", "batch_size", "error"),
    [
        pytest.param(False, 1, "URL must end with", id="missing-endpoint"),
        pytest.param(True, 2, "batched prompts", id="batched-prompts"),
    ],
)
def test_decisions_benchmark_rejects_invalid_requests(
    server, backend, endpoint, use_endpoint, batch_size, error
):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--base-url",
        server.url_root,
        "--backend",
        backend,
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-batch-size",
        str(batch_size),
        "--num-prompts",
        "2",
        "--ready-check-timeout-sec",
        "0",
    ]
    if use_endpoint:
        command.extend(["--endpoint", endpoint])
    result = subprocess.run(
        command,
        env={**os.environ, "VLLM_USE_RUST_BENCH": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, result.stdout
    assert error in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("backend", "endpoint", "input_field", "questions"),
    [
        ("vllm-systemone", "/v1/systemone", "state", SYSTEMONE_QUESTIONS),
        ("openai-decisions", "/v1/decisions", "input", DECISIONS_QUESTIONS),
    ],
)
def test_decisions_custom_benchmark_does_not_wrap_prompt(
    server, tmp_path, backend, endpoint, input_field, questions
):
    """Custom benchmarks must use the same tokens as a plain-text request."""
    prompt = "The package arrived with a broken screen."
    dataset = tmp_path / "custom.jsonl"
    dataset.write_text(json.dumps({"prompt": prompt}) + "\n")
    response = requests.post(
        server.url_for(endpoint),
        json={"model": MODEL_NAME, input_field: prompt, "questions": questions},
    )
    assert response.status_code == 200, response.text

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    argv = ["--model", MODEL_NAME, "--base-url", server.url_root]
    argv += ["--backend", backend, "--endpoint", endpoint]
    argv += ["--dataset-name", "custom", "--dataset-path", str(dataset)]
    argv += ["--num-prompts", "1", "--disable-tqdm"]
    result = benchmark(parser.parse_args(argv))

    assert result["completed"] == 1
    assert result["total_input_tokens"] == response.json()["usage"]["input_tokens"]
