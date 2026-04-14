# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `vllm bench serve --eval.*`.

The feature is a thin shim that desugars `--eval.*` into a dataset
selection (`LMEvalDataset` or `GptOssEvalDataset`). The data path,
request loop, and metrics collection are the *same* code that powers
any other `--dataset-name` run, so the parity assertions are:

  1. Accuracy parity vs the canonical eval harness: `bench serve
     --eval` against a live server produces the same gsm8k
     strict-match score as a real `lm_eval` CLI run against the same
     server, when both clients pin to the same concurrency.

  2. Perf parity vs plain `bench serve`: `bench serve --eval`
     throughput on a given prompt set must match what plain
     `bench serve --dataset-name custom` produces on the *same
     prompts*, *same stop sequences*, *same concurrency*. This proves
     the eval path is just `bench serve` with different prompts and
     adds no overhead of its own.
"""

import json
import subprocess

import pytest

from tests.utils import RemoteOpenAIServer, large_gpu_test, multi_gpu_only

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUMMY_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
QWEN_35B_FP8 = "Qwen/Qwen3.5-35B-A3B-FP8"
GPT_OSS_20B = "openai/gpt-oss-20b"

NUM_QUESTIONS = 1024
NUM_FEWSHOT = 5
SCORE_KEY = "exact_match,strict-match"
LLAMA_8B_PARITY_CONCURRENCY = 512
QWEN_35B_FP8_PARITY_CONCURRENCY = 512

# Both paths run greedy with the same seed against the same server, so
# we expect the strict-match scores to be effectively identical.
ACCURACY_ATOL = 0.03
PERF_RTOL = 0.03

_SERVER_ARGS = [
    "--max-model-len",
    "4096",
    "--enforce-eager",
    "--no-enable-prefix-caching",
]
_DUMMY_SERVER_ARGS = _SERVER_ARGS + ["--load-format", "dummy"]


@pytest.fixture(scope="function")
def dummy_server():
    """A small server with dummy weights for tests that exercise the
    CLI plumbing without needing real generations."""
    with RemoteOpenAIServer(DUMMY_MODEL, _DUMMY_SERVER_ARGS) as s:
        yield s


@pytest.fixture(scope="function")
def llama_8b_server():
    with RemoteOpenAIServer(LLAMA_8B, _SERVER_ARGS) as s:
        yield s


@pytest.fixture(scope="function")
def qwen_35b_fp8_server():
    with RemoteOpenAIServer(QWEN_35B_FP8, _SERVER_ARGS) as s:
        yield s


def _run_lm_eval_cli(server, model, tmp_path, *, concurrency: int) -> float:
    """Shell out to `lm_eval` and return the gsm8k strict-match score."""
    out_dir = tmp_path / "lm_eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{server.host}:{server.port}/v1/completions"
    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        "gsm8k",
        "--num_fewshot",
        str(NUM_FEWSHOT),
        "--limit",
        str(NUM_QUESTIONS),
        "--model_args",
        f"model={model},base_url={base_url},tokenized_requests=False,"
        f"num_concurrent={concurrency},max_retries=3",
        "--output_path",
        str(out_dir),
        "--seed",
        "0",
    ]
    print(f"[lm_eval] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"lm_eval CLI failed:\n{result.stderr}"

    json_files = list(out_dir.glob("**/results_*.json"))
    assert json_files, f"No lm_eval results JSON under {out_dir}"
    return float(json.loads(json_files[-1].read_text())["results"]["gsm8k"][SCORE_KEY])


def _run_bench_serve_eval(server, model, out_path, *, max_concurrency: int) -> dict:
    """Run `vllm bench serve --eval.*` and return the merged JSONL record.

    This is the simplest possible invocation of the feature under test:
    point at a server, name a task, set a sample limit, point at an
    output file. `LMEvalDataset` builds the prompts; the rest of the
    pipeline is the same as a vanilla `bench serve` run.
    """
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--seed",
        "0",
        "--max-concurrency",
        str(max_concurrency),
        "--eval.tasks",
        "gsm8k",
        "--eval.num_samples",
        str(NUM_QUESTIONS),
        "--eval.num_fewshot",
        str(NUM_FEWSHOT),
        "--eval.output",
        str(out_path),
    ]
    print(f"[bench serve --eval] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"bench serve --eval failed:\n{result.stderr}"

    record = json.loads(out_path.read_text().splitlines()[-1])
    perf = record["performance"]
    assert perf.get("failed", 0) == 0, (
        f"{perf.get('failed', 0)} requests failed during the eval run"
    )
    return record


def _strict_match_score(record: dict) -> float:
    return float(record["accuracy"]["gsm8k"][SCORE_KEY])


@pytest.mark.benchmark
def test_parse_eval_config():
    """`_parse_eval_config` coerces types, supports the
    `num_samples`/`limit` alias, and validates `backend`."""
    from vllm.entrypoints.cli.benchmark.serve import _parse_eval_config

    cfg = _parse_eval_config(
        {
            "tasks": "gsm8k,hellaswag",
            "limit": 100,
            "num_fewshot": 5,
            "max_tokens": 512,
            "output": "/tmp/out.jsonl",
        }
    )
    assert cfg.backend == "lm_eval"  # default
    assert cfg.tasks == "gsm8k,hellaswag"
    assert cfg.limit == 100.0
    assert cfg.num_fewshot == 5
    assert cfg.max_tokens == 512
    assert cfg.output == "/tmp/out.jsonl"

    minimal = _parse_eval_config({"tasks": "gsm8k"})
    assert minimal.limit is None
    assert minimal.num_samples is None
    assert minimal.max_tokens is None
    assert minimal.reasoning_effort is None

    aliased = _parse_eval_config({"tasks": "gsm8k", "num_samples": 50})
    assert aliased.limit == 50.0

    # `num_samples` wins over `limit` if both are set.
    both = _parse_eval_config({"tasks": "gsm8k", "limit": 99, "num_samples": 7})
    assert both.limit == 7.0

    # gpt_oss backend with reasoning_effort.
    gpt_oss = _parse_eval_config(
        {
            "backend": "gpt_oss",
            "tasks": "gpqa,aime25",
            "num_samples": 10,
            "reasoning_effort": "low",
        }
    )
    assert gpt_oss.backend == "gpt_oss"
    assert gpt_oss.tasks == "gpqa,aime25"
    assert gpt_oss.limit == 10.0
    assert gpt_oss.reasoning_effort == "low"

    # Unknown backend is rejected.
    with pytest.raises(SystemExit, match="--eval.backend"):
        _parse_eval_config({"backend": "bogus", "tasks": "x"})


@pytest.mark.benchmark
def test_gpt_oss_eval_dataset_rejects_healthbench():
    """healthbench is intentionally not supported by the gpt_oss eval
    path because scoring requires a live grader model."""
    from vllm.benchmarks.datasets import GptOssEvalDataset

    with pytest.raises(NotImplementedError, match="healthbench"):
        GptOssEvalDataset._build_gpt_oss_eval("healthbench", num_examples=1)
    with pytest.raises(NotImplementedError, match="healthbench"):
        GptOssEvalDataset._build_gpt_oss_eval("healthbench_hard", num_examples=1)


@pytest.mark.benchmark
def test_gpt_oss_eval_dataset_rejects_unknown_task():
    from vllm.benchmarks.datasets import GptOssEvalDataset

    with pytest.raises(ValueError, match="Unknown gpt_oss task"):
        GptOssEvalDataset._build_gpt_oss_eval("not_a_task", num_examples=1)


@pytest.mark.benchmark
def test_run_eval_requires_tasks():
    from argparse import Namespace

    from vllm.entrypoints.cli.benchmark.serve import _parse_eval_config, _run_eval

    with pytest.raises(SystemExit, match="--eval requires tasks"):
        _run_eval(Namespace(model=LLAMA_8B), _parse_eval_config({"limit": 3}))


@pytest.mark.benchmark
def test_run_eval_requires_model():
    from argparse import Namespace

    from vllm.entrypoints.cli.benchmark.serve import _parse_eval_config, _run_eval

    with pytest.raises(SystemExit, match="--eval requires --model"):
        _run_eval(Namespace(model=None), _parse_eval_config({"tasks": "gsm8k"}))


@pytest.mark.benchmark
def test_eval_not_on_bench_throughput():
    """`--eval` is only registered on `bench serve`."""
    result = subprocess.run(
        ["vllm", "bench", "throughput", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--eval" not in result.stdout


@pytest.mark.benchmark
def test_eval_not_on_bench_latency():
    """`--eval` is only registered on `bench serve`."""
    result = subprocess.run(
        ["vllm", "bench", "latency", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--eval" not in result.stdout


@pytest.mark.benchmark
def test_plain_bench_serve_unaffected(dummy_server):
    """Adding the `--eval.*` arg group must not break a vanilla
    `bench serve` run that does not pass any `--eval` flags."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        DUMMY_MODEL,
        "--host",
        dummy_server.host,
        "--port",
        str(dummy_server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"plain bench serve failed:\n{result.stderr}"


@pytest.mark.slow_test
@pytest.mark.benchmark
def test_eval_matches_lm_eval_llama_8b(llama_8b_server, tmp_path):
    """`bench serve --eval` gsm8k accuracy matches the `lm_eval` CLI."""
    concurrency = LLAMA_8B_PARITY_CONCURRENCY
    baseline = _run_lm_eval_cli(
        llama_8b_server, LLAMA_8B, tmp_path, concurrency=concurrency
    )
    record = _run_bench_serve_eval(
        llama_8b_server,
        LLAMA_8B,
        tmp_path / "results.jsonl",
        max_concurrency=concurrency,
    )
    score = _strict_match_score(record)
    print(
        f"[Llama-8B] lm_eval={baseline:.4f}  bench={score:.4f}  "
        f"drop={baseline - score:+.4f}"
    )
    assert baseline - score <= ACCURACY_ATOL, (
        f"bench serve --eval ({score:.4f}) is below lm_eval baseline "
        f"({baseline:.4f}) by more than {ACCURACY_ATOL}"
    )


@large_gpu_test(min_gb=64)
@pytest.mark.slow_test
@pytest.mark.benchmark
def test_eval_matches_lm_eval_qwen_35b_fp8(qwen_35b_fp8_server, tmp_path):
    """`bench serve --eval` gsm8k accuracy matches the `lm_eval` CLI
    on a quantized FP8 model (concurrency tuned to keep FP8 kernels
    deterministic across runs)."""
    concurrency = QWEN_35B_FP8_PARITY_CONCURRENCY
    baseline = _run_lm_eval_cli(
        qwen_35b_fp8_server, QWEN_35B_FP8, tmp_path, concurrency=concurrency
    )
    record = _run_bench_serve_eval(
        qwen_35b_fp8_server,
        QWEN_35B_FP8,
        tmp_path / "results.jsonl",
        max_concurrency=concurrency,
    )
    score = _strict_match_score(record)
    print(
        f"[Qwen-35B-FP8] lm_eval={baseline:.4f}  bench={score:.4f}  "
        f"drop={baseline - score:+.4f}"
    )
    assert baseline - score <= ACCURACY_ATOL, (
        f"bench serve --eval ({score:.4f}) is below lm_eval baseline "
        f"({baseline:.4f}) by more than {ACCURACY_ATOL}"
    )


@pytest.fixture(scope="function")
def gpt_oss_20b_server(tmp_path_factory):
    """Server fixture for the gpt-oss-20b reasoning model.

    Mirrors `tests/evals/gpt_oss/test_gpqa_correctness.py`: needs
    `--tensor-parallel-size 2`, the tiktoken encoding files
    downloaded locally, and the `TIKTOKEN_ENCODINGS_BASE` env var
    pointing at them so the server can load gpt-oss tokenization.
    """
    import urllib.request

    tiktoken_dir = tmp_path_factory.mktemp("tiktoken")
    for fname, url in (
        (
            "cl100k_base.tiktoken",
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        ),
        (
            "o200k_base.tiktoken",
            "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        ),
    ):
        target = tiktoken_dir / fname
        if not target.exists():
            urllib.request.urlretrieve(url, target)

    server_args = [
        "--tensor-parallel-size",
        "2",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len",
        "65536",
        "--no-enable-prefix-caching",
    ]
    env = {"TIKTOKEN_ENCODINGS_BASE": str(tiktoken_dir)}
    with RemoteOpenAIServer(
        GPT_OSS_20B,
        server_args,
        env_dict=env,
        max_wait_seconds=1800,
    ) as s:
        yield s


@large_gpu_test(min_gb=64)
@multi_gpu_only(num_gpus=2)
@pytest.mark.slow_test
@pytest.mark.benchmark
def test_eval_gpt_oss_basic(gpt_oss_20b_server, tmp_path):
    """`bench serve --eval.backend gpt_oss --eval.tasks basic` runs
    end-to-end against gpt-oss-20b. `basic` has a single trivial
    example so this is the cheapest path through the gpt_oss harness;
    it proves the dispatch, dataset, request loop, and offline scorer
    all wire up correctly without spending GPU hours on gpqa/aime25.
    """
    out = tmp_path / "results.jsonl"
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        GPT_OSS_20B,
        "--host",
        gpt_oss_20b_server.host,
        "--port",
        str(gpt_oss_20b_server.port),
        "--seed",
        "0",
        "--max-concurrency",
        "8",
        "--eval.backend",
        "gpt_oss",
        "--eval.tasks",
        "basic",
        "--eval.num_samples",
        "1",
        "--eval.reasoning_effort",
        "low",
        "--eval.max_tokens",
        "8192",
        "--eval.output",
        str(out),
    ]
    print(f"[bench serve --eval gpt_oss] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, (
        f"bench serve --eval.backend gpt_oss failed:\n{result.stderr}"
    )

    record = json.loads(out.read_text().splitlines()[-1])
    assert record["performance"].get("failed", 0) == 0
    accuracy = record["accuracy"]
    assert "basic" in accuracy, f"missing 'basic' task in accuracy: {accuracy}"
    # `basic` scores 1.0 if the model emits any non-empty text, which
    # any working reasoning model will do.
    assert float(accuracy["basic"]["score"]) == 1.0


def _dump_lm_eval_prompts_as_custom_jsonl(tmp_path, model):
    """Build the exact same gsm8k prompts `LMEvalDataset` would build,
    write them to a `custom`-dataset JSONL, and return `(jsonl_path,
    extra_body)` where `extra_body` is the per-request stop list and
    seed that `LMEvalDataset` would have set on each `SampleRequest`.

    This is what makes the perf parity test fair: plain `bench serve
    --dataset-name custom` then sees the *same* prompts, stop
    sequences, max_tokens, and seed as `bench serve --eval`, so any
    throughput difference is purely from the eval shim plumbing.
    """
    from vllm.benchmarks.datasets import LMEvalDataset
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model)
    dataset = LMEvalDataset(
        tasks="gsm8k",
        model=model,
        num_fewshot=NUM_FEWSHOT,
        limit=NUM_QUESTIONS,
    )
    samples = dataset.sample(tokenizer=tokenizer, num_requests=NUM_QUESTIONS)

    out_path = tmp_path / "gsm8k_prompts.jsonl"
    with out_path.open("w") as fh:
        for s in samples:
            fh.write(
                json.dumps({"prompt": s.prompt, "output_tokens": s.expected_output_len})
                + "\n"
            )

    # gsm8k uses one global stop list across instances, so it's safe to
    # broadcast a single `extra_body` over all custom-dataset requests.
    first = samples[0].extra_body or {}
    return out_path, {
        "stop": first.get("stop", []),
        "seed": first.get("seed", 1234),
    }


def _run_plain_bench_serve(
    server, model, dataset_path, result_dir, *, max_concurrency, extra_body
):
    """Run vanilla `bench serve --dataset-name custom` against `server`.

    Two flags are critical for parity with the eval path:

    - `--skip-chat-template`: `CustomDataset.sample` would otherwise
      wrap each prompt in `apply_chat_template([{"role":"user",...}])`
      which on instruct-tuned models turns the few-shot template into a
      chat turn and inflates output_tokens by ~50%. The eval path sends
      the raw prompt, so we must too.
    - `--disable-shuffle`: `CustomDataset.load_data` shuffles the
      JSONL by default; `LMEvalDataset` returns prompts in length-sorted
      order. Disabling shuffle preserves JSONL order so both runs send
      the same prompts in the same order.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--seed",
        "0",
        "--temperature",
        "0",
        "--max-concurrency",
        str(max_concurrency),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(NUM_QUESTIONS),
        "--skip-chat-template",
        "--disable-shuffle",
        "--extra-body",
        json.dumps(extra_body),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "bench_serve.json",
    ]
    print(f"[bench serve] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"plain bench serve failed:\n{result.stderr}"
    return json.loads((result_dir / "bench_serve.json").read_text())


@pytest.mark.slow_test
@pytest.mark.benchmark
def test_eval_perf_matches_plain_bench_serve(llama_8b_server, tmp_path):
    """`bench serve --eval` throughput must match plain `bench serve`
    on the *same* prompts, *same* stop sequences, *same* concurrency.

    This is the perf parity claim: the eval shim is just a dataset
    selector. Anything beyond a few-percent throughput difference would
    mean the eval path is doing extra work the plain path isn't.
    """
    concurrency = LLAMA_8B_PARITY_CONCURRENCY

    # 1. Build the eval prompts via `LMEvalDataset`, dump them to a
    #    `custom` JSONL, and capture the matching stop list + seed.
    jsonl, extra_body = _dump_lm_eval_prompts_as_custom_jsonl(tmp_path, LLAMA_8B)

    # 2. Plain `bench serve --dataset-name custom` baseline.
    plain = _run_plain_bench_serve(
        llama_8b_server,
        LLAMA_8B,
        jsonl,
        tmp_path / "plain",
        max_concurrency=concurrency,
        extra_body=extra_body,
    )

    # 3. `bench serve --eval` against the same server, same concurrency.
    eval_record = _run_bench_serve_eval(
        llama_8b_server,
        LLAMA_8B,
        tmp_path / "eval.jsonl",
        max_concurrency=concurrency,
    )

    # 4. Compare. `request_throughput` and `output_throughput` should
    #    match within PERF_RTOL.
    eval_perf = eval_record["performance"]
    for metric in ("request_throughput", "output_throughput"):
        eval_v = float(eval_perf[metric])
        plain_v = float(plain[metric])
        rel = abs(eval_v - plain_v) / max(plain_v, 1e-9)
        print(
            f"[Llama-8B] {metric}: plain={plain_v:.2f}  eval={eval_v:.2f}  "
            f"rel_delta={rel:.2%}  (limit: {PERF_RTOL:.0%})"
        )
        assert rel <= PERF_RTOL, (
            f"{metric} from bench serve --eval ({eval_v:.2f}) diverges from "
            f"plain bench serve ({plain_v:.2f}) by {rel:.2%}, "
            f"more than the {PERF_RTOL:.0%} relative tolerance."
        )
