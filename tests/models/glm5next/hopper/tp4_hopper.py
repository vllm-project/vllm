# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash's specialized decode on four SM90 GPUs (H100/H200), end to
end: the model served at TP4 with MTP5 and a BF16 KV cache (the Hopper recipe),
with `decode_backend=latency_specialized` and without.

- Every planned FULL decode capture (6, 12, 18 and 24 tokens: MTP5 verify steps
  of 1-4 requests) holds the Hopper stages on every rank. The executor logs one
  line per captured size and rank with the Hopper / all calls of each stage.
- Greedy outputs match the stock decode's; where a completion departs from it,
  the stock top-2 logprob margin at that token must be a near-tie.

Run with a checkpoint: `GLM53_MODEL=<dir> pytest -s
tests/models/glm5next/hopper/tp4_hopper.py` (not collected by default). It skips
without four SM90 GPUs or GLM53_MODEL. The helpers also drive the 4xH200
integration run (TPOT and per-step GPU time)."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from collections import defaultdict

import pytest
import regex as re

SWITCH = {"decode_backend": "latency_specialized"}
# MTP5 verify steps of 1-4 requests: the planned sizes among the default
# FULL decode capture sizes.
PLANNED = (6, 12, 18, 24)
# A departure at a token whose stock top-2 margin is below this (in nats) counts
# as a near-tie.
TIE_MARGIN = float(os.environ.get("GLM53_TIE_MARGIN", "0.5"))
PROMPTS = (
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April "
        "and May? Let's think step by step."
    ),
    (
        "Write a Python function that returns the n-th Fibonacci number, then "
        "explain its time complexity."
    ),
    "The capital of France is",
    (
        "List the planets of the solar system in order from the Sun, with one "
        "fact about each."
    ),
    (
        "Translate into German: The weather is nice today, so we will walk to "
        "the park after lunch."
    ),
    "Explain why the sky is blue to a ten-year-old.",
    (
        "A train travels 120 km in 1.5 hours. What is its average speed in km/h? "
        "Show your work."
    ),
    "Summarize the plot of Romeo and Juliet in three sentences.",
)


def serve_command(
    model: str,
    on: bool,
    port: int = 8000,
    tp: int = 4,
    speculative_tokens: int | None = 5,
    language_model_only: bool = True,
    max_model_len: int = 16384,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """`vllm serve` for the Hopper recipe: TP4, MTP5, the default (BF16) KV
    cache and FULL decode graphs, with the specialized decode `on` or off."""
    cmd = ["vllm", "serve", model, "--served-model-name", "glm"]
    if language_model_only:
        cmd.append("--language-model-only")
    cmd += [
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.9",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
    ]
    if speculative_tokens:
        spec = {"method": "mtp", "num_speculative_tokens": speculative_tokens}
        cmd += ["--speculative-config", json.dumps(spec)]
    if on:
        cmd += ["--additional-config", json.dumps(SWITCH)]
    return cmd + list(extra)


def start(cmd: list[str], log_path, env: dict | None = None) -> subprocess.Popen:
    log = open(log_path, "w")  # noqa: SIM115 - closed with the process
    return subprocess.Popen(
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )


def wait_ready(proc: subprocess.Popen, port: int, limit: float) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            url = f"http://127.0.0.1:{port}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def stop(proc: subprocess.Popen, timeout: float = 120) -> None:
    """SIGINT to the server's process group (lets profilers finalize), then
    SIGKILL after `timeout`."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def complete(port: int, prompts, max_tokens: int, logprobs: int = 2) -> list[dict]:
    """Greedy completions, with the top-`logprobs` logprobs of each token
    unless `logprobs` is 0."""
    out = []
    for prompt in prompts:
        body = {
            "model": "glm",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if logprobs:
            body["logprobs"] = logprobs
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            choice = json.load(response)["choices"][0]
        lp = choice.get("logprobs") or {}
        out.append(
            {
                "text": choice["text"],
                "tokens": lp.get("tokens"),
                "top": lp.get("top_logprobs"),
            }
        )
    return out


def compare(stock: list[dict], other: list[dict]) -> list[dict]:
    """Per prompt: whether the completions match, and at the first departing
    token its index and the stock top-2 logprob margin (None without
    logprobs)."""
    rows = []
    for index, (a, b) in enumerate(zip(stock, other)):
        ta, tb = a["tokens"], b["tokens"]
        if ta is None or tb is None:
            same = a["text"] == b["text"]
            rows.append({"prompt": index, "same": same, "at": None, "margin": None})
            continue
        at = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if at is None and len(ta) != len(tb):
            at = min(len(ta), len(tb))
        margin = None
        if at is not None and a["top"] and at < len(a["top"]) and a["top"][at]:
            values = sorted(a["top"][at].values(), reverse=True)
            margin = values[0] - values[1] if len(values) > 1 else None
        rows.append({"prompt": index, "same": at is None, "at": at, "margin": margin})
    return rows


_RANK = re.compile(r"\((?:Worker_)?TP(\d+)[^)]*\)|\(Worker pid=\d+\)")
_CAPTURED = re.compile(
    r"Hopper specialized (decode|draft) captured a (\d+)-token step "
    r"\(q=(\d+)\): (.*)$"
)
_STAGE = re.compile(r"(\w+) (\d+)/(\d+)")


def captured_steps(log_text: str) -> dict:
    """{(kind, tokens, q): {rank: {stage: (hopper calls, all calls)}}} from the
    executor's capture lines (`kind`: decode or draft; rank -1 if the line
    carries no TP rank)."""
    found: dict = defaultdict(dict)
    for line in log_text.splitlines():
        match = _CAPTURED.search(line)
        if match is None:
            continue
        rank_match = _RANK.search(line)
        rank = int(rank_match.group(1)) if rank_match and rank_match.group(1) else -1
        kind, tokens, q, stages = match.groups()
        found[kind, int(tokens), int(q)][rank] = {
            name: (int(h), int(n)) for name, h, n in _STAGE.findall(stages)
        }
    return dict(found)


def path_report(log_text: str, planned=PLANNED, q: int = 6, ranks: int = 4) -> dict:
    """Whether every planned decode size was captured on the executor on every
    rank, and which stages ran on Hopper at each size (Hopper / all calls,
    from rank 0)."""
    steps = captured_steps(log_text)
    problems = []
    table = {}
    for tokens in planned:
        by_rank = steps.get(("decode", tokens, q), {})
        if len(by_rank) < ranks:
            problems.append(f"{tokens} tokens: captured on {len(by_rank)} ranks")
        if len({json.dumps(v, sort_keys=True) for v in by_rank.values()}) > 1:
            problems.append(f"{tokens} tokens: ranks disagree")
        if by_rank:
            table[tokens] = next(iter(by_rank.values()))
    draft = {k: v for k, v in steps.items() if k[0] == "draft"}
    stock_stages = sorted(
        {name for stages in table.values() for name, (h, n) in stages.items() if h < n}
    )
    return {
        "ok": not problems,
        "problems": problems,
        "decode": {str(t): v for t, v in table.items()},
        "draft": {f"{t}x{q}": next(iter(v.values())) for (_, t, q), v in draft.items()},
        "stages_not_all_hopper": stock_stages,
        "built": [
            line.split("Hopper specialized ", 1)[1]
            for line in log_text.splitlines()
            if "Hopper specialized decode at " in line
            or "Hopper specialized draft at " in line
            or ("Hopper specialized decode runs the " in line)
        ][:16],
    }


def _hopper_gpus() -> bool:
    try:
        import torch

        from vllm.platforms import current_platform
    except ImportError:
        return False
    return (
        torch.accelerator.device_count() >= 4
        and current_platform.is_device_capability_family(90)
    )


@pytest.mark.skipif(
    "GLM53_MODEL" not in os.environ or not _hopper_gpus(),
    reason="Requires GLM53_MODEL and four SM90 GPUs",
)
def test_tp4_hopper_matches_stock(tmp_path):
    model = os.environ["GLM53_MODEL"]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    completions, logs = {}, {}
    for on in (False, True):
        log_path = tmp_path / f"server-{'on' if on else 'off'}.log"
        proc = start(serve_command(model, on), log_path, env)
        try:
            assert wait_ready(proc, 8000, 3600), log_path.read_text()[-4000:]
            completions[on] = complete(8000, PROMPTS, 128)
        finally:
            stop(proc)
        logs[on] = log_path.read_text()
    assert not captured_steps(logs[False])
    report = path_report(logs[True])
    print(json.dumps(report, indent=1))
    assert report["ok"], report["problems"]
    rows = compare(completions[False], completions[True])
    print(json.dumps(rows, indent=1))
    for row in rows:
        assert row["same"] or (row["margin"] is not None and row["margin"] < TIE_MARGIN)
