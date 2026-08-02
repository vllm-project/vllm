# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HTTP-based batch invariance test: send requests to a running
vLLM server and compare BS=1 vs BS=N results (tokens and per-step logprobs).

Environment variables:
  - VLLM_TEST_MODEL: served model name (e.g., Qwen/Qwen3-1.7B / DeepSeek-R1)
  - VLLM_TP_SIZE: tensor parallelism size (e.g., 4)

"""

import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import openai
import pytest
from utils import BACKENDS, TEST_MODEL, _random_prompt, skip_if_not_cuda

from tests.utils import RemoteOpenAIServer


def _request_completion(
    client: openai.OpenAI,
    model: str,
    prompt: Any,
    sp: dict[str, Any],
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)

    for attempt in range(max_retries + 1):
        try:
            completion = client.completions.create(**payload)
            # Convert to plain dict so downstream logic can keep using
            # dict-style access just like with raw HTTP JSON.
            return completion.model_dump()
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"Error: {e}\n")
            return None
    return None


def _extract_tokens_and_logprobs(
    choice: dict[str, Any],
) -> tuple[list[Any], list[float] | None]:
    tokens: list[Any] = []
    token_logprobs: list[float] | None = None
    lp = choice.get("logprobs")
    if lp and isinstance(lp, dict):
        tokens = lp.get("token_ids") or lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs", None)
    return tokens, token_logprobs


def _compare_bs1_vs_bsn_single_process(
    prompts: list[str],
    sp_kwargs: dict[str, Any],
    client: openai.OpenAI,
    model_name: str,
) -> None:
    # BS=1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    for p in prompts:
        resp = _request_completion(client, model_name, p, sp_kwargs)
        if resp is None or not resp.get("choices"):
            raise AssertionError("BS=1 empty/failed response")
        choice = resp["choices"][0]
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(
                "logprobs not returned; ensure server supports 'logprobs'"
            )
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))

    # BS=N
    bsN_tokens_per_prompt: list[list[Any]] = [None] * len(prompts)  # type: ignore[list-item]
    bsN_logprobs_per_prompt: list[list[float] | None] = [None] * len(prompts)
    resp = _request_completion(client, model_name, prompts, sp_kwargs)
    if resp is None or not resp.get("choices"):
        raise AssertionError("BS=N empty/failed batched response")
    choices = resp.get("choices", [])
    if len(choices) != len(prompts):
        raise AssertionError(
            f"BS=N choices length {len(choices)} != num prompts {len(prompts)}"
        )
    for idx, choice in enumerate(choices):
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(f"BS=N missing logprobs for prompt {idx}")
        bsN_tokens_per_prompt[idx] = list(toks)
        bsN_logprobs_per_prompt[idx] = list(lps)

    # compare
    for i, (tokens_bs1, tokens_bsN, logprobs_bs1, logprobs_bsN) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
        )
    ):
        if tokens_bs1 != tokens_bsN:
            raise AssertionError(
                f"Prompt {i} (sampling): Different tokens sampled. "
                f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
            )
        if logprobs_bs1 is None or logprobs_bsN is None:
            raise AssertionError(f"Prompt {i}: Missing logprobs in one of the runs")
        if len(logprobs_bs1) != len(logprobs_bsN):
            raise AssertionError(
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS=N)."
            )
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a != b:
                diff = abs(a - b)
                raise AssertionError(
                    f"Prompt {i} Step {t}: Bitwise mismatch "
                    f"(abs diff={diff:.6e}). "
                    f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
                )


@skip_if_not_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    backend: str,
) -> None:
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))
    prompts_all = [_random_prompt(10, 50) for _ in range(32)]

    sp_kwargs: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 8,
        "seed": 42,
        "logprobs": 5,
    }

    tp_size = os.getenv("VLLM_TP_SIZE", "1")
    server_args: list[str] = [
        "--max-model-len=8192",
        "--max-num-seqs=32",
        f"--attention-backend={backend}",
    ]
    if tp_size:
        server_args += ["-tp", tp_size]

    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()
        _compare_bs1_vs_bsn_single_process(
            prompts=prompts_all,
            sp_kwargs=sp_kwargs,
            client=client,
            model_name=TEST_MODEL,
        )


@skip_if_not_cuda
def test_qwen_gdn_chunked_prefill_is_batch_invariant() -> None:
    if TEST_MODEL != "Qwen/Qwen3.5-0.8B":
        pytest.skip("This regression requires a public Qwen GDN model.")

    server_args = [
        "--max-model-len=512",
        "--max-num-batched-tokens=128",
        "--max-num-seqs=8",
        "--enforce-eager",
    ]
    target_prompt = [100] * 193
    target_sampling: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 8,
        "logprobs": 5,
    }

    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()
        baseline = _request_completion(
            client, TEST_MODEL, target_prompt, target_sampling
        )
        assert baseline is not None

        decode_started = threading.Event()
        load_client = server.get_client()

        def run_decode_load() -> None:
            stream = load_client.completions.create(
                model=TEST_MODEL,
                prompt=[101] * 73,
                temperature=0.0,
                max_tokens=128,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    decode_started.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            load = executor.submit(run_decode_load)
            assert decode_started.wait(timeout=30), "Decode load did not start."
            mixed = _request_completion(
                client, TEST_MODEL, target_prompt, target_sampling
            )
            load.result(timeout=30)

        assert mixed is not None
        baseline_tokens, baseline_logprobs = _extract_tokens_and_logprobs(
            baseline["choices"][0]
        )
        mixed_tokens, mixed_logprobs = _extract_tokens_and_logprobs(mixed["choices"][0])
        assert baseline_tokens == mixed_tokens
        assert baseline_logprobs == mixed_logprobs
