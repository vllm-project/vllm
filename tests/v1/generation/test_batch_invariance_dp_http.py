# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HTTP-based batch invariance test: send requests to a running
vLLM server and compare BS=1 vs BS=N results (tokens and per-step logprobs).

Environment variables:
  - VLLM_API_BASE: base URL like http://127.0.0.1:9256/v1 (default used)
  - VLLM_TEST_MODEL: served model name (e.g., Qwen/Qwen3-1.7B)

Note:
  - The server must be started beforehand via `vllm serve ...` and should
    honor the same sampling params for determinism.
  - For BS=N, we dispatch requests concurrently to encourage server-side
    dynamic batching.
"""

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils import _random_prompt, skip_unsupported


def _post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    import json

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _request_completion(
    api_base: str,
    model: str,
    prompt: str,
    sp: dict[str, Any],
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> dict[str, Any] | None:
    url = api_base.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}

    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)

    for attempt in range(max_retries + 1):
        try:
            return _post_json(url, headers, payload, timeout)
        except HTTPError as e:  # type: ignore[reportGeneralTypeIssues]
            status = getattr(e, "code", None)
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"HTTPError: {e}\n")
            return None
        except URLError as e:  # type: ignore[reportGeneralTypeIssues]
            if attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"URLError: {e}\n")
            return None
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
    api_base: str,
    model_name: str,
) -> None:
    # BS=1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    for p in prompts:
        resp = _request_completion(api_base, model_name, p, sp_kwargs)
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

    # BS=N: dispatch concurrently to encourage server batching
    bsN_tokens_per_prompt: list[list[Any]] = [None] * len(prompts)  # type: ignore[list-item]
    bsN_logprobs_per_prompt: list[list[float] | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(64, max(1, len(prompts)))) as ex:
        futures = {
            ex.submit(_request_completion, api_base, model_name, p, sp_kwargs): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            resp = fut.result()
            if resp is None or not resp.get("choices"):
                raise AssertionError(f"BS=N empty/failed response for prompt {idx}")
            choice = resp["choices"][0]
            toks, lps = _extract_tokens_and_logprobs(choice)
            if lps is None:
                raise AssertionError(f"BS=N missing logprobs for prompt {idx}")
            bsN_tokens_per_prompt[idx] = list(toks)
            bsN_logprobs_per_prompt[idx] = list(lps)

    # Compare
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


@skip_unsupported
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN_dp_http():
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))

    api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:9256/v1")
    model_name = os.getenv("VLLM_TEST_MODEL", "deepseek-ai/DeepSeek-V2-lite")
    num_prompts = int(os.getenv("VLLM_TEST_NUM_PROMPTS", "32"))
    prompts_all = [_random_prompt(10, 50) for _ in range(num_prompts)]

    sp_kwargs: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 8,
        "seed": 42,
        "logprobs": 5,
    }

    _compare_bs1_vs_bsn_single_process(
        prompts=prompts_all,
        sp_kwargs=sp_kwargs,
        api_base=api_base,
        model_name=model_name,
    )
