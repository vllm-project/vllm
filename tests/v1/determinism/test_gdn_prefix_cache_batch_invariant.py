# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN hybrid: VLLM_BATCH_INVARIANT=1 + prefix caching, same-path bitwise.

Skipped in default CI. Needs a GDN checkpoint (Qwen3.5 / Qwen3-Next):

    export VLLM_BATCH_INVARIANT=1
    export VLLM_GDN_PC_TEST_MODEL=/path/to/Qwen3.5-35B-A3B
    export VLLM_TP_SIZE=2
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    pytest tests/v1/determinism/test_gdn_prefix_cache_batch_invariant.py -s -v

Same-path: both runs hit the same cached prefix (BS=1 vs mixed BS=N).
Cold vs hit is logged only; bf16 SSM cache is not required to match.
"""

from __future__ import annotations

import os
import random

import pytest
from utils import skip_if_not_cuda

from vllm import LLM, SamplingParams

MODEL = os.getenv("VLLM_GDN_PC_TEST_MODEL")

pytestmark = [
    skip_if_not_cuda,
    pytest.mark.skipif(
        not MODEL,
        reason="Set VLLM_GDN_PC_TEST_MODEL to a Qwen3.5/Qwen3-Next GDN checkpoint",
    ),
    pytest.mark.timeout(1800),
]


def _extract(request_output):
    out = request_output.outputs[0]
    token_ids = list(out.token_ids)
    step_logprobs: list[float] = []
    if not out.logprobs:
        return token_ids, None
    for step in out.logprobs:
        chosen = token_ids[len(step_logprobs)]
        lp = step.get(chosen)
        if lp is None:
            return token_ids, None
        step_logprobs.append(float(lp.logprob))
    return token_ids, step_logprobs


def _words(rng: random.Random, n: int) -> str:
    vocab = ["alpha", "beta", "gamma", "delta", "the", "of", "and", "to"]
    return " ".join(rng.choice(vocab) for _ in range(n))


def _prefix_of_n_tokens(tok, rng: random.Random, n_tokens: int, chunk: int) -> str:
    text = _words(rng, max(chunk, n_tokens // 2))
    while len(tok.encode(text)) < n_tokens:
        text += " " + _words(rng, chunk)
    return tok.decode(tok.encode(text)[:n_tokens])


def _run_same_path(
    llm: LLM,
    tok,
    rng: random.Random,
    prefix_text: str,
    label: str,
    suffix: str,
    sampling: SamplingParams,
    warm_sp: SamplingParams,
    batch_size: int,
) -> None:
    needle = prefix_text + suffix
    needle_ids = tok.encode(needle)
    prefix_ids = tok.encode(prefix_text)
    print(
        f"=== {label} prefix_tokens={len(prefix_ids)} "
        f"needle_tokens={len(needle_ids)} ==="
    )

    llm.reset_prefix_cache()
    cold = llm.generate([needle], sampling, use_tqdm=False)[0]
    cold_tokens, cold_lps = _extract(cold)

    llm.reset_prefix_cache()
    llm.generate([{"prompt_token_ids": prefix_ids}], warm_sp, use_tqdm=False)
    hit_bs1 = llm.generate([needle], sampling, use_tqdm=False)[0]
    hit_bs1_tokens, hit_bs1_lps = _extract(hit_bs1)

    llm.reset_prefix_cache()
    llm.generate([{"prompt_token_ids": prefix_ids}], warm_sp, use_tqdm=False)
    prompts = [_words(rng, 24) + suffix for _ in range(batch_size)]
    pos = rng.randrange(batch_size)
    prompts[pos] = needle
    outs = llm.generate(prompts, sampling, use_tqdm=False)
    hit_bsn_tokens, hit_bsn_lps = _extract(outs[pos])

    print(f"cold tokens={cold_tokens}")
    print(f"hit_bs1 tokens={hit_bs1_tokens}")
    print(f"hit_bsN tokens={hit_bsn_tokens}")

    assert hit_bs1_tokens == hit_bsn_tokens and hit_bs1_lps == hit_bsn_lps, (
        f"{label}: same-path hit BS=1 vs BS=N mismatch"
    )
    print(f"PASS {label}: same-path hit BS=1 vs BS=N bitwise")

    if cold_tokens == hit_bs1_tokens and cold_lps == hit_bs1_lps:
        print(f"INFO {label}: cold vs hit also bitwise (fp32-cache-like)")
    else:
        print(
            f"INFO {label}: cold vs hit diverged (expected with default bf16 SSM cache)"
        )


def test_gdn_prefix_cache_same_path_bitwise(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model = MODEL
    assert model is not None
    tp = int(os.getenv("VLLM_TP_SIZE", "2"))
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "4"))
    max_tokens = int(os.getenv("VLLM_NEEDLE_MAX_TOKENS", "8"))
    gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))

    print(
        f"model={model} tp={tp} "
        f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')}"
    )

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        max_num_seqs=max(batch_size, 8),
        gpu_memory_utilization=gpu_mem,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    )
    print("SMOKE: engine started with prefix caching + BI")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=5,
        seed=seed,
    )
    warm_sp = SamplingParams(temperature=0.0, max_tokens=1, seed=seed)
    tok = llm.get_tokenizer()
    rng = random.Random(seed)
    suffix = " Explain batch invariance in one sentence."

    _run_same_path(
        llm,
        tok,
        rng,
        _prefix_of_n_tokens(tok, rng, 1100, 32),
        "prefix-1100",
        suffix,
        sampling,
        warm_sp,
        batch_size,
    )
    _run_same_path(
        llm,
        tok,
        rng,
        _prefix_of_n_tokens(tok, rng, 200, 8),
        "ragged-200",
        suffix,
        sampling,
        warm_sp,
        batch_size,
    )
    print("RESULT: PASS")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
