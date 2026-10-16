# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode/prefill consistency of a Mamba2 model in batch-invariant mode.

Several requests with prompt lengths unaligned to the SSD chunk size decode
concurrently. The raw logprob of every generated token must equal, bit for
bit, the raw logprob of that token in a single-shot prefill of the whole
prompt-plus-generation sequence.
"""

import pytest
import torch
from utils import skip_if_not_cuda

from tests.utils import wait_for_memory_to_settle
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt

MODEL = "AntonV/mamba2-130m-hf"
PROMPT_LENS = (37, 300, 513, 700)
NUM_NEW_TOKENS = 96
GPU_MEMORY_UTILIZATION = 0.4


@pytest.fixture
def mamba2_llm():
    """Build the engine and release its GPU memory before the next test.

    Waiting for garbage collection is not enough: the next engine's memory
    profiling fails when this one's memory is still being released, so the
    engine core is shut down explicitly, as ``VllmRunner.__exit__`` does.
    """
    llms: list[LLM] = []

    def build(max_num_seqs: int, **kwargs) -> LLM:
        llm = LLM(
            model=MODEL,
            # this checkpoint's config has no `architectures` entry
            hf_overrides={"architectures": ["Mamba2ForCausalLM"]},
            runner="generate",
            dtype="bfloat16",
            max_model_len=2048,
            max_num_seqs=max_num_seqs,
            logprobs_mode="raw_logprobs",
            max_logprobs=1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            seed=0,
            **{"enable_prefix_caching": False, **kwargs},
        )
        llms.append(llm)
        return llm

    yield build
    for llm in llms:
        llm.llm_engine.engine_core.shutdown()
    llms.clear()
    cleanup_dist_env_and_memory()
    wait_for_memory_to_settle(threshold_ratio=1.0 - GPU_MEMORY_UTILIZATION)


@skip_if_not_cuda
@pytest.mark.timeout(900)
def test_mamba2_decode_logprobs_match_prefill_logprobs(mamba2_llm):
    llm = mamba2_llm(max_num_seqs=len(PROMPT_LENS))
    gen = torch.Generator().manual_seed(0)
    prompts = [
        torch.randint(100, 50000, (n,), generator=gen).tolist() for n in PROMPT_LENS
    ]

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        SamplingParams(max_tokens=NUM_NEW_TOKENS, temperature=0.0, logprobs=0),
        use_tqdm=False,
    )

    mismatches = []
    for prompt, output in zip(prompts, outputs):
        generated = list(output.outputs[0].token_ids)
        decode_logprobs = output.outputs[0].logprobs
        ref = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt + generated)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
            use_tqdm=False,
        )[0]
        for j, token in enumerate(generated):
            decode_lp = decode_logprobs[j][token].logprob
            prefill_lp = ref.prompt_logprobs[len(prompt) + j][token].logprob
            if decode_lp != prefill_lp:
                mismatches.append((len(prompt), j, decode_lp, prefill_lp))
    assert not mismatches, mismatches[:10]


@skip_if_not_cuda
@pytest.mark.timeout(900)
def test_one_token_prompt_joining_a_decode_batch_matches_prefill_logprobs(
    mamba2_llm,
):
    """A request whose whole prompt is one token arrives while others decode.

    The step is then a uniform one-token batch and runs in a full CUDA graph,
    so the new request must take the decode path from its zeroed state; every
    token it and the running requests generate must still match a single-shot
    prefill bit for bit.
    """
    llm = mamba2_llm(max_num_seqs=4)
    engine = llm.llm_engine
    gen = torch.Generator().manual_seed(1)
    prompts = {
        "long-a": torch.randint(100, 50000, (300,), generator=gen).tolist(),
        "long-b": torch.randint(100, 50000, (513,), generator=gen).tolist(),
        "one-token": torch.randint(100, 50000, (1,), generator=gen).tolist(),
    }
    params = SamplingParams(max_tokens=48, temperature=0.0, logprobs=0)
    for name in ("long-a", "long-b"):
        engine.add_request(name, TokensPrompt(prompt_token_ids=prompts[name]), params)
    finished = {}
    steps = 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = out
        steps += 1
        if steps == 6:
            # both long prompts are decoding: a uniform one-token batch
            engine.add_request(
                "one-token", TokensPrompt(prompt_token_ids=prompts["one-token"]), params
            )
    assert set(finished) == set(prompts)

    mismatches = []
    for name, prompt in prompts.items():
        generated = list(finished[name].outputs[0].token_ids)
        decode_logprobs = finished[name].outputs[0].logprobs
        ref = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt + generated)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
            use_tqdm=False,
        )[0]
        for j, token in enumerate(generated):
            decode_lp = decode_logprobs[j][token].logprob
            prefill_lp = ref.prompt_logprobs[len(prompt) + j][token].logprob
            if decode_lp != prefill_lp:
                mismatches.append((name, j, decode_lp, prefill_lp))
    assert not mismatches, mismatches[:10]


# Mid-chunk block boundaries for the model's 256-token SSD chunk: 400 % 256 = 144
# and 800 % 256 = 32 tokens sit in the partial-chunk buffers there.
BLOCK_SIZE = 400


def _greedy(llm: LLM, prompts: list[list[int]], max_tokens: int):
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=0),
        use_tqdm=False,
    )
    results = []
    for output in outputs:
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        assert completion.logprobs is not None
        logprobs = [lp and lp[t].logprob for lp, t in zip(completion.logprobs, tokens)]
        results.append((tokens, logprobs, output.num_cached_tokens))
    return results


@skip_if_not_cuda
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("model_runner_v2", [False, True])
def test_prefix_cache_hits_match_prefix_caching_off(
    mamba2_llm, monkeypatch, model_runner_v2
):
    """Prefix-cache hits and misses produce the bits of a run without caching.

    A hit resumes from the state cached at a block boundary. Under exact replay
    that state includes the inputs buffered since the last chunk boundary, so
    restoring it without them diverges at every mid-chunk boundary, on hits
    and on misses whose prefill or decode crosses a block.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    gen = torch.Generator().manual_seed(2)

    def rand(n: int) -> list[int]:
        return torch.randint(100, 50000, (n,), generator=gen).tolist()

    max_tokens = 64
    shorter, equal, longer, two_blocks = rand(300), rand(400), rand(700), rand(1000)
    # 350 + 64 generated tokens: the block boundary is crossed while decoding
    crosses_in_decode = rand(350)
    cold = [shorter, equal, longer, two_blocks, crosses_in_decode]

    llm = mamba2_llm(max_num_seqs=16, enable_prefix_caching=True, block_size=BLOCK_SIZE)
    cold_out = _greedy(llm, cold, max_tokens)
    assert [cached for *_, cached in cold_out] == [0] * len(cold)
    warm = [
        (shorter, 0),
        (longer, 400),
        (longer[:600] + rand(50), 400),
        (two_blocks, 800),
    ]
    warm_out = _greedy(llm, [p for p, _ in warm], max_tokens)
    assert [cached for *_, cached in warm_out] == [c for _, c in warm]

    prompts = cold + [p for p, _ in warm]
    ref_out = _greedy(mamba2_llm(max_num_seqs=16), prompts, max_tokens)
    mismatches = [
        (i, len(prompts[i]))
        for i, (out, ref) in enumerate(zip(cold_out + warm_out, ref_out))
        if out[:2] != ref[:2]
    ]
    assert not mismatches, mismatches
