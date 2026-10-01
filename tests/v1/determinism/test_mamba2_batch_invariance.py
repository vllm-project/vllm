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

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = "AntonV/mamba2-130m-hf"
PROMPT_LENS = (37, 300, 513, 700)
NUM_NEW_TOKENS = 96


@skip_if_not_cuda
@pytest.mark.timeout(900)
def test_mamba2_decode_logprobs_match_prefill_logprobs():
    llm = LLM(
        model=MODEL,
        # this checkpoint's config has no `architectures` entry
        hf_overrides={"architectures": ["Mamba2ForCausalLM"]},
        runner="generate",
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=len(PROMPT_LENS),
        enable_prefix_caching=False,
        logprobs_mode="raw_logprobs",
        max_logprobs=1,
        gpu_memory_utilization=0.4,
        seed=0,
    )
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
