# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-path canary: a prompt that reads its own cached KV must keep its
answer when it shares the step with a longer fresh prompt.

The test needs no stored reference. Each needle prompt reads its cached KV
twice, with the same engine and KV-cache dtype: once without a co-scheduled
prompt (the control), and once in a step shared with a fresh prompt longer
than it, so that max_query_len == max_seq_len in that step. A code that the
control returns but the shared step loses points to the cache-read path, not
to quantization error, which the control has too.

Layouts:
  prefix:  prefix caching on. The needle's first blocks (code included) are
           cached, then the needle runs next to the longer prompt.
  chunked: prefix caching off. A filler prompt takes most of the first step,
           so the needle's last tokens are a continuation chunk in the next
           step, next to the longer prompt.
"""

import pytest
import torch

from vllm import SamplingParams, TokensPrompt

MODEL = "Qwen/Qwen3-0.6B"
CODES = ["7310462", "4825913", "9064157", "2378640"]
QUESTION = " What is the secret code? Answer with the number only."
BUDGET, TAIL = 1024, 16
# needle length -> tokens cached before the shared step (whole blocks)
NEEDLE_LENS = {512: 256, 768: 512}

KV_CACHE_DTYPES = [
    "auto",
    "fp8",
    pytest.param(
        "turboquant_k8v4",
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="TurboQuant's prefill fast path skips cached context (#67789)",
        ),
    ),
]


def _needle_prompt_factory(tok):
    """`prompt(text, question, n_tokens)`: a chat prompt of exactly `n_tokens`
    tokens with `text` at the start and `question` at the end."""
    head, end = tok.apply_chat_template(
        [{"role": "user", "content": "@@"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ).split("@@")
    filler = tok.encode("The grass is green. The sky is blue. " * 300)

    def prompt(text, question, n_tokens):
        pre = tok.encode(head + text)
        post = tok.encode(question + end)
        ids = pre + filler[: n_tokens - len(pre) - len(post)] + post
        assert len(ids) == n_tokens
        return TokensPrompt(prompt_token_ids=ids)

    return prompt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("layout", ["prefix", "chunked"])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
def test_cache_read_keeps_needle(vllm_runner, monkeypatch, kv_cache_dtype, layout):
    # In-process engine so all prompts of one generate call are queued before
    # the first step.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if layout == "prefix":
        engine_args = dict(enable_prefix_caching=True, max_num_batched_tokens=2048)
    else:
        engine_args = dict(
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=BUDGET,
        )
    with vllm_runner(
        MODEL,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=2048,
        enforce_eager=True,
        **engine_args,
    ) as runner:
        llm = runner.get_llm()
        prompt = _needle_prompt_factory(llm.get_tokenizer())
        params = SamplingParams(temperature=0.0, max_tokens=16)
        one = SamplingParams(temperature=0.0, max_tokens=1)

        def generate(prompts, sampling_params):
            outs = llm.generate(prompts, sampling_params, use_tqdm=False)
            return [o.outputs[0].text for o in outs]

        found, lost = 0, []
        for n_tokens, n_cached in NEEDLE_LENS.items():
            longer = prompt("", " Summarize.", n_tokens + 64)
            for code in CODES:
                needle = prompt(f"The secret code is {code}.", QUESTION, n_tokens)
                if layout == "prefix":
                    head = TokensPrompt(
                        prompt_token_ids=needle["prompt_token_ids"][:n_cached]
                    )
                    llm.reset_prefix_cache()
                    generate([head], one)
                    control = generate([needle], params)[0]
                    llm.reset_prefix_cache()
                    generate([head], one)
                    shared = generate([needle, longer], params)[0]
                else:
                    first = prompt("", " Summarize.", BUDGET - (n_tokens - TAIL))
                    control = generate([first, needle], [one, params])[1]
                    shared = generate([first, needle, longer], [one, params, params])[1]
                if code in control:
                    found += 1
                    if code not in shared:
                        lost.append((n_tokens, code, shared))

    total = len(NEEDLE_LENS) * len(CODES)
    if found < total // 2:
        # Not an AssertionError: the xfail above must not absorb this.
        pytest.fail(f"control found only {found}/{total} codes; canary not exercised")
    assert not lost, f"{len(lost)}/{found} codes lost in the shared step: {lost}"
