# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests of `SamplingParams.return_last_hidden_states`."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "facebook/opt-125m"
# Float agreement between outputs that should match: four fp16 ULPs.
FP16_ULPS_4 = dict(rtol=4 * 2**-10, atol=4 * 2**-10)
PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "In a hole in the ground there lived",
]


@pytest.fixture(scope="module")
def llm():
    # raw_logits: the returned "logprobs" are the logits themselves, so they
    # can be checked against the returned hidden states. No prefix caching, so
    # repeated batches run the same forward.
    llm = LLM(
        MODEL,
        enable_return_last_hidden_states=True,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
        max_model_len=128,
        seed=0,
    )
    try:
        yield llm
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def lm_head() -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return model.get_output_embeddings().weight.detach()


def test_one_row_per_generated_token(llm):
    params = SamplingParams(
        max_tokens=5, temperature=0.0, ignore_eos=True, return_last_hidden_states=True
    )
    for output in llm.generate(PROMPTS, params):
        completion = output.outputs[0]
        hidden = completion.last_hidden_states
        assert hidden is not None
        assert hidden.device.type == "cpu"
        assert hidden.dtype == llm.llm_engine.model_config.dtype
        assert hidden.shape == (
            len(completion.token_ids),
            llm.llm_engine.model_config.get_hidden_size(),
        )


def test_rows_follow_stops_and_parallel_samples(llm):
    params = SamplingParams(
        n=2,
        max_tokens=8,
        temperature=1.0,
        seed=1,
        stop_token_ids=[4],  # "." in the OPT vocabulary
        return_last_hidden_states=True,
    )
    for output in llm.generate(PROMPTS, params):
        for completion in output.outputs:
            hidden = completion.last_hidden_states
            assert hidden is not None
            assert hidden.shape[0] == len(completion.token_ids)


def test_rows_reproduce_the_engine_logits(llm, lm_head):
    """Each row times the output embedding gives the logits the engine sampled
    from (up to the precision of the engine's own matmul)."""
    params = SamplingParams(
        max_tokens=4,
        temperature=0.0,
        ignore_eos=True,
        logprobs=5,
        return_last_hidden_states=True,
    )
    for output in llm.generate(PROMPTS, params):
        completion = output.outputs[0]
        logits = completion.last_hidden_states.float() @ lm_head.T
        for step, top in enumerate(completion.logprobs):
            # The recomputed maximum is among the engine's top logits (a float32
            # recomputation may order near-ties differently).
            assert int(logits[step].argmax()) in top
            engine = torch.tensor([top[t].logprob for t in top])
            mine = logits[step, list(top)]
            torch.testing.assert_close(mine, engine, rtol=1e-2, atol=5e-2)


def test_asking_changes_no_other_output(llm):
    """Asking for the hidden states changes no other output, and the rows do not
    depend on which other requests asked; requests that did not ask get None.

    Token ids are compared exactly; logits and rows within four fp16 ULPs
    (``FP16_ULPS_4``), both within one batch (the same prompt asked for twice
    and not asked for once) and across runs of the same batch asked for by
    some, all or none of its requests. Not bit for bit, because neither holds
    on every GPU regardless of who asks: with CUDA graphs, an identical batch
    repeated moved one row by 3 ULPs between runs on an RTX PRO 6000 Max-Q
    (eager runs were bit-identical), and on an RTX 4090 one of three identical
    sequences in a batch differed from the other two by one fp16 step in its
    logits and rows in 1 of 30 batches, with all three asking.
    """
    greedy = dict(max_tokens=6, temperature=0.0, ignore_eos=True, logprobs=3)
    asked = SamplingParams(**greedy, return_last_hidden_states=True)
    plain = SamplingParams(**greedy)

    def assert_logits_close(a, b):
        assert a.token_ids == b.token_ids
        for step, (x, y) in enumerate(zip(a.logprobs, b.logprobs)):
            for token in x.keys() & y.keys() | {a.token_ids[step]}:
                torch.testing.assert_close(
                    torch.tensor(x[token].logprob),
                    torch.tensor(y[token].logprob),
                    **FP16_ULPS_4,
                )

    first, second, unasked = (
        output.outputs[0]
        for output in llm.generate([PROMPTS[0]] * 3, [asked, asked, plain])
    )
    assert_logits_close(first, second)
    assert_logits_close(first, unasked)
    torch.testing.assert_close(
        first.last_hidden_states, second.last_hidden_states, **FP16_ULPS_4
    )
    assert unasked.last_hidden_states is None

    some = llm.generate(PROMPTS, [asked, plain, asked])
    every = llm.generate(PROMPTS, [asked, asked, asked])
    none = llm.generate(PROMPTS, [plain, plain, plain])

    for a, b, c in zip(some, every, none):
        assert_logits_close(a.outputs[0], b.outputs[0])
        assert_logits_close(a.outputs[0], c.outputs[0])
        assert c.outputs[0].last_hidden_states is None

    assert some[1].outputs[0].last_hidden_states is None
    for i in (0, 2):
        torch.testing.assert_close(
            some[i].outputs[0].last_hidden_states,
            every[i].outputs[0].last_hidden_states,
            **FP16_ULPS_4,
        )
