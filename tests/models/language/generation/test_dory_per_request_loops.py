# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request Dory loop counts (SamplingParams.extra_args["loops"]).

A request with ``loops=N`` must produce exactly what a server started with
``n_recurrent_loops=N`` produces, alone and inside a batch that mixes loop
counts. Exact equality needs batch-invariant kernels (VLLM_BATCH_INVARIANT=1).
"""

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

from ...registry import HF_EXAMPLE_MODELS

MODEL = HF_EXAMPLE_MODELS.get_hf_info("DoryForCausalLM").default
PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Which team should handle a double charge on a card: billing or shipping?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 * 23? Show your work.",
]
MAX_LOOPS = 3
COMMON = dict(max_model_len=1024, max_num_seqs=16, dtype="bfloat16")

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="per-request loops need CUDA graphs"
)


def _params(loops: int | None) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=48,
        logprobs=0,
        extra_args=None if loops is None else {"loops": loops},
    )


def _run(llm, jobs: list[tuple[int, int | None]], together: bool):
    """Jobs = [(prompt index, loops)] -> {(prompt index, loops): (ids, logprobs)}"""
    out = {}
    batches = [jobs] if together else [[j] for j in jobs]
    for batch in batches:
        llm.llm.reset_prefix_cache()
        res = llm.llm.generate(
            [PROMPTS[i] for i, _ in batch], [_params(n) for _, n in batch]
        )
        for (i, n), r in zip(batch, res):
            c = r.outputs[0]
            lps = [lp[t].logprob for lp, t in zip(c.logprobs, c.token_ids)]
            out[(i, n)] = (list(c.token_ids), lps)
    return out


@pytest.fixture
def batch_invariant(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


@pytest.mark.usefixtures("batch_invariant")
def test_per_request_loops_match_fixed_servers(vllm_runner):
    prompts = range(len(PROMPTS))
    fixed = {}
    for n in range(1, MAX_LOOPS + 1):
        overrides = {"n_recurrent_loops": n}
        with vllm_runner(MODEL, hf_overrides=overrides, **COMMON) as llm:
            for (i, _), v in _run(llm, [(i, None) for i in prompts], False).items():
                fixed[(i, n)] = v

    jobs = [(i, n) for i in prompts for n in range(1, MAX_LOOPS + 1)]
    overrides = {"n_recurrent_loops": 2, "max_recurrent_loops": MAX_LOOPS}
    with vllm_runner(MODEL, hf_overrides=overrides, **COMMON) as llm:
        alone = _run(llm, jobs, together=False)
        # interleaved loop counts in one batch: the graph of the largest count
        mixed = _run(llm, jobs, together=True)
        default = _run(llm, [(i, None) for i in prompts], together=False)
    for key in jobs:
        assert alone[key] == fixed[key], f"alone {key}"
        assert mixed[key] == fixed[key], f"mixed batch {key}"
    for i in prompts:
        assert default[(i, None)] == fixed[(i, 2)]


@pytest.mark.usefixtures("batch_invariant")
def test_prefix_cache_is_not_shared_across_loop_counts(vllm_runner):
    overrides = {"n_recurrent_loops": 2, "max_recurrent_loops": MAX_LOOPS}
    with vllm_runner(MODEL, hf_overrides=overrides, **COMMON) as llm:
        cold = _run(llm, [(0, 3)], together=False)[(0, 3)]
        llm.llm.reset_prefix_cache()
        # prompt 0 at the default count fills the prefix cache ...
        llm.llm.generate([PROMPTS[0]], [_params(2)])
        # ... which a 3-loop request must not reuse
        res = llm.llm.generate([PROMPTS[0]], [_params(3)])[0].outputs[0]
        lps = [lp[t].logprob for lp, t in zip(res.logprobs, res.token_ids)]
    assert (list(res.token_ids), lps) == cold
