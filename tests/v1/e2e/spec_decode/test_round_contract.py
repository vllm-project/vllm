# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end round contract checks against decoding without speculation, with
the test speculators of the ``vllm_add_test_speculators`` plugin."""

import asyncio
import json
import os
from contextlib import ExitStack
from typing import Any

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.reader import get_metrics_snapshot

from .utils import (
    _skip_if_insufficient_gpus_for_tp,
    compute_acceptance_len,
    require_test_speculators,
)

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA"),
    pytest.mark.skipif(
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") == "0",
        reason="the round contract is the V2 model runner's",
    ),
]


@pytest.fixture(autouse=True, scope="module")
def _test_speculators_plugin():
    require_test_speculators()


DENSE = "Qwen/Qwen3-0.6B"
HYBRID = "Qwen/Qwen3.5-0.8B"
# Ungated mirror of meta-llama/Llama-3.2-1B-Instruct, the draft's verifier.
EAGLE_TARGET = "unsloth/Llama-3.2-1B-Instruct"
EAGLE_DRAFT = "nm-testing/Llama3_2_1B_speculator.eagle3"
MOE = "ibm-research/PowerMoE-3b"
NUM_LOGPROBS = 5
SP = SamplingParams(max_tokens=48, temperature=0, logprobs=NUM_LOGPROBS)
# Repetitive prompts so that the n-gram drafts hit.
PROMPTS = [
    "The capital of France is Paris. The capital of Italy is Rome. "
    "The capital of Spain is Madrid. The capital of France is",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\n"
    "def mul(a, b):\n    return",
    "one two three four five one two three four five one two three four",
    "red green blue red green blue red green blue red green",
]
NGRAM = {"prompt_lookup_max": 5, "prompt_lookup_min": 3, "num_speculative_tokens": 3}
LOWERED = "spec decode: tree -> chain"


def _metric(llm: LLM, name: str) -> float:
    for metric in llm.get_metrics():
        if metric.name == name:
            return float(getattr(metric, "value", 0.0))
    return 0.0


def _tuples(outputs) -> list[tuple[list[int], str, Any]]:
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


def _run(
    model: str,
    speculative_config: dict | None = None,
    params: SamplingParams | list[SamplingParams] = SP,
    **kwargs,
):
    llm = LLM(
        model=model,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        disable_log_stats=False,
        speculative_config=speculative_config,
        **kwargs,
    )
    try:
        outputs = _tuples(llm.generate(PROMPTS, params))
        drafted = _metric(llm, "vllm:spec_decode_num_draft_tokens")
        acceptance = compute_acceptance_len(llm.get_metrics()) if drafted else 1.0
    finally:
        del llm
    return outputs, drafted, acceptance


_BASELINES: dict[tuple[str, str], list] = {}


def _baseline(model: str, **kwargs):
    # The engine kwargs hold dicts (attention and compilation configs).
    key = (model, json.dumps(kwargs, sort_keys=True))
    if key not in _BASELINES:
        _BASELINES[key] = _run(model, **kwargs)[0]
    return _BASELINES[key]


def _check(model, got, name, **kwargs):
    check_logprobs_close(
        outputs_0_lst=_baseline(model, **kwargs),
        outputs_1_lst=got,
        name_0="no_spec_decode",
        name_1=name,
    )


@pytest.mark.parametrize(
    "model, speculative_config",
    [
        (DENSE, {"method": "test_ngram_budget_one", **NGRAM}),
        (
            EAGLE_TARGET,
            {
                "method": "test_eagle3_budget_one",
                "model": EAGLE_DRAFT,
                "num_speculative_tokens": 3,
            },
        ),
    ],
    ids=["ngram", "eagle3"],
)
def test_budget_one_is_plain_decode(model, speculative_config):
    """A budget of 1 every step drafts nothing and decodes like no speculation."""
    got, drafted, _ = _run(model, speculative_config)
    assert drafted == 0
    _check(model, got, "budget_one")


def test_alternating_budget_on_hybrid():
    """Budgets alternating 1 and K + 1: the drafter's lookahead after budget 1,
    the dynamic speculative blocks and the GDN state across shrink and grow."""
    config = {"method": "test_mtp_alternating_budget", "num_speculative_tokens": 3}
    got, drafted, _ = _run(HYBRID, config)
    assert drafted > 0
    _check(HYBRID, got, "alternating_budget")


# No in-tree backend verifies trees, so one dense backend stands for all.
@pytest.mark.parametrize("model, tp", [(DENSE, 1), (HYBRID, 1), (HYBRID, 2)])
def test_tree_speculator_runs_chains(model, tp, capfd):
    """No in-tree attention backend verifies trees, and the hybrid's GDN layers
    have no tree kernel: a branching speculator starts, logs each reason once
    (the engine core logs the reduced limits of all workers), runs as a chain
    and decodes like no speculation."""
    _skip_if_insufficient_gpus_for_tp(tp)
    kwargs = {"tensor_parallel_size": tp}
    config = {"method": "test_ngram_tree", **NGRAM}
    got, drafted, acceptance = _run(model, config, **kwargs)
    out, err = capfd.readouterr()
    lowered = [line for line in (out + err).splitlines() if LOWERED in line]
    assert lowered and len(lowered) == len(set(lowered))
    assert drafted > 0 and acceptance > 1
    _check(model, got, "tree_runs_chains", **kwargs)


_TREE_MODES: dict[str, dict[str, Any]] = {
    "eager": {"enforce_eager": True},
    "piecewise": {"compilation_config": {"cudagraph_mode": "PIECEWISE"}},
}


@pytest.mark.parametrize("mode", _TREE_MODES)
def test_tree_speculator_verifies_trees(mode, capfd):
    """On the plugin's tree attention backend (TRITON_ATTN with the plan rows
    recomputed under their ancestor masks) a dense model verifies the 2-way
    trees: nothing lowers them, a step accepts a path that skips a sibling, and
    greedy tokens and logprobs match decoding without speculation."""
    from vllm_add_test_speculators import OFF_CHAIN

    kwargs = {"attention_config": {"backend": "CUSTOM"}, **_TREE_MODES[mode]}
    config = {"method": "test_ngram_tree", **NGRAM}
    got, drafted, acceptance = _run(DENSE, config, **kwargs)
    out, err = capfd.readouterr()
    assert LOWERED not in out + err
    assert drafted > 0 and acceptance > 1
    _check(DENSE, got, "tree_verifies_trees", **kwargs)
    assert OFF_CHAIN in out + err


def test_tree_speculator_samples_trees(capfd):
    """At temperature > 0 the 2-way trees verify by sampling: nothing lowers
    them, drafts are accepted and every request decodes."""
    kwargs = {"attention_config": {"backend": "CUSTOM"}, "enforce_eager": True}
    config = {"method": "test_ngram_tree", **NGRAM}
    params = SamplingParams(max_tokens=SP.max_tokens, temperature=0.8, seed=0)
    got, drafted, acceptance = _run(DENSE, config, params, **kwargs)
    out, err = capfd.readouterr()
    assert LOWERED not in out + err
    assert drafted > 0 and acceptance > 1
    assert all(tokens for tokens, _, _ in got)


# FULL CUDA graphs of one shape: 4 requests of K + 1 rows. Requests end at
# different lengths, so the later verify batches are padded to it.
_ONE_FULL_SHAPE = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [4 * (NGRAM["num_speculative_tokens"] + 1)],
}
_STAGGERED = [
    SamplingParams(max_tokens=n, temperature=0, logprobs=NUM_LOGPROBS)
    for n in (48, 40, 32, 24)
]
# float32, but TRITON_ATTN's dots round to TF32 and with decode CUDA graphs
# it picks its split-softmax kernel by capture size, so the graph and no-graph
# runs differ in logprob by up to 8e-3 here. A stale tree changes tokens.
LOGPROB_ATOL = 3e-2


def test_tree_replays_one_full_cudagraph(capfd):
    """One captured FULL CUDA graph replays different trees: the speculator
    cycles two trees and a chain with the same rows through the graph of the
    4-request verify batch, then padded batches as requests end. Tokens and
    logprobs match the run without CUDA graphs, so nothing of the tree
    (parents, ancestor mask, KV commit) is frozen at capture. Steps with a
    prefill run without the graph."""
    from vllm_add_test_speculators import EAGER_UNIFORM_VERIFY, PLANNED, UNIFORM_VERIFY

    config = {"method": "test_ngram_tree_cycle", **NGRAM}
    kwargs: dict[str, Any] = {
        "attention_config": {"backend": "CUSTOM"},
        "dtype": "float32",
        "max_num_seqs": len(PROMPTS),
    }
    no_graphs = {"cudagraph_mode": "NONE"}
    ref, _, _ = _run(DENSE, config, _STAGGERED, compilation_config=no_graphs, **kwargs)
    capfd.readouterr()
    kwargs["compilation_config"] = _ONE_FULL_SHAPE
    got, drafted, acceptance = _run(DENSE, config, _STAGGERED, **kwargs)
    out, err = capfd.readouterr()
    assert LOWERED not in out + err
    assert drafted > 0 and acceptance > 1
    # Each topology replayed the graph: a step built but not run in Python.
    for topology in ("sibling_of_first", "sibling_of_second", "chain"):
        assert f"{PLANNED} {topology}" in out + err
        built = (out + err).count(f"{UNIFORM_VERIFY} {topology}\n")
        ran = (out + err).count(f"{EAGER_UNIFORM_VERIFY} {topology}\n")
        print(f"{topology}: {built - ran} of {built} uniform steps replayed")
        assert built > ran, topology
    assert [tokens for tokens, _, _ in got] == [tokens for tokens, _, _ in ref]
    diffs = [
        abs(step[token_id].logprob - want[token_id].logprob)
        for (_, _, ref_logprobs), (_, _, logprobs) in zip(ref, got)
        for want, step in zip(ref_logprobs, logprobs)
        for token_id in want.keys() & step.keys()
    ]
    print(f"max |logprob diff| to no graphs over {len(diffs)}: {max(diffs):.2e}")
    assert max(diffs) <= LOGPROB_ATOL


async def _generate(engine: AsyncLLM, index: int, prompt: str):
    params = SamplingParams(
        max_tokens=SP.max_tokens,
        temperature=0,
        logprobs=NUM_LOGPROBS,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    final = None
    async for out in engine.generate(
        request_id=f"req-{index}", prompt=prompt, sampling_params=params
    ):
        final = out
    return final


async def _run_dp(speculative_config: dict | None):
    engine_args = AsyncEngineArgs(
        model=MOE,
        data_parallel_size=2,
        data_parallel_backend="mp",
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)
        prompts = PROMPTS * 4
        outputs = await asyncio.wait_for(
            asyncio.gather(*(_generate(engine, i, p) for i, p in enumerate(prompts))),
            timeout=600,
        )
    return _tuples(outputs)


def test_data_parallel_rank_at_budget_one():
    """DP=2 on an MoE (ranks step in lockstep) where rank 0 never drafts and
    rank 1 drafts: no hang, and both decode like no speculation."""
    _skip_if_insufficient_gpus_for_tp(2)
    baseline = asyncio.run(_run_dp(None))
    config = {"method": "test_ngram_budget_one_on_rank_zero", **NGRAM}
    got = asyncio.run(_run_dp(config))
    drafted = sum(
        float(getattr(m, "value", 0.0))
        for m in get_metrics_snapshot()
        if m.name == "vllm:spec_decode_num_draft_tokens"
    )
    assert drafted > 0, "rank 1 drafted nothing; the test would be vacuous"
    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=got,
        name_0="no_spec_decode",
        name_1="rank_zero_at_budget_one",
    )
