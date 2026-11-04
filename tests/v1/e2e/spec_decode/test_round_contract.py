# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks of the speculative round contract on the V2 model runner.

Each test compares greedy tokens and logprobs with a run without speculative
decoding and checks that it drafted (or, at budget 1, that it did not), so it
cannot pass vacuously. The speculators are the test-only ones of the
``vllm_add_test_speculators`` general plugin in ``tests/plugins``, which
registers them in every vLLM process.
"""

import asyncio
import importlib.util
import os
from contextlib import ExitStack

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


def _tuples(outputs) -> list[tuple[list[int], str, object]]:
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


def _run(model: str, speculative_config: dict | None = None, **kwargs):
    llm = LLM(
        model=model,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        disable_log_stats=False,
        speculative_config=speculative_config,
        **kwargs,
    )
    try:
        outputs = _tuples(llm.generate(PROMPTS, SP))
        drafted = _metric(llm, "vllm:spec_decode_num_draft_tokens")
        acceptance = compute_acceptance_len(llm.get_metrics()) if drafted else 1.0
    finally:
        del llm
    return outputs, drafted, acceptance


_BASELINES: dict[tuple, list] = {}


def _baseline(model: str, **kwargs):
    key = (model, *sorted(kwargs.items()))
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


# --- E2: budget 1 is plain decode ---------------------------------------------


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


# --- E3: alternating budgets on a hybrid --------------------------------------


def test_alternating_budget_on_hybrid():
    """Budgets alternating 1 and K + 1: the drafter's lookahead after budget 1,
    the dynamic speculative blocks and the GDN state across shrink and grow."""
    config = {"method": "test_mtp_alternating_budget", "num_speculative_tokens": 3}
    got, drafted, _ = _run(HYBRID, config)
    assert drafted > 0
    _check(HYBRID, got, "alternating_budget")


# --- E4: a tree speculator on full attention -----------------------------------

_CUDAGRAPH_MODES = ["FULL_AND_PIECEWISE", "NONE"]


def _backends():
    backends = ["FLASH_ATTN", "TRITON_ATTN"]
    if importlib.util.find_spec("flashinfer") is not None:
        backends.append("FLASHINFER")
    return backends


def _tree_run(method, backend="FLASH_ATTN", cudagraph_mode="FULL_AND_PIECEWISE"):
    kwargs = {
        "attention_config": {"backend": backend},
        "compilation_config": {"cudagraph_mode": cudagraph_mode},
    }
    got, drafted, acceptance = _run(DENSE, {"method": method, **NGRAM}, **kwargs)
    assert drafted > 0
    assert acceptance > 1
    _check(DENSE, got, method, **kwargs)


@pytest.mark.parametrize("cudagraph_mode", _CUDAGRAPH_MODES)
@pytest.mark.parametrize("backend", _backends())
def test_tree_plan_matches_plain_decode(backend, cudagraph_mode):
    """A speculator with ``max_branching > 1`` on a model that verifies trees:
    its plan is laid out, accepted by the tree walk and committed along the
    accepted path. The plan is the n-gram chain, so a chain's mask is right."""
    _tree_run("test_ngram_chain_as_tree", backend, cudagraph_mode)


@pytest.mark.xfail(
    reason="no in-tree attention backend reads tree_ancestors, but the startup "
    "rule lets a full-attention model plan trees: siblings attend to each other"
)
def test_branching_tree_matches_plain_decode():
    _tree_run("test_ngram_tree")


# --- E5: a tree speculator on a hybrid without a provider ---------------------


@pytest.mark.parametrize("tp", [1, 2])
def test_tree_speculator_on_hybrid_runs_chains(tp, capfd):
    """Without a recurrent tree kernel the hybrid lowers the tree to chains: it
    starts, logs why once per worker, drafts and decodes like no speculation."""
    _skip_if_insufficient_gpus_for_tp(tp)
    config = {"method": "test_ngram_tree", **NGRAM}
    got, drafted, _ = _run(HYBRID, config, tensor_parallel_size=tp)
    out, err = capfd.readouterr()
    assert (out + err).count(LOWERED) == tp
    assert drafted > 0
    _check(HYBRID, got, "tree_on_hybrid", tensor_parallel_size=tp)


# --- E7: data parallel ranks at different budgets -----------------------------


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
