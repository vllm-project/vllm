# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blocking gate: FlashInfer ReplaySSM spec decode on the V2 GPU model runner.

This is the gate for the DSpark failure mode, not a generic parity test. PR
#49847 plumbed its per-request ``decode_base`` only through the classic worker.
DSpark is the one spec method with no classic proposer -- ``vllm/config/vllm.py``
forces the V2 runner for it -- so its ring admission reset never fired, the SSM
state replayed from the wrong origin, and GSM8K 5-shot fell from ~0.87 to 0.368
against a 0.872 baseline. A generic Nemotron-H engine test passes throughout,
because MTP and DFlash run on the classic worker.

The V2 half of the FlashInfer admission flag therefore needs coverage that
actually runs on the V2 runner. ``VLLM_USE_V2_MODEL_RUNNER=1`` forces it for any
config, so these cases exercise the same plumbing (``MambaHybridModelState``
slots, the ``idx_mapping`` gather, the scatter-clear) without needing a DSpark
checkpoint; the DSpark case at the bottom confirms it with the real drafter.

The conditions that broke DSpark are provoked deliberately:
  * chunked prefill so a request runs several prefill steps before its first
    decode -- the flag must survive all of them;
  * prompt lengths swept across a chunk boundary so at least one request ends on
    a single-token prompt chunk, the case the Triton path handles with a forced
    flush that FlashInfer cannot express;
  * more prompts than ``max_num_seqs`` so request slots churn and physical
    blocks are recycled with stale cursors;
  * a constrained block budget so requests are preempted and readmitted.

Accuracy oracle (run manually before merging, not in CI -- it is a 1319-sample
eval): GSM8K 5-shot must land within 0.01 of the no-ReplaySSM baseline, per
tests/evals/gsm8k. The historical numbers are 0.867 fixed vs 0.872 baseline vs
0.368 with the reset missing.
"""

import pytest

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark

MAMBA2_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MODELS = [pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40))]

# Chunked-prefill chunk size. Prompts are swept across it so one lands on a
# final single-token chunk regardless of how the tokenizer splits the text.
CHUNK = 64

_FILLER = "state space models process sequences recurrently and efficiently "


def _prompts_across_a_chunk_boundary(n: int = 8) -> list[str]:
    """Prompts whose token counts straddle CHUNK, CHUNK+1, ... CHUNK+n.

    Exact token counts depend on the tokenizer, so sweep a window instead of
    trying to hit CHUNK+1 exactly: the sweep guarantees at least one request
    ends its prefill on a one-token chunk.
    """
    return [_FILLER * (CHUNK // 8) + "word " * i + "In summary," for i in range(n)]


# Many more requests than max_num_seqs, so slots churn and blocks are recycled.
CHURN_PROMPTS = _prompts_across_a_chunk_boundary() + [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "Explain in one sentence why recurrent models are memory efficient:",
    "List three prime numbers:",
]


def _spec_config(num_spec_tokens: int = 3) -> dict:
    return {
        "method": "ngram",
        "num_speculative_tokens": num_spec_tokens,
        "prompt_lookup_max": 3,
    }


def _stress_kwargs(num_gpu_blocks_override: int | None) -> dict:
    kwargs = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK,
        # Forces slot churn: CHURN_PROMPTS is several times this.
        max_num_seqs=4,
        speculative_config=_spec_config(),
    )
    if num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = num_gpu_blocks_override
    return kwargs


def _run_v2_parity(
    vllm_runner,
    monkeypatch,
    model_name,
    *,
    enforce_eager: bool,
    num_gpu_blocks_override: int | None = None,
    algorithm: str = "auto",
):
    """Baseline spec vs FlashInfer ReplaySSM spec, both pinned to the V2 runner.

    Both sides run on V2 so runner numerics are common-mode and only ReplaySSM
    varies. Logprobs rather than greedy ids: ReplaySSM's arithmetic can flip a
    near-tie, but a wrong ring origin diverges far beyond that.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    common = _stress_kwargs(num_gpu_blocks_override)
    common["enforce_eager"] = enforce_eager

    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name,
        use_replayssm_spec=True,
        replayssm_buffer_len=16,
        mamba_backend="flashinfer",
        replayssm_spec_algorithm=algorithm,
        **common,
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="v2_baseline_spec",
        name_1="v2_replayssm_spec_flashinfer",
    )


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_v2_runner_chunked_prefill_churn_and_recycling(
    vllm_runner, monkeypatch, model_name, enforce_eager
):
    """The core gate: chunked prefill, slot churn, block recycling, both modes.

    Eager and CUDA graphs are both covered because the admission flag is
    consumed in the metadata builder, which the capture path enters with a
    synthesised batch and no reset mask at all.
    """
    _run_v2_parity(vllm_runner, monkeypatch, model_name, enforce_eager=enforce_eager)


@pytest.mark.parametrize("model_name", MODELS)
def test_v2_runner_survives_preemption_and_readmission(
    vllm_runner, monkeypatch, model_name
):
    """A readmitted request re-enters add_request, so its flag must be set again.

    Starved of blocks, requests are preempted and recomputed. If readmission did
    not re-arm the reset, the resumed request would append to a ring whose origin
    belongs to its previous life.
    """
    _run_v2_parity(
        vllm_runner,
        monkeypatch,
        model_name,
        enforce_eager=True,
        num_gpu_blocks_override=64,
    )


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("algorithm", ["monolith", "two-kernel"])
def test_v2_runner_forced_algorithms(vllm_runner, monkeypatch, model_name, algorithm):
    """Force each kernel: 'auto' alone may never reach the two-kernel path.

    Its crossover is batch * nheads against the device SM count, and this gate
    deliberately runs a small batch.
    """
    _run_v2_parity(
        vllm_runner,
        monkeypatch,
        model_name,
        enforce_eager=False,
        algorithm=algorithm,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_v2_runner_no_proposal_steps(vllm_runner, monkeypatch, model_name):
    """Steps where the drafter proposes nothing must still run the ReplaySSM path.

    The prompts here rarely repeat an n-gram, so most steps have no draft. On
    the classic runner `use_spec_decode` is step-level and goes false on those
    steps; the V2 gate on `num_speculative_tokens > 0` is config-level, and this
    pins that difference so a refactor cannot quietly align them the wrong way.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    common = _stress_kwargs(None)
    common["speculative_config"] = _spec_config(num_spec_tokens=1)
    common["enforce_eager"] = True

    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name,
        use_replayssm_spec=True,
        replayssm_buffer_len=16,
        mamba_backend="flashinfer",
        **common,
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="v2_baseline_no_proposal",
        name_1="v2_replayssm_no_proposal",
    )


# DSpark itself: the drafter that exposed the bug. Skipped unless a checkpoint
# is configured, because it needs one that ships DSpark weights in the target.
DSPARK_MODEL = None


@pytest.mark.skipif(
    DSPARK_MODEL is None,
    reason="set DSPARK_MODEL to a checkpoint shipping DSpark draft weights",
)
def test_dspark_replayssm_flashinfer_matches_baseline(vllm_runner):
    """DSpark forces the V2 runner (see VllmConfig.use_v2_model_runner).

    No VLLM_USE_V2_MODEL_RUNNER override here on purpose: this asserts the real
    dispatch path, not a forced one.
    """
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK,
        max_num_seqs=4,
        speculative_config={"method": "dspark", "num_speculative_tokens": 3},
    )
    with vllm_runner(DSPARK_MODEL, **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        DSPARK_MODEL,
        use_replayssm_spec=True,
        replayssm_buffer_len=16,
        mamba_backend="flashinfer",
        **common,
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            CHURN_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="dspark_baseline",
        name_1="dspark_replayssm_flashinfer",
    )
