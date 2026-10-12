# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-sharded sampling must be a drop-in for replicated sampling.

The two boots are compared token for token, which is only meaningful because
both schedule the same batches. With the engine in the caller's process every
request is queued before the first step, so the first prefill has the same
shape in both runs. Give the engine its own process and the comparison turns
into a coin toss: prefill starts before the last requests arrive over IPC, so
one boot batches two requests where the other batches one. The batch shape
sets the GEMM reduction order, logprobs shift by up to ~0.25, and near-ties
flip — measured at up to 5 of 16 completions, with sharding disabled on both
sides. A real sharding bug — rows routed to the wrong request, misaligned
gathers — produces many-nats divergence and scrambled top-k sets."""

import pytest

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.logprobs import Logprob, LogprobsOnePosition

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Misrouted rows move logprobs by nats; anything under this is float noise.
# With matched batches the two boots agreed bitwise over 320 positions, so the
# bound is not expected to be reached.
LOGPROB_TOL = 0.5

PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
    "Write a haiku about the ocean.",
    "1, 1, 2, 3, 5, 8,",
    "The chemical symbol for gold is",
    "Once upon a time, in a distant kingdom,",
    "Python's GIL stands for",
    "The fastest land animal is",
]

# Exercise greedy, seeded random, top-k/top-p, penalties, and logprobs.
SAMPLING_PARAMS = [
    SamplingParams(temperature=0.0, max_tokens=32, logprobs=5, prompt_logprobs=1),
    SamplingParams(temperature=0.8, seed=42, top_p=0.9, max_tokens=32, logprobs=3),
    SamplingParams(temperature=0.7, seed=7, top_k=50, min_p=0.05, max_tokens=32),
    SamplingParams(
        temperature=0.9,
        seed=123,
        frequency_penalty=0.5,
        repetition_penalty=1.2,
        max_tokens=32,
        logprobs=2,
    ),
]


def _is_sharded_sampling_active(worker) -> bool:
    return worker.model_runner.batch_sharder is not None


def _sampling_params() -> list[SamplingParams]:
    return [SAMPLING_PARAMS[i % len(SAMPLING_PARAMS)] for i in range(len(PROMPTS))]


def _generate(monkeypatch: pytest.MonkeyPatch, disable_sharding: bool):
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        enable_batch_sharded_sampling=not disable_sharding,
        max_model_len=1024,
        # More requests than ranks, so round-robin slot ownership splits the
        # batch across both ranks.
        max_num_seqs=len(PROMPTS),
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        # Timing-based kernel selection is the dominant source of cross-boot
        # numeric noise; disable it so the comparison bounds stay tight.
        kernel_config={"enable_flashinfer_autotune": False},
    )
    try:
        # Guard against a vacuous comparison: verify every worker is in the
        # intended sampling mode.
        modes = llm.llm_engine.collective_rpc(_is_sharded_sampling_active)
        assert all(mode == (not disable_sharding) for mode in modes), modes
        params = _sampling_params()
        # Two waves: the second reuses the request slots freed by the first
        # (in whatever order requests finished), covering the cross-rank
        # slot-recycling path that request ownership derives from.
        wave1 = llm.generate(PROMPTS, params)
        wave2 = llm.generate(PROMPTS[::-1], params[::-1])
        return wave1 + wave2
    finally:
        del llm
        cleanup_dist_env_and_memory()


def _requested_top_k(lps: LogprobsOnePosition, num_logprobs: int) -> set[int]:
    """Exclude the extra sampled/prompt token, including at rank k+1."""
    top = {
        token_id
        for token_id, logprob in lps.items()
        if logprob.rank is not None and 1 <= logprob.rank <= num_logprobs
    }
    assert len(top) == num_logprobs, (
        f"expected {num_logprobs} ranked top-k entries, got {len(top)}"
    )
    return top


def _assert_logprob_dicts_close(
    ref_lps: LogprobsOnePosition,
    out_lps: LogprobsOnePosition,
    what: str,
    num_logprobs: int,
) -> None:
    # Allow one boundary entry of the top-k set to swap at a near-tie. Compare
    # the top-k alone: counting the sampled token would spend that allowance
    # on the extra entry itself whenever only one side carries one.
    ref_top = _requested_top_k(ref_lps, num_logprobs)
    out_top = _requested_top_k(out_lps, num_logprobs)
    shared_top = ref_top & out_top
    assert len(shared_top) >= num_logprobs - 1, (
        f"{what}: top-k token sets diverge: {sorted(ref_top)} vs {sorted(out_top)}"
    )
    # Values are comparable for every shared token, sampled one included.
    for token_id in set(ref_lps) & set(out_lps):
        diff = abs(ref_lps[token_id].logprob - out_lps[token_id].logprob)
        assert diff <= LOGPROB_TOL, (
            f"{what}[{token_id}]: logprob diff {diff} "
            f"({ref_lps[token_id].logprob} vs {out_lps[token_id].logprob})"
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("num_logprobs", "ref_ranks", "out_ranks"),
    [
        (2, {0: 1, 1: 2, 2: 3}, {0: 1, 3: 2, 2: 4}),
        (1, {0: 1, 1: 2}, {2: 1, 1: 3}),
        (2, {0: 1, 1: 2}, {0: 1, 2: 2, 1: 3}),
        (2, {0: 1, 1: 2, 2: 4}, {0: 1, 1: 2, 2: 5}),
        (0, {0: 2}, {0: 3}),
    ],
    ids=["decode-boundary", "prompt-boundary", "sampled-in-top-k", "gap", "zero"],
)
def test_logprob_comparison_allows_sampled_token_rank_changes(
    num_logprobs: int, ref_ranks: dict[int, int], out_ranks: dict[int, int]
):
    ref_lps, out_lps = (
        {token: Logprob(-2.0 - rank / 100, rank=rank) for token, rank in ranks.items()}
        for ranks in (ref_ranks, out_ranks)
    )
    _assert_logprob_dicts_close(ref_lps, out_lps, "boundary swap", num_logprobs)


@pytest.mark.cpu_test
def test_logprob_comparison_rejects_multiple_top_k_swaps():
    ref = {0: Logprob(-2.0, rank=1), 1: Logprob(-2.01, rank=2)}
    out = {2: Logprob(-2.0, rank=1), 3: Logprob(-2.01, rank=2)}
    with pytest.raises(AssertionError, match="top-k token sets diverge"):
        _assert_logprob_dicts_close(ref, out, "misrouted top-k", num_logprobs=2)


@pytest.mark.cpu_test
def test_logprob_comparison_checks_extra_sampled_token_value():
    ref = {0: Logprob(-1.0, rank=1), 1: Logprob(-3.0, rank=3)}
    out = {0: Logprob(-1.0, rank=1), 1: Logprob(-4.0, rank=3)}
    with pytest.raises(AssertionError, match="logprob diff"):
        _assert_logprob_dicts_close(ref, out, "sampled token", num_logprobs=1)


@pytest.mark.cpu_test
@pytest.mark.parametrize("rank", [None, 2])
def test_logprob_comparison_rejects_missing_top_k_rank(rank: int | None):
    lps = {0: Logprob(-1.0, rank=rank)}
    with pytest.raises(AssertionError, match="expected 1 ranked top-k entries"):
        _assert_logprob_dicts_close(lps, lps, "missing rank", num_logprobs=1)


@multi_gpu_test(num_gpus=2)
def test_sharded_sampling_outputs_match(monkeypatch: pytest.MonkeyPatch):
    """Generation, logprobs, and prompt logprobs match between batch-sharded
    sampling and the replicated fallback."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Required for the collective_rpc mode check below.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Queue every request before the first step, so both boots schedule the
    # same batches -- see the module docstring.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    ref_outputs = _generate(monkeypatch, disable_sharding=True)
    shard_outputs = _generate(monkeypatch, disable_sharding=False)

    assert len(ref_outputs) == len(shard_outputs) == 2 * len(PROMPTS)
    params = _sampling_params()
    divergent = []
    for i, (ref, out, sampling_params) in enumerate(
        zip(ref_outputs, shard_outputs, params + params[::-1])
    ):
        ref_completion = ref.outputs[0]
        out_completion = out.outputs[0]
        ref_ids = list(ref_completion.token_ids)
        out_ids = list(out_completion.token_ids)

        # Logits are only comparable at positions whose context (prompt +
        # previously sampled tokens) is identical: every position up to and
        # including the first divergence.
        prefix = 0
        while prefix < min(len(ref_ids), len(out_ids)):
            if ref_ids[prefix] != out_ids[prefix]:
                break
            prefix += 1
        if ref_ids != out_ids:
            divergent.append((i, prefix))

        if ref_completion.logprobs is not None:
            assert out_completion.logprobs is not None
            assert sampling_params.logprobs is not None
            comparable = min(prefix + 1, len(ref_ids), len(out_ids))
            for pos in range(comparable):
                _assert_logprob_dicts_close(
                    ref_completion.logprobs[pos],
                    out_completion.logprobs[pos],
                    f"prompt {i} logprobs[{pos}]",
                    sampling_params.logprobs,
                )

        # The prompt is fixed, so every prompt-logprob position is comparable.
        if ref.prompt_logprobs is not None:
            assert out.prompt_logprobs is not None
            assert sampling_params.prompt_logprobs is not None
            for pos, (ref_lps, out_lps) in enumerate(
                zip(ref.prompt_logprobs, out.prompt_logprobs)
            ):
                if ref_lps is None or out_lps is None:
                    assert ref_lps is None and out_lps is None
                    continue
                _assert_logprob_dicts_close(
                    ref_lps,
                    out_lps,
                    f"prompt {i} prompt_logprobs[{pos}]",
                    sampling_params.prompt_logprobs,
                )

    assert not divergent, (
        f"{len(divergent)}/{2 * len(PROMPTS)} completions diverged at "
        f"(prompt, position) {divergent}; the sampling modes did not agree "
        "token for token with in-process scheduling"
    )
