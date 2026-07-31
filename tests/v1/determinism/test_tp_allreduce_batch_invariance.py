# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the tensor-parallel all-reduce path (#50136).

Every other test in this directory runs at TP=1, so the collective path is
untested. This compares a request against itself across batch *compositions*,
not across repeated identical runs: re-running the same batch twice passes
even when a backend is batch-dependent.

It also asserts the custom kernel actually carried the collectives. Without
that, a green result is ambiguous -- if custom all-reduce disabled itself,
NCCL would carry everything and the run would be invariant for the
pre-existing reason. ``test_engagement_assertion_can_fail`` is the
counterpart proving that assertion can fail.
"""

import random

import pytest
import torch
from tests.utils import multi_gpu_marks
from utils import TEST_MODEL, _extract_step_logprobs, skip_unsupported

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MIN_GPUS = 4
BATCH_SIZES = [17, 65]
PROMPT_LENS = [512, 1024]


def requires_tp4(fn):
    for mark in reversed(multi_gpu_marks(num_gpus=MIN_GPUS)):
        fn = mark(fn)
    return fn


def _ar_state(self):
    """Run on each worker: which all-reduce backends are live, and engagement."""
    from vllm.distributed.parallel_state import get_tp_group

    dc = get_tp_group().device_communicator
    ca = getattr(dc, "ca_comm", None)
    return {
        "flashinfer": getattr(dc, "fi_ar_comm", None) is not None,
        "aiter": getattr(dc, "aiter_ar_comm", None) is not None,
        "custom": ca is not None and not ca.disabled,
        "calls": getattr(ca, "_custom_ar_calls", 0),
        "largest": getattr(ca, "_custom_ar_max_bytes", 0),
        "max_size": getattr(ca, "max_size", 0),
    }


def _prompts(seed=1234):
    rng = random.Random(seed)
    needles = [[rng.randrange(1000, 29000) for _ in range(n)] for n in PROMPT_LENS]
    filler = [
        [rng.randrange(1000, 29000) for _ in range(rng.choice(PROMPT_LENS))]
        for _ in range(max(BATCH_SIZES))
    ]
    return needles, filler


def _run(llm, prompts, params):
    return llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )


def _llm(disable_custom_all_reduce=False):
    return LLM(
        model=TEST_MODEL,
        tensor_parallel_size=MIN_GPUS,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        seed=0,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )


@skip_unsupported
@requires_tp4
@pytest.mark.timeout(1800)
def test_tp_allreduce_is_batch_invariant(monkeypatch):
    """A request is bitwise identical however it is batched, at TP=4."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # collective_rpc(callable) requires pickle-based serialization.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    params = SamplingParams(
        temperature=0.0, max_tokens=32, ignore_eos=True, logprobs=1, seed=1234
    )
    needles, filler = _prompts()
    llm = _llm()
    try:
        refs = []
        for needle in needles:
            lp, ids = _extract_step_logprobs(_run(llm, [needle], params)[0])
            assert ids is not None, "logprobs were not returned"
            refs.append((lp, ids))

        for batch_size in BATCH_SIZES:
            for idx, needle in enumerate(needles):
                pos = random.Random(batch_size + idx).randrange(batch_size)
                batch = list(filler[: batch_size - 1])
                batch.insert(pos, needle)
                got_lp, got_ids = _extract_step_logprobs(
                    _run(llm, batch, params)[pos]
                )
                label = f"batch_size={batch_size} position={pos} needle={idx}"
                assert list(refs[idx][1]) == list(got_ids), f"{label}: token ids"
                # Bitwise, not allclose: sub-ULP drift still breaks the
                # trainer/sampler logprob ratios invariance exists to protect.
                assert torch.equal(refs[idx][0], got_lp), f"{label}: logprobs"

        # The result above is only meaningful if the custom kernel carried the
        # collectives, and only sound if no unaudited backend also ran.
        for rank, s in enumerate(llm.collective_rpc(_ar_state)):
            assert s["custom"] and s["calls"] > 0, (
                f"rank {rank}: custom all-reduce never dispatched, so the "
                "invariance result above is vacuous -- NCCL carried it."
            )
            assert not s["flashinfer"] and not s["aiter"], (
                f"rank {rank}: an unaudited all-reduce backend was live."
            )
            # The point of sizing at init: every all-reduce fits, so the
            # custom-vs-NCCL choice cannot change mid-run.
            assert s["largest"] <= s["max_size"], (
                f"rank {rank}: {s['largest']} B exceeded the "
                f"{s['max_size']} B buffer; the size gate can flip again."
            )
    finally:
        del llm


@skip_unsupported
@requires_tp4
@pytest.mark.timeout(1800)
def test_engagement_assertion_can_fail(monkeypatch):
    """With the custom path off, output stays invariant but engagement is zero.

    A test that only checked reproducibility would pass here. The engagement
    assertion above is what distinguishes "the custom path is batch-invariant"
    from "no custom path ran".
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    needles, _ = _prompts()
    llm = _llm(disable_custom_all_reduce=True)
    try:
        _run(llm, [needles[0]], SamplingParams(temperature=0.0, max_tokens=8))
        for rank, s in enumerate(llm.collective_rpc(_ar_state)):
            assert not s["custom"] and s["calls"] == 0, (
                f"rank {rank}: custom all-reduce engaged despite "
                "disable_custom_all_reduce=True."
            )
    finally:
        del llm
