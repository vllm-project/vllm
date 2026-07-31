# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the tensor-parallel all-reduce path.

Every other test in this directory runs at TP=1, so the collective path is
untested. That gap is what allowed the bug in #50136: ``should_custom_ar()``
selects between the custom kernel and NCCL by tensor size, the all-reduce
tensor is ``[num_tokens, hidden_size]``, and so kernel selection -- and with it
the floating-point reduction order -- tracked batch composition.

The test compares a request against itself across batch *compositions*, not
across repeated identical runs. Re-running the same batch twice passes even
when a backend is batch-dependent; only varying what a request is batched with
exposes it.

It also asserts that the custom kernel actually handled collectives during the
run being judged. Without that, a green result is ambiguous: if custom
all-reduce disables itself, NCCL carries every collective and the run is
invariant for the pre-existing reason, proving nothing about the code under
test. ``test_engagement_assertion_can_fail`` is the counterpart that proves
this assertion is load-bearing rather than decorative -- it runs the same
engine with the custom path disabled, where output is still invariant but
engagement must be zero.

Engagement is read from the workers with ``collective_rpc`` rather than by
scraping the log for the ``BATCH_INVARIANT_CUSTOM_AR_ENGAGED`` marker. Log
scraping was tried and is unreliable: under pytest's ``capfd`` fixture fd 1 is
a regular file rather than a tty, so worker processes block-buffer their
output and only flush it when they exit -- after the assertion has already
read an empty buffer. Reading the counters directly also gives exact per-rank
call counts instead of the marker's milestone floors.
"""

import random

import pytest
import torch
from utils import TEST_MODEL, _extract_step_logprobs, skip_unsupported

from tests.utils import multi_gpu_marks
from vllm import LLM, SamplingParams

MIN_GPUS = 4
BATCH_SIZES = [17, 65]  # straddle kernel tile boundaries
PROMPT_LENS = [512, 1024]

TP4_MARKS = multi_gpu_marks(num_gpus=MIN_GPUS)


def requires_tp4(fn):
    """Apply the repo-standard multi-GPU marks.

    ``multi_gpu_test`` is deliberately not used: it additionally wraps the
    test in ``create_new_process_for_each_test()``, and these tests hold an
    engine across several generate calls on purpose.
    """
    for mark in reversed(TP4_MARKS):
        fn = mark(fn)
    return fn


def _custom_ar_stats(self):
    """Read the custom all-reduce engagement counters on one worker.

    Runs inside the worker process via ``collective_rpc``; ``self`` is the
    worker. Returns plain data so it survives the RPC boundary.
    """
    from vllm.distributed.parallel_state import get_tp_group

    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    if ca is None or ca.disabled:
        return {"engaged": False, "calls": 0, "largest": 0, "max_size": 0}
    return {
        "engaged": True,
        "calls": getattr(ca, "_custom_ar_calls", 0),
        "largest": getattr(ca, "_custom_ar_max_bytes", 0),
        "max_size": ca.max_size,
    }


def _prompts(seed: int = 1234) -> tuple[list[list[int]], list[list[int]]]:
    """Deterministic synthetic token-id prompts: (needles, filler)."""
    rng = random.Random(seed)
    needles = [[rng.randrange(1000, 29000) for _ in range(n)] for n in PROMPT_LENS]
    filler = [
        [rng.randrange(1000, 29000) for _ in range(rng.choice(PROMPT_LENS))]
        for _ in range(max(BATCH_SIZES))
    ]
    return needles, filler


def _run(llm, prompts, params):
    from vllm.inputs import TokensPrompt

    return llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        params,
        use_tqdm=False,
    )


def _assert_bitwise_equal(ref, got, label: str) -> None:
    ref_lp, ref_ids = ref
    got_lp, got_ids = got
    assert list(ref_ids) == list(got_ids), (
        f"{label}: token ids diverged from the bs=1 reference"
    )
    # Bitwise, not allclose: sub-ULP drift still breaks the trainer/sampler
    # logprob ratios that batch invariance exists to protect.
    assert torch.equal(ref_lp, got_lp), (
        f"{label}: logprobs diverged bitwise from the bs=1 reference "
        "(tokens matched, so this is reduction-order drift)"
    )


def _build_llm(disable_custom_all_reduce: bool = False) -> LLM:
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

    llm = _build_llm()
    try:
        # bs=1 reference for each needle.
        refs = []
        for needle in needles:
            out = _run(llm, [needle], params)
            lp, ids = _extract_step_logprobs(out[0])
            assert ids is not None, "logprobs were not returned"
            refs.append((lp, ids))

        # Same needles inside batches of varying composition.
        for batch_size in BATCH_SIZES:
            for idx, needle in enumerate(needles):
                pos = random.Random(batch_size + idx).randrange(batch_size)
                batch = list(filler[: batch_size - 1])
                batch.insert(pos, needle)
                outs = _run(llm, batch, params)
                _assert_bitwise_equal(
                    refs[idx],
                    _extract_step_logprobs(outs[pos]),
                    f"batch_size={batch_size} position={pos} needle={idx}",
                )

        # The invariance result above is only meaningful if the custom kernel
        # actually carried the collectives; if it had disabled itself, NCCL
        # would have carried them and the run would be invariant for the
        # pre-existing reason.
        stats = llm.collective_rpc(_custom_ar_stats)
        assert len(stats) == MIN_GPUS
        for rank, s in enumerate(stats):
            assert s["engaged"], (
                f"rank {rank}: custom all-reduce was disabled, so the "
                "invariance result above is vacuous -- NCCL carried the "
                "collectives and this test proved nothing about the custom "
                "path."
            )
            assert s["calls"] > 0, (
                f"rank {rank}: custom all-reduce never dispatched during this run."
            )
            # The whole point of sizing buffers at init: every all-reduce
            # fits, so the custom-vs-NCCL choice cannot change mid-run.
            assert s["largest"] <= s["max_size"], (
                f"rank {rank}: largest all-reduce {s['largest']} B exceeded "
                f"the {s['max_size']} B buffer, which is the condition that "
                "makes kernel selection batch-dependent."
            )
    finally:
        del llm


@skip_unsupported
@requires_tp4
@pytest.mark.timeout(1800)
def test_engagement_assertion_can_fail(monkeypatch):
    """The engagement assertion must fail when the custom path is disabled.

    Output stays invariant here -- NCCL is used uniformly -- which is exactly
    the point: a test that only checked reproducibility would pass on this
    configuration. The engagement assertion is what distinguishes "the custom
    path is batch-invariant" from "no custom path ran".
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    params = SamplingParams(
        temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=1, seed=1234
    )
    needles, _ = _prompts()

    llm = _build_llm(disable_custom_all_reduce=True)
    try:
        _run(llm, [needles[0]], params)
        stats = llm.collective_rpc(_custom_ar_stats)
        assert len(stats) == MIN_GPUS
        for rank, s in enumerate(stats):
            assert not s["engaged"] and s["calls"] == 0, (
                f"rank {rank}: custom all-reduce engaged despite "
                "disable_custom_all_reduce=True; the engagement assertion in "
                "test_tp_allreduce_is_batch_invariant would pass "
                "unconditionally and prove nothing."
            )
    finally:
        del llm
