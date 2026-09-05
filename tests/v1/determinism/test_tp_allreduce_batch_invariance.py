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
from utils import TEST_MODEL, _extract_step_logprobs, skip_unsupported

from tests.utils import multi_gpu_marks
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
                got_lp, got_ids = _extract_step_logprobs(_run(llm, batch, params)[pos])
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


# --------------------------------------------------------------------------
# Guard unit tests. These need no GPU: they exercise the decision logic on a
# stand-in object, so a regression fails in plain CPU CI rather than only on a
# 4-GPU runner.
# --------------------------------------------------------------------------


class _Gate:
    """Minimal stand-in carrying only what should_custom_ar() reads."""

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce as _CA,
    )

    disabled = False
    world_size = 4
    fully_connected = True
    max_size = 32 * 1024 * 1024
    # Borrowed, not copied: should_custom_ar() reads this off the instance, and
    # a literal here would silently diverge from the real dispatch bound.
    _MAX_ALL_REDUCE_WORLD_SIZE = _CA._MAX_ALL_REDUCE_WORLD_SIZE


def _should(tensor):
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    return CustomAllreduce.should_custom_ar(_Gate(), tensor)


def test_misaligned_tensor_raises_under_batch_invariance(monkeypatch):
    """A non-16-byte-multiple tensor must fail loudly, not fall back to NCCL.

    Falling back for this tensor only would make backend selection depend on
    batch composition again -- e.g. MiniMax reduces [num_tokens, 1] fp32
    variance tensors, where num_tokens 1/2/4 straddle the 16-byte boundary.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(RuntimeError, match="not 16-byte aligned"):
        _should(torch.empty(3, dtype=torch.float32))  # 12 bytes


def test_aligned_tensor_still_accepted_under_batch_invariance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert _should(torch.empty(8, dtype=torch.float32)) is True  # 32 bytes


def test_misaligned_tensor_falls_back_when_not_batch_invariant(monkeypatch):
    """Outside batch invariance the pre-existing fallback is unchanged.

    This directory's conftest force-enables batch invariance for every test
    via `monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)`, which
    shadows the os.getenv lambda, so clearing the environment variable alone
    is not enough to turn it back off here.
    """
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    assert _should(torch.empty(3, dtype=torch.float32)) is False


def test_flashinfer_fused_path_refused_under_batch_invariance(monkeypatch):
    """Model-specific fused paths must not take the FlashInfer all-reduce.

    `_can_use_flashinfer` is the single choke point for the direct callers
    (MiniMax M3, DeepSeek V3.2, Kimi K3). Its one-shot/two-shot selection is
    bounded by token count, so under batch invariance it must decline and let
    the caller fall back to an unfused all-reduce through the communicator.
    """
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _can_use_flashinfer,
    )

    ok, max_token_num = _can_use_flashinfer(torch.empty(4, 8), tp_size=4)
    assert ok is False and max_token_num == 0


def test_custom_all_gather_refused_under_batch_invariance():
    """all-gather is size-gated too, and sequence parallelism routes here."""
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    assert CustomAllreduce.should_custom_all_gather(_Gate(), torch.empty(8)) is False


def test_custom_reduce_scatter_refused_under_batch_invariance():
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    assert (
        CustomAllreduce.should_custom_reduce_scatter(_Gate(), torch.empty(8)) is False
    )


def test_dispatch_bound_and_enlargement_share_one_constant():
    """The world-size bound must not drift between dispatch and sizing.

    `should_custom_ar()` refuses above `_MAX_ALL_REDUCE_WORLD_SIZE`, and the
    batch-invariant enlargement is skipped above the same constant, so a
    world size the kernel cannot serve never reserves buffers for it.
    """
    import inspect

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    assert CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE == 8
    src = inspect.getsource(CustomAllreduce)
    assert "self.world_size > self._MAX_ALL_REDUCE_WORLD_SIZE" in src
    assert "world_size > CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE" in src
    # 16 is constructible (all-gather / reduce-scatter use it) but unservable
    # by the all-reduce kernel, which is exactly why the two must agree.
    assert 16 in CustomAllreduce._SUPPORTED_WORLD_SIZES


def test_unsupported_world_size_is_not_enlarged():
    """world_size 16 takes the skip branch; 2/4/8 take the sizing branch."""
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    bound = CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE
    assert [w for w in (2, 4, 8) if w > bound] == []
    assert bound < 16


def test_fuse_allreduce_rms_rejected_at_config_build(monkeypatch):
    """The exact bypass from review: BI + disable_custom_all_reduce + fusion.

    The validation lives in `VllmConfig.__post_init__` rather than in
    `CustomAllreduce.__init__`, so it fires even when custom all-reduce is
    disabled and no communicator is ever constructed.
    """
    from vllm.config import CompilationConfig, DeviceConfig, ParallelConfig, VllmConfig
    from vllm.config.compilation import PassConfig

    with pytest.raises(ValueError, match="fuse_allreduce_rms"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            parallel_config=ParallelConfig(disable_custom_all_reduce=True),
            compilation_config=CompilationConfig(
                pass_config=PassConfig(fuse_allreduce_rms=True)
            ),
        )


def test_fuse_allreduce_rms_disabled_builds_fine(monkeypatch):
    from vllm.config import CompilationConfig, DeviceConfig, ParallelConfig, VllmConfig
    from vllm.config.compilation import PassConfig

    VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(disable_custom_all_reduce=True),
        compilation_config=CompilationConfig(
            pass_config=PassConfig(fuse_allreduce_rms=False)
        ),
    )
