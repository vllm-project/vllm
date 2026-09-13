# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the tensor-parallel all-reduce path (#50136).

Every other test here runs at TP=1, leaving the collective path untested. This
compares a request against itself across batch *compositions* -- repeated
identical runs pass even when a backend is batch-dependent -- and asserts the
custom kernel carried the collectives, since NCCL alone would be invariant for
the pre-existing reason and make a green result ambiguous.
"""

import inspect
import random
from contextlib import nullcontext

import pytest
import torch
from utils import TEST_MODEL, _extract_step_logprobs, skip_unsupported

from tests.utils import multi_gpu_marks
from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.inputs import TokensPrompt

MIN_GPUS = 4
BATCH_SIZES = [17, 65]
PROMPT_LENS = [512, 1024]


def requires_tp4(fn):
    for mark in reversed(multi_gpu_marks(num_gpus=MIN_GPUS)):
        fn = mark(fn)
    return fn


def _ar_state(self):
    """Per-worker: which all-reduce backends are live, and engagement counts."""
    from vllm.distributed.parallel_state import get_tp_group

    dc = get_tp_group().device_communicator
    ca = getattr(dc, "ca_comm", None)
    return {
        "other": getattr(dc, "fi_ar_comm", None) or getattr(dc, "aiter_ar_comm", None),
        "custom": ca is not None and not ca.disabled,
        "calls": getattr(ca, "_custom_ar_calls", 0),
        "largest": getattr(ca, "_custom_ar_max_bytes", 0),
        "max_size": getattr(ca, "max_size", 0),
    }


def _prompts(seed=1234):
    rng = random.Random(seed)
    return (
        [[rng.randrange(1000, 29000) for _ in range(n)] for n in PROMPT_LENS],
        [
            [rng.randrange(1000, 29000) for _ in range(rng.choice(PROMPT_LENS))]
            for _ in range(max(BATCH_SIZES))
        ],
    )


def _run(llm, prompts, params):
    return llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )


@pytest.fixture
def tp4_llm(monkeypatch):
    """Batch-invariant env plus a TP=4 engine whose teardown is guaranteed."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # collective_rpc(callable) requires pickle-based serialization.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    built = []

    def make(**kw):
        built.append(
            LLM(
                model=TEST_MODEL,
                tensor_parallel_size=MIN_GPUS,
                max_model_len=4096,
                gpu_memory_utilization=0.6,
                seed=0,
                **kw,
            )
        )
        return built[-1]

    yield make
    built.clear()


@skip_unsupported
@requires_tp4
@pytest.mark.timeout(1800)
def test_tp_allreduce_is_batch_invariant(tp4_llm):
    """A request is bitwise identical however it is batched, at TP=4."""
    params = SamplingParams(
        temperature=0.0, max_tokens=32, ignore_eos=True, logprobs=1, seed=1234
    )
    needles, filler = _prompts()
    llm = tp4_llm()
    refs = []
    for needle in needles:
        lp, ids = _extract_step_logprobs(_run(llm, [needle], params)[0])
        assert ids is not None, "logprobs were not returned"
        refs.append((lp, ids))

    for size in BATCH_SIZES:
        for idx, needle in enumerate(needles):
            pos = random.Random(size + idx).randrange(size)
            batch = list(filler[: size - 1])
            batch.insert(pos, needle)
            got_lp, got_ids = _extract_step_logprobs(_run(llm, batch, params)[pos])
            at = f"batch={size} pos={pos} needle={idx}"
            assert list(refs[idx][1]) == list(got_ids), f"{at}: token ids"
            # Bitwise, not allclose: sub-ULP drift still breaks the logprob
            # ratios that invariance exists to protect.
            assert torch.equal(refs[idx][0], got_lp), f"{at}: logprobs"

    for rank, s in enumerate(llm.collective_rpc(_ar_state)):
        assert s["custom"] and s["calls"] > 0, (
            f"rank {rank}: custom all-reduce never ran, so NCCL carried it and "
            "the invariance result above is vacuous."
        )
        assert not s["other"], f"rank {rank}: an unaudited backend was live."
        # The point of sizing at init: every all-reduce fits, so the
        # custom-vs-NCCL choice cannot change mid-run.
        assert s["largest"] <= s["max_size"], (
            f"rank {rank}: {s['largest']} B exceeded the {s['max_size']} B "
            "buffer, so the size gate can flip again."
        )


@skip_unsupported
@requires_tp4
@pytest.mark.timeout(1800)
def test_engagement_assertion_can_fail(tp4_llm):
    """Custom path off: output stays invariant but engagement is zero, which is
    what makes the engagement assertion above load-bearing."""
    llm = tp4_llm(disable_custom_all_reduce=True)
    _run(llm, [_prompts()[0][0]], SamplingParams(temperature=0.0, max_tokens=8))
    for rank, s in enumerate(llm.collective_rpc(_ar_state)):
        assert not s["custom"] and s["calls"] == 0, (
            f"rank {rank}: custom all-reduce ran despite being disabled."
        )


# The guard tests below need no GPU, so a regression fails in plain CPU CI.


class _Gate:
    """Stand-in carrying only what should_custom_ar() reads."""

    disabled = False
    world_size = 4
    fully_connected = True
    max_size = 32 * 1024 * 1024
    # Borrowed, not copied: a literal would drift from the real bound.
    _MAX_ALL_REDUCE_WORLD_SIZE = CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE


@pytest.mark.parametrize(
    "batch_invariant,numel,expect",
    # 12 B is not a 16-byte multiple. Under batch invariance that must raise,
    # not fall back: taking NCCL for this tensor only would make backend
    # selection batch-dependent again (MiniMax reduces [num_tokens, 1] fp32,
    # where num_tokens 1/2/4 straddle the boundary).
    [(True, 3, "raise"), (True, 8, True), (False, 3, False)],
)
def test_misalignment_gate(monkeypatch, batch_invariant, numel, expect):
    # conftest force-enables batch invariance with setattr, shadowing the
    # os.getenv lambda, so delenv alone cannot turn it off here.
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    tensor = torch.empty(numel, dtype=torch.float32)
    if expect == "raise":
        with pytest.raises(RuntimeError, match="not 16-byte aligned"):
            CustomAllreduce.should_custom_ar(_Gate(), tensor)
    else:
        assert CustomAllreduce.should_custom_ar(_Gate(), tensor) is expect


def test_flashinfer_fused_path_refused_under_batch_invariance():
    """The choke point for the direct callers (MiniMax M3, DeepSeek V3.2, Kimi
    K3): one-shot/two-shot selection is token-count bounded, so it must decline
    and let the caller fall back through the communicator."""
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _can_use_flashinfer,
    )

    assert _can_use_flashinfer(torch.empty(4, 8), tp_size=4) == (False, 0)


def test_dispatch_bound_and_enlargement_share_one_constant():
    """16 is constructible (all-gather / reduce-scatter use it) but unservable
    by the all-reduce kernel, so both sites must read one constant."""
    bound = CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE
    src = inspect.getsource(CustomAllreduce)
    assert bound == 8 and bound < 16
    assert "self.world_size > self._MAX_ALL_REDUCE_WORLD_SIZE" in src
    assert "world_size > CustomAllreduce._MAX_ALL_REDUCE_WORLD_SIZE" in src
    assert 16 in CustomAllreduce._SUPPORTED_WORLD_SIZES
    assert [w for w in (2, 4, 8) if w > bound] == []


@pytest.mark.parametrize("fuse,rejected", [(True, True), (False, False)])
def test_fuse_allreduce_rms_config_gate(fuse, rejected):
    """The exact bypass from review: BI + disable_custom_all_reduce + fusion.
    Validation lives in VllmConfig.__post_init__, not CustomAllreduce.__init__,
    so it fires even when no communicator is ever constructed."""
    from vllm.config import CompilationConfig, DeviceConfig, ParallelConfig, VllmConfig
    from vllm.config.compilation import PassConfig

    expect = (
        pytest.raises(ValueError, match="fuse_allreduce_rms")
        if rejected
        else nullcontext()
    )
    with expect:
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            parallel_config=ParallelConfig(disable_custom_all_reduce=True),
            compilation_config=CompilationConfig(
                pass_config=PassConfig(fuse_allreduce_rms=fuse)
            ),
        )
