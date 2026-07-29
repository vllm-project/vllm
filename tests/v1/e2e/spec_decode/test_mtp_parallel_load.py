# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tests.utils import multi_gpu_marks
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.reader import Metric

DEEPSEEK_MTP_MAIN_RANDOM = "luccafong/deepseek_mtp_main_random"
DEEPSEEK_MTP_DRAFT_RANDOM = "luccafong/deepseek_mtp_draft_random"

PROMPT = "The capital of France is"
MAX_TOKENS = 8
MAX_MODEL_LEN = 2048
GPU_MEM_UTIL = 0.85
# Spec / no-spec greedy output should match, but exact match is not guaranteed
# across parallelism layouts (reduction order can flip near-tie argmaxes), so
# gate on a similarity ratio like the spec-decode E2E tests.
MIN_MATCH_RATIO = 0.8


def _token_match_ratio(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a, b)) / n if n else 0.0


@dataclass(frozen=True)
class InlineConfig:
    id: str
    tp_size: int = 1
    pp_size: int = 1
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    skip_reason: str | None = None


INLINE_CONFIGS = [
    InlineConfig(id="tp2", tp_size=2),
    InlineConfig(id="ep2", tp_size=2, enable_expert_parallel=True),
    InlineConfig(
        id="ep2_eplb",
        tp_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
    ),
    InlineConfig(
        id="pp2",
        pp_size=2,
        skip_reason=(
            "DeepSeek MTP pipeline-parallel support is in flight upstream; "
            "see https://github.com/vllm-project/vllm/pull/38104"
        ),
    ),
]


def _inline_kwargs(config: InlineConfig, *, with_spec: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": DEEPSEEK_MTP_MAIN_RANDOM,
        "tensor_parallel_size": config.tp_size,
        "pipeline_parallel_size": config.pp_size,
        "enable_expert_parallel": config.enable_expert_parallel,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "enforce_eager": True,
        "trust_remote_code": True,
        "disable_log_stats": False,
        # MLA + batch invariance currently disables prefix caching at runtime
        # (see vllm/v1/attention/backends/mla/common.py); set explicitly so the
        # configuration is unambiguous.
        "enable_prefix_caching": False,
    }
    if config.enable_eplb:
        kwargs["enable_eplb"] = True
        # Rearrangement first fires after step_interval // 4 forward steps
        # (the step counter starts at 3/4 of step_interval, see
        # vllm/distributed/eplb/eplb_state.py). Keep these small so the EPLB
        # routine actually runs within the short MAX_TOKENS generation instead
        # of never triggering.
        kwargs["eplb_config"] = {
            "num_redundant_experts": config.tp_size,
            "window_size": 2,
            "step_interval": 4,
            "log_balancedness": False,
        }
    if with_spec:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "model": DEEPSEEK_MTP_DRAFT_RANDOM,
            "num_speculative_tokens": 1,
        }
    return kwargs


def _generate_token_ids_inline(llm: LLM) -> tuple[int, ...]:
    outputs = llm.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True),
    )
    assert outputs and outputs[0].outputs, "expected one completion"
    return tuple(outputs[0].outputs[0].token_ids)


def _spec_decode_num_drafts(metrics: list[Metric]) -> int:
    name2metric = {m.name: m for m in metrics}
    counter = name2metric.get("vllm:spec_decode_num_drafts")
    assert counter is not None, (
        "spec_decode_num_drafts metric missing; check disable_log_stats=False"
    )
    return int(counter.value)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(c, id=c.id, marks=multi_gpu_marks(num_gpus=2))
        for c in INLINE_CONFIGS
    ],
)
def test_deepseek_mtp_load_inline(
    monkeypatch: pytest.MonkeyPatch,
    config: InlineConfig,
):
    """MTP loads and drafts under TP/EP/EPLB; spec output matches no-spec greedy."""
    if config.skip_reason is not None:
        pytest.skip(config.skip_reason)

    # Reduces run-to-run nondeterminism so spec / no-spec stay close;
    # see tests/v1/distributed/test_eagle_dp.py for the same pattern.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    spec_llm = LLM(**_inline_kwargs(config, with_spec=True))
    try:
        spec_tokens = _generate_token_ids_inline(spec_llm)
        n_drafts = _spec_decode_num_drafts(spec_llm.get_metrics())
    finally:
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

    no_spec_llm = LLM(**_inline_kwargs(config, with_spec=False))
    try:
        no_spec_tokens = _generate_token_ids_inline(no_spec_llm)
    finally:
        del no_spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

    # Non-vacuity: a silently-broken MTP drafter that falls back to
    # verifier-only decoding would still produce matching output below.
    assert n_drafts > 0, (
        f"MTP drafter never fired under {config.id}: vllm:spec_decode_num_drafts == 0"
    )
    match_ratio = _token_match_ratio(spec_tokens, no_spec_tokens)
    print(f"\n{config.id}: spec/no-spec match_ratio={match_ratio:.3f}")
    assert match_ratio >= MIN_MATCH_RATIO, (
        f"Spec / no-spec output divergence under {config.id}: "
        f"match_ratio={match_ratio:.2f} < {MIN_MATCH_RATIO}.\n"
        f"  spec_tokens   = {spec_tokens}\n"
        f"  no_spec_tokens= {no_spec_tokens}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dp_size",
    [
        pytest.param(2, marks=pytest.mark.distributed(num_gpus=2), id="dp2"),
        pytest.param(1, id="dp1"),
    ],
)
async def test_deepseek_mtp_load_dp(monkeypatch: pytest.MonkeyPatch, dp_size: int):
    """MTP loads under DP (on and off) via AsyncLLM; spec matches no-spec greedy."""
    if torch.accelerator.device_count() < dp_size:
        pytest.skip(f"dp{dp_size} requires at least {dp_size} GPUs")

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    base_args = AsyncEngineArgs(
        model=DEEPSEEK_MTP_MAIN_RANDOM,
        tensor_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_backend="mp",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    spec_args = replace(
        base_args,
        speculative_config={
            "method": "mtp",
            "model": DEEPSEEK_MTP_DRAFT_RANDOM,
            "num_speculative_tokens": 1,
        },
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        temperature=0.0,
    )

    async def _generate(args: AsyncEngineArgs, request_id: str) -> tuple[int, ...]:
        try:
            async with AsyncExitStack() as stack:
                engine = AsyncLLM.from_engine_args(args)
                stack.callback(engine.shutdown)
                async for out in engine.generate(
                    request_id=request_id,
                    prompt=PROMPT,
                    sampling_params=sampling_params,
                ):
                    token_ids = tuple(out.outputs[0].token_ids)
                    assert len(token_ids) == MAX_TOKENS
                    return token_ids
            raise AssertionError("AsyncLLM produced no output")
        finally:
            torch.accelerator.empty_cache()
            cleanup_dist_env_and_memory()

    spec_tokens = await asyncio.wait_for(
        _generate(spec_args, f"deepseek-mtp-dp{dp_size}-spec"), timeout=600
    )
    no_spec_tokens = await asyncio.wait_for(
        _generate(base_args, f"deepseek-mtp-dp{dp_size}-no-spec"), timeout=600
    )

    match_ratio = _token_match_ratio(spec_tokens, no_spec_tokens)
    print(f"\ndp{dp_size}: spec/no-spec match_ratio={match_ratio:.3f}")
    assert match_ratio >= MIN_MATCH_RATIO, (
        f"Spec / no-spec output divergence under dp{dp_size}: "
        f"match_ratio={match_ratio:.2f} < {MIN_MATCH_RATIO}.\n"
        f"  spec_tokens   = {spec_tokens}\n"
        f"  no_spec_tokens= {no_spec_tokens}"
    )
