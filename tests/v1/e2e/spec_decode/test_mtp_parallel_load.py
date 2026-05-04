# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-parallelism weight-loading + basic correctness tests for DeepSeek MTP.

The existing ``test_mtp_correctness[deepseek]`` covers TP=1 only. This file
adds non-TP=1 coverage:
  * TP=2, EP=2, EP=2 + EPLB via the inline ``LLM`` constructor.
  * DP=2 via ``AsyncLLM`` (mirrors ``tests/v1/distributed/test_eagle_dp.py``,
    since inline ``LLM(data_parallel_size>1)`` is rejected at
    ``vllm/entrypoints/llm.py:333``).
  * PP=2 statically skipped pending upstream support
    (https://github.com/vllm-project/vllm/pull/38104).

For each cell we build two engines at the same parallelism shape (one with
the MTP drafter, one without), run greedy decode on the same prompt under
``VLLM_BATCH_INVARIANT=1``, and assert the output token ids match exactly.
For the inline cells we additionally check
``vllm:spec_decode_num_drafts > 0`` from ``llm.get_metrics()`` so a
silently-broken drafter that falls through to verifier-only decoding is not
an automatic pass. ``AsyncLLM`` does not expose ``get_metrics()``, so the
DP cell relies on the same exact-match guard upstream's eagle DP test uses.

Random-weight checkpoint (``luccafong/deepseek_mtp_main_random``, 5+1
layers, ~7.3 GB FP8) keeps the test fast and within a 2-GPU shape.

Closes a coverage gap that allowed
https://github.com/vllm-project/vllm/pull/29545 (TP+EP load-time crash) to
land without an upstream regression gate before merge.
"""

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
    }
    if config.enable_eplb:
        kwargs["enable_eplb"] = True
        kwargs["eplb_config"] = {
            "num_redundant_experts": config.tp_size,
            "window_size": 128,
            "step_interval": 1024,
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
    """
    Smoke-load the random DeepSeek-MTP-shaped checkpoint at non-trivial
    parallelism using the inline ``LLM`` constructor. Assert (a) the spec
    output matches the no-spec output exactly under greedy + batch-invariant,
    and (b) the MTP drafter actually fired.
    """
    if config.skip_reason is not None:
        pytest.skip(config.skip_reason)

    # DeepSeek MTP testing disables MLA; matches the existing
    # test_mtp_correctness setup.
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    # Required for exact-match equality between the spec and no-spec runs;
    # see https://github.com/vllm-project/vllm/pull/30018 and
    # tests/v1/distributed/test_eagle_dp.py for the same pattern.
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
    assert spec_tokens == no_spec_tokens, (
        f"Spec / no-spec output divergence under {config.id}.\n"
        f"  spec_tokens   = {spec_tokens}\n"
        f"  no_spec_tokens= {no_spec_tokens}"
    )


@pytest.mark.asyncio
@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="DP=2 cell requires at least 2 GPUs",
)
async def test_deepseek_mtp_load_dp(monkeypatch: pytest.MonkeyPatch):
    """
    Load the random DeepSeek-MTP-shaped checkpoint with data_parallel_size=2
    via ``AsyncLLM`` (the only supported path for DP) and assert the spec
    output matches the no-spec output exactly under greedy + batch-invariant.

    Mirrors ``tests/v1/distributed/test_eagle_dp.py``. ``AsyncLLM`` does not
    expose ``get_metrics()``, so the non-vacuity guard from the inline cells
    is omitted here; output equality across spec / no-spec is the gate.
    """
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    base_args = AsyncEngineArgs(
        model=DEEPSEEK_MTP_MAIN_RANDOM,
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_backend="mp",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,
        trust_remote_code=True,
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

    spec_tokens = await asyncio.wait_for(
        _generate(spec_args, "deepseek-mtp-dp-spec"), timeout=600
    )
    no_spec_tokens = await asyncio.wait_for(
        _generate(base_args, "deepseek-mtp-dp-no-spec"), timeout=600
    )

    assert spec_tokens == no_spec_tokens, (
        "Spec / no-spec output divergence under dp2.\n"
        f"  spec_tokens   = {spec_tokens}\n"
        f"  no_spec_tokens= {no_spec_tokens}"
    )
