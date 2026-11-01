# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request skipped for one step must keep its hybrid spec-decode state.

The GDN layers read the recurrent state from slot ``num_accepted_tokens - 1``,
so re-adding a skipped request must not reset that count.
"""

from typing import Any

import pytest

import vllm.envs as envs
from tests.utils import single_gpu_only
from vllm import SamplingParams

from .... import skipped_step_scheduler
from ....skipped_step_scheduler import (
    SkipStepOnceAsyncScheduler,
    SkipStepOnceScheduler,
)

MODEL = "Qwen/Qwen3.5-0.8B-Base"
PROMPTS = [
    "Write a detailed, step-by-step explanation of how a binary search tree "
    "insertion works, with an example inserting 50, 30, 70, 20, 40, 60, 80. Begin:",
    "List the planets of the solar system and one fact about each.",
    "Explain how a hash table works, step by step.",
]
SKIP_AT = 12


@pytest.mark.parametrize(
    "mode",
    [
        {},
        {"async_scheduling": True, "scheduler_cls": SkipStepOnceAsyncScheduler},
        {
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "enable_chunked_prefill": True,
        },
    ],
    ids=["sync", "async", "align"],
)
@single_gpu_only
def test_mtp_skipped_step_matches_no_skip(vllm_runner, monkeypatch, mode):
    kwargs: dict[str, Any] = dict(
        max_model_len=1024,
        max_num_seqs=len(PROMPTS),
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        scheduler_cls=SkipStepOnceScheduler,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    kwargs.update(mode)

    def params(skip_at):
        return SamplingParams(
            temperature=0.0,
            max_tokens=64,
            ignore_eos=True,
            extra_args=None if skip_at is None else {"skip_step_at": skip_at},
        )

    skipped_step_scheduler.skipped.clear()
    try:
        with monkeypatch.context() as m:
            m.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
            m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            envs.disable_envs_cache()
            with vllm_runner(MODEL, **kwargs) as runner:
                ref = runner.llm.generate(PROMPTS, params(None))
                runner.llm.reset_prefix_cache()
                # Only the first request is held back; the others keep running,
                # so it comes back to a different batch row.
                out = runner.llm.generate(
                    PROMPTS, [params(SKIP_AT)] + [params(None)] * (len(PROMPTS) - 1)
                )
    finally:
        envs.disable_envs_cache()

    assert len(skipped_step_scheduler.skipped) == 1, skipped_step_scheduler.skipped
    for r, o in zip(ref, out):
        assert list(o.outputs[0].token_ids) == list(r.outputs[0].token_ids)
