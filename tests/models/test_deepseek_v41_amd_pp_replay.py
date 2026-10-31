# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decoder SWA bounded replay on ROCm under pipeline parallelism.

With PP=2 the second stage starts at layer 20, the last KV source and the
replay cut, so it enters replay with no layer of its own run first.
"""

import pytest

from vllm.platforms import current_platform

from .utils import check_logprobs_close

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
# Longer than the 128-token window, so prefill builds a replay batch.
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 60
SHORT_PROMPT = "The capital of France is"


def _on_cdna4_with_two_gpus() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4 and current_platform.device_count() >= 2


@pytest.mark.skipif(
    not _on_cdna4_with_two_gpus(), reason="needs two CDNA4 GPUs (ROCm V4.1 path)"
)
def test_pp_stage_starting_at_replay_cut(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    prompts = [LONG_PROMPT, SHORT_PROMPT]
    outputs = {}
    for replay in (False, True):
        with vllm_runner(
            MODEL,
            pipeline_parallel_size=2,
            language_model_only=True,
            moe_backend="aiter",
            engram_config={"cpu_offload": True},
            attention_config={
                "indexer_kv_dtype": "mxfp4",
                "indexer_sparse_logits": True,
            },
            block_size=128,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enforce_eager=True,
            swa_bounded_replay=replay,
        ) as llm:
            outputs[replay] = llm.generate_greedy_logprobs(
                prompts, max_tokens=8, num_logprobs=5
            )

    check_logprobs_close(
        outputs_0_lst=outputs[False],
        outputs_1_lst=outputs[True],
        name_0="no_replay",
        name_1="replay",
    )
