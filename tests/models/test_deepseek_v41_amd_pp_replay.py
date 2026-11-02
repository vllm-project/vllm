# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decoder SWA bounded replay on ROCm under pipeline parallelism.

With PP=2 the second stage starts at layer 20 by default, the last KV source
and the replay cut, so it enters replay with no layer of its own run first.
Other cuts start it at an Engram layer.
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


def _generate(vllm_runner, **kwargs):
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
        **kwargs,
    ) as llm:
        return llm.generate_greedy_logprobs(
            [LONG_PROMPT, SHORT_PROMPT], max_tokens=8, num_logprobs=5
        )


@pytest.mark.skipif(
    not _on_cdna4_with_two_gpus(), reason="needs two CDNA4 GPUs (ROCm V4.1 path)"
)
def test_pp_stage_starting_at_replay_cut(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    outputs = {
        replay: _generate(vllm_runner, enforce_eager=True, swa_bounded_replay=replay)
        for replay in (False, True)
    }
    check_logprobs_close(
        outputs_0_lst=outputs[False],
        outputs_1_lst=outputs[True],
        name_0="no_replay",
        name_1="replay",
    )


@pytest.mark.skipif(
    not _on_cdna4_with_two_gpus(), reason="needs two CDNA4 GPUs (ROCm V4.1 path)"
)
@pytest.mark.parametrize("partition", ["14,26"])
def test_pp_partition_matches_default_cut(vllm_runner, monkeypatch, partition):
    """Stage 1 starting at Engram layer 14 matches the default 20/20 cut, with
    the decode steps on CUDA graphs."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    reference = _generate(vllm_runner)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=_generate(vllm_runner),
        name_0="20,20",
        name_1=partition,
    )
