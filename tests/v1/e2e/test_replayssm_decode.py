# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

from typing import Any

import pytest

import vllm.envs as envs
from vllm.v1.metrics.reader import Counter

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

# Mamba2 (Nemotron-3) hybrid.
MAMBA2_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MAMBA2_MTP_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
MODELS = [
    pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]

try:
    from flashinfer.mamba.checkpointing_ssu import CheckpointingSSURunner

    HAS_FLASHINFER_CHECKPOINTING_SSU = CheckpointingSSURunner is not None
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False


def _check_replayssm_parity(
    vllm_runner,
    model_name,
    *,
    tensor_parallel_size=1,
    mamba_backend: str = "triton",
    name_1: str = "replayssm",
    require_v2: bool = False,
):
    # Compare logprobs, not greedy ids: ReplaySSM's fp arithmetic can flip a
    # near-tie. Baseline and ReplaySSM run at the same TP, so TP numerics are
    # common-mode and only ReplaySSM varies.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        tensor_parallel_size=tensor_parallel_size,
        mamba_backend=mamba_backend,
    )
    with vllm_runner(model_name, **common) as llm:
        if require_v2:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        if require_v2:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
        replay = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline",
        name_1=name_1,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_decode_matches_baseline(vllm_runner, model_name):
    _check_replayssm_parity(vllm_runner, model_name)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_replayssm_decode_matches_baseline_tp2(vllm_runner, model_name):
    # Tensor-parallel correctness: ReplaySSM's caches and checkpoint state are
    # sharded per rank, so TP2 decode must still match the baseline at TP2.
    _check_replayssm_parity(vllm_runner, model_name, tensor_parallel_size=2)


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_flashinfer_decode_matches_baseline_v2(
    vllm_runner, model_name, monkeypatch
):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            _check_replayssm_parity(
                vllm_runner,
                model_name,
                mamba_backend="flashinfer",
                name_1="replayssm_flashinfer_v2",
                require_v2=True,
            )
    finally:
        # The context restores the environment before the final cache reset.
        envs.disable_envs_cache()


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_flashinfer_spec_decode_matches_baseline(vllm_runner, model_name):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
        },
    )
    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        replay = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_spec",
        name_1="replayssm_flashinfer_spec",
    )


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@large_gpu_mark(min_gb=40)
def test_replayssm_flashinfer_mtp_matches_baseline_v2(vllm_runner, monkeypatch):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    outputs: dict[str, Any] = {}
    draft_counts: dict[str, int | float] = {}
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            for name, use_replayssm in (("baseline", False), ("replayssm", True)):
                with vllm_runner(
                    MAMBA2_MTP_MODEL,
                    use_replayssm=use_replayssm,
                    replayssm_buffer_len=16,
                    **common,
                ) as llm:
                    assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    outputs[name] = llm.generate_greedy_logprobs(
                        PROMPTS, max_tokens=32, num_logprobs=5
                    )
                    draft_counts[name] = sum(
                        metric.value
                        for metric in llm.llm.get_metrics()
                        if isinstance(metric, Counter)
                        and metric.name == "vllm:spec_decode_num_drafts"
                    )
    finally:
        envs.disable_envs_cache()

    assert all(count > 0 for count in draft_counts.values()), draft_counts
    check_logprobs_close(
        outputs_0_lst=outputs["baseline"],
        outputs_1_lst=outputs["replayssm"],
        name_0="baseline_mtp_v2",
        name_1="replayssm_flashinfer_mtp_v2",
    )


# Prefix spans several mamba blocks; prefix caching only reuses full blocks.
_PC_SENTENCE = (
    "In a detailed survey of state space models, the authors compared many "
    "architectures across a wide range of long-context language tasks and "
    "measured their throughput, memory use, and accuracy in careful detail. "
)
_PC_PREFIX = _PC_SENTENCE * 120
PREFIX_CACHING_PROMPTS = [
    _PC_PREFIX + "The most important conclusion was that",
    _PC_PREFIX + "Surprisingly, the experiments showed that",
    _PC_PREFIX + "The most important conclusion was that",
]


def _prefix_cache_hits(llm) -> int:
    return sum(
        m.value
        for m in llm.llm.get_metrics()
        if isinstance(m, Counter) and m.name == "vllm:prefix_cache_hits"
    )


def _check_replayssm_prefix_caching_parity(
    vllm_runner, model_name, *, tensor_parallel_size=1
):
    # align mode materializes the exact SSM state at each block boundary, so
    # ReplaySSM's cached prefixes must match the always-materialized baseline.
    common = dict(
        max_model_len=8192,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        disable_log_stats=False,  # required for llm.get_metrics()
        tensor_parallel_size=tensor_parallel_size,
    )
    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        # Prime the cache, then measure, so cache hits are deterministic.
        llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        replay = llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        replay_hits = _prefix_cache_hits(llm)

    # Without real cache hits the cached path is never exercised.
    assert replay_hits > 0, (
        "ReplaySSM align-mode run produced no prefix-cache hits; the shared "
        "prefix may be shorter than one mamba block, so prefix caching is inert"
    )
    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_align_pc",
        name_1="replayssm_align_pc",
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_prefix_caching_matches_baseline(vllm_runner, model_name):
    _check_replayssm_prefix_caching_parity(vllm_runner, model_name)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_replayssm_prefix_caching_matches_baseline_tp2(vllm_runner, model_name):
    _check_replayssm_prefix_caching_parity(
        vllm_runner, model_name, tensor_parallel_size=2
    )
