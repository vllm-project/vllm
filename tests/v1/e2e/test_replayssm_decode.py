# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import os
from inspect import signature

import pytest

import vllm.envs as envs
from vllm.v1.metrics.reader import Counter

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

try:
    from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
        CheckpointingSSURunner,
        allocate_checkpointing_ssu_scratch,
    )

    HAS_FLASHINFER_CHECKPOINTING_SSU = True
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False

try:
    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = (
        "active_request_indices" in signature(replayssm_materialize).parameters
    )
except ImportError:
    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = False

if os.environ.get("VLLM_TEST_REQUIRE_REPLAYSSM") == "1" and not (
    HAS_FLASHINFER_CHECKPOINTING_SSU and HAS_FLASHINFER_REPLAYSSM_MATERIALIZE
):
    raise RuntimeError(
        "Dedicated ReplaySSM CI requires FlashInfer 0.7.0 checkpointing and "
        "materialization APIs, including active_request_indices"
    )

# Mamba2 (Nemotron-3) hybrid.
MAMBA2_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MAMBA2_MTP_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
FLASHINFER_MODELS = [pytest.param(MAMBA2_MTP_MODEL, marks=large_gpu_mark(min_gb=40))]
MODELS = [
    pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]
# The NVFP4 model is not reproducible across engine instances; these prompts
# avoid near-ties and decode confidently past the replay window.
FLASHINFER_PROMPTS = [
    "The capital of France is",
    "It was the best of times, it was the worst of times,",
]

requires_flashinfer_replayssm_materialization = pytest.mark.skipif(
    not (HAS_FLASHINFER_CHECKPOINTING_SSU and HAS_FLASHINFER_REPLAYSSM_MATERIALIZE),
    reason="FlashInfer ReplaySSM materialization APIs not available",
)


@pytest.fixture(autouse=True)
def _use_v1_model_runner_by_default(monkeypatch):
    # Triton ReplaySSM is V1-only. FlashInfer V2 tests override this locally.
    with monkeypatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        envs.disable_envs_cache()
        yield
    envs.disable_envs_cache()


def _check_replayssm_parity(
    vllm_runner,
    model_name,
    *,
    tensor_parallel_size=1,
    mamba_backend: str = "triton",
    prompts: list[str] = PROMPTS,
    name_1: str = "replayssm",
    expected_v2: bool | None = None,
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
        if expected_v2 is not None:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is expected_v2
        baseline = llm.generate_greedy_logprobs(prompts, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        if expected_v2 is not None:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is expected_v2
        replay = llm.generate_greedy_logprobs(prompts, max_tokens=32, num_logprobs=5)

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


@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replayssm_flashinfer_decode_matches_baseline(
    vllm_runner, model_name, monkeypatch, use_v2_model_runner
):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
            envs.disable_envs_cache()
            _check_replayssm_parity(
                vllm_runner,
                model_name,
                mamba_backend="flashinfer",
                prompts=FLASHINFER_PROMPTS,
                name_1="replayssm_flashinfer",
                expected_v2=use_v2_model_runner,
            )
    finally:
        # The context restores the environment before the final cache reset.
        envs.disable_envs_cache()


@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
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
        baseline = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_spec",
        name_1="replayssm_flashinfer_spec",
    )


@multi_gpu_test(num_gpus=2)
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replayssm_flashinfer_mtp(vllm_runner, monkeypatch, use_v2_model_runner):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        tensor_parallel_size=2,
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
            envs.disable_envs_cache()
            with vllm_runner(MAMBA2_MTP_MODEL, **common) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                baseline = llm.generate_greedy_logprobs(
                    PROMPTS, max_tokens=32, num_logprobs=5
                )
            with vllm_runner(
                MAMBA2_MTP_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                replay = llm.generate_greedy_logprobs(
                    PROMPTS, max_tokens=32, num_logprobs=5
                )
                draft_count = sum(
                    metric.value
                    for metric in llm.llm.get_metrics()
                    if isinstance(metric, Counter)
                    and metric.name == "vllm:spec_decode_num_drafts"
                )
    finally:
        envs.disable_envs_cache()

    assert any(len(output[0]) > 16 for output in replay)
    assert draft_count > 0
    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0=f"baseline_mtp_{'v2' if use_v2_model_runner else 'v1'}",
        name_1=f"replayssm_flashinfer_mtp_{'v2' if use_v2_model_runner else 'v1'}",
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
# MTP prefix caching must retain at least two full state blocks: the drafter drops
# the volatile trailing block before resuming from the preceding boundary.
_PC_MTP_PREFIX = _PC_SENTENCE * 240
MTP_PREFIX_CACHING_PROMPTS = [
    _PC_MTP_PREFIX + prompt.removeprefix(_PC_PREFIX)
    for prompt in PREFIX_CACHING_PROMPTS
]


def _prefix_cache_hits(llm) -> int:
    return sum(
        m.value
        for m in llm.llm.get_metrics()
        if isinstance(m, Counter) and m.name == "vllm:prefix_cache_hits"
    )


def _check_replayssm_prefix_caching(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_v2: bool,
    speculative_method: str | None = None,
    mamba_backend: str = "flashinfer",
):
    common = dict(
        max_model_len=8192,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        mamba_backend=mamba_backend,
        disable_log_stats=False,  # required for llm.get_metrics()
        tensor_parallel_size=1,
    )
    prompts = PREFIX_CACHING_PROMPTS
    if speculative_method is not None:
        speculative_config = {
            "method": speculative_method,
            "num_speculative_tokens": 3,
        }
        if speculative_method == "ngram":
            speculative_config["prompt_lookup_max"] = 3
        common["speculative_config"] = speculative_config
    if speculative_method == "mtp":
        prompts = MTP_PREFIX_CACHING_PROMPTS
        common.update(
            max_model_len=12288,
            max_num_seqs=4,
            dtype="bfloat16",
            mamba_ssm_cache_dtype="float16",
            enable_mamba_cache_stochastic_rounding=True,
            mamba_cache_philox_rounds=5,
        )
    outputs = []
    block_sizes = []
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0")
            envs.disable_envs_cache()
            for use_replayssm in (False, True):
                with vllm_runner(
                    model_name,
                    use_replayssm=use_replayssm,
                    replayssm_buffer_len=16,
                    **common,
                ) as llm:
                    config = llm.llm.llm_engine.vllm_config
                    assert config.use_v2_model_runner is use_v2
                    block_sizes.append(config.cache_config.block_size)
                    first_pass = llm.generate_greedy_logprobs(
                        prompts, max_tokens=32, num_logprobs=5
                    )
                    hits_before = _prefix_cache_hits(llm)
                    cached = llm.generate_greedy_logprobs(
                        prompts, max_tokens=32, num_logprobs=5
                    )
                    assert _prefix_cache_hits(llm) > hits_before, (
                        f"No prefix-cache hits with use_replayssm={use_replayssm}"
                    )
                    outputs.append(cached)
                    if use_replayssm and speculative_method is not None:
                        metric_names = ["vllm:spec_decode_num_drafts"]
                        if speculative_method == "mtp":
                            metric_names.append("vllm:spec_decode_num_accepted_tokens")
                        for metric_name in metric_names:
                            assert (
                                sum(
                                    metric.value
                                    for metric in llm.llm.get_metrics()
                                    if isinstance(metric, Counter)
                                    and metric.name == metric_name
                                )
                                > 0
                            )
    finally:
        envs.disable_envs_cache()

    if mamba_backend == "flashinfer":
        # Auxiliary FlashInfer rings cannot affect the shared page.
        assert block_sizes[1] == block_sizes[0]
    else:
        # Triton's packed rings may require larger attention blocks.
        assert block_sizes[1] >= block_sizes[0]
    name = f"{mamba_backend}_align_{speculative_method or 'stp'}_v{2 if use_v2 else 1}"
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0=f"baseline_{name}_cached",
        name_1=f"replayssm_{name}_cached",
    )
    if speculative_method == "mtp":
        check_logprobs_close(
            outputs_0_lst=first_pass,
            outputs_1_lst=cached,
            name_0=f"replayssm_{name}_first_pass",
            name_1=f"replayssm_{name}_cached",
        )


@requires_flashinfer_replayssm_materialization
@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
@pytest.mark.parametrize(
    ("use_v2", "use_ngram"),
    [
        pytest.param(False, True, id="align-v1-ngram-t4"),
        pytest.param(True, False, id="align-v2-stp"),
    ],
)
def test_flashinfer_replayssm_prefix_cache_tp1(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    use_v2: bool,
    use_ngram: bool,
):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        speculative_method="ngram" if use_ngram else None,
        use_v2=use_v2,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_triton_replayssm_align_prefix_cache_matches_baseline_v1(
    vllm_runner, model_name, monkeypatch: pytest.MonkeyPatch
):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        use_v2=False,
        mamba_backend="triton",
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize(
    "use_v2",
    [False, True],
    ids=["v1-align", "v2-align"],
)
def test_flashinfer_replayssm_prefix_cache_mtp(vllm_runner, monkeypatch, use_v2):
    _check_replayssm_prefix_caching(
        vllm_runner,
        MAMBA2_MTP_MODEL,
        monkeypatch,
        use_v2=use_v2,
        speculative_method="mtp",
    )
