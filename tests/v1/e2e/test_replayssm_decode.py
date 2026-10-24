# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import os
from inspect import signature

import pytest

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
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

if os.environ.get("VLLM_TEST_REQUIRE_REPLAYSSM") == "1":
    if not (HAS_FLASHINFER_CHECKPOINTING_SSU and HAS_FLASHINFER_REPLAYSSM_MATERIALIZE):
        raise RuntimeError(
            "Dedicated ReplaySSM CI requires FlashInfer 0.7.0 checkpointing and "
            "materialization APIs, including active_request_indices"
        )
    if (
        not current_platform.is_cuda()
        or current_platform.device_count() < 2
        or any(
            current_platform.get_device_total_memory(device) < 40_000_000_000
            for device in range(2)
        )
    ):
        raise RuntimeError(
            "Dedicated ReplaySSM CI requires two CUDA GPUs with at least 40 GB each"
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
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replayssm_flashinfer_spec_decode_matches_baseline(
    vllm_runner, model_name, monkeypatch, use_v2_model_runner
):
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
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
            envs.disable_envs_cache()
            with vllm_runner(model_name, **common) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                baseline = llm.generate_greedy_logprobs(
                    FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
                )
            with vllm_runner(
                model_name,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                replay = llm.generate_greedy_logprobs(
                    FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
                )
    finally:
        envs.disable_envs_cache()

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


# Prefix exceeds the FP32 FlashInfer ReplaySSM block at TP1.
# Prefix caching only reuses full blocks.
_PC_SENTENCE = (
    "In a detailed survey of state space models, the authors compared many "
    "architectures across a wide range of long-context language tasks and "
    "measured their throughput, memory use, and accuracy in careful detail. "
)
_PC_PREFIX = _PC_SENTENCE * 160
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
    replayssm_state_dtype: str = "auto",
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
    if replayssm_state_dtype != "auto":
        common.update(
            dtype="bfloat16",
            mamba_ssm_cache_dtype="auto",
            enable_mamba_cache_stochastic_rounding=False,
        )
    outputs = []
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0")
            envs.disable_envs_cache()
            for use_replayssm in (False, True):
                with vllm_runner(
                    model_name,
                    use_replayssm=use_replayssm,
                    replayssm_buffer_len=16,
                    replayssm_state_dtype=(
                        replayssm_state_dtype if use_replayssm else "auto"
                    ),
                    **common,
                ) as llm:
                    config = llm.llm.llm_engine.vllm_config
                    assert config.use_v2_model_runner is use_v2
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

    name = f"{mamba_backend}_align_{speculative_method or 'stp'}_v{2 if use_v2 else 1}"
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0=f"baseline_{name}_cached",
        name_1=f"replayssm_{name}_cached",
    )
    if speculative_method == "mtp" or replayssm_state_dtype != "auto":
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


@pytest.mark.parametrize("read_dtype", ["fp8", "int8"])
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
@large_gpu_mark(min_gb=40)
def test_quantized_replayssm_prefix_cache(vllm_runner, monkeypatch, read_dtype, use_v2):
    """Compare quantized reads across prefill, decode, and prefix-cache reuse."""
    _check_replayssm_prefix_caching(
        vllm_runner,
        MAMBA2_MTP_MODEL,
        monkeypatch,
        use_v2=use_v2,
        speculative_method="ngram",
        replayssm_state_dtype=read_dtype,
    )


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability(103)
        or current_platform.is_device_capability(121)
    ),
    reason="Separate-checkpoint kernels require SM103 or SM121",
)
@pytest.mark.parametrize("read_dtype", ["fp8", "nvfp4"])
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_separate_replayssm_decode_and_slot_reuse(
    vllm_runner, monkeypatch, read_dtype, use_v2
):
    """Exercise chunked prefill, two flushes, padded decode, and reused slots."""
    model_name = os.environ.get("VLLM_TEST_REPLAYSSM_SEPARATE_MODEL", MAMBA2_MTP_MODEL)
    revision = os.environ.get("VLLM_TEST_REPLAYSSM_SEPARATE_MODEL_REVISION")
    model_kwargs = {}
    if current_platform.is_device_capability(121):
        # GB10 NVFP4 model execution uses the validated B12x backend profile.
        model_kwargs = dict(
            linear_backend="b12x",
            moe_backend="b12x",
            attention_backend="FLASHINFER",
            trust_remote_code=True,
            language_model_only=True,
            kv_cache_dtype="fp8",
            mamba_cache_dtype="bfloat16",
        )
    prefill_budget = 8
    prompts = [
        "The observatory records the temperature and wind speed every hour. " * 16
        + "Explain why regular weather measurements are useful.",
        FLASHINFER_PROMPTS[0],
    ]
    outputs = []
    with monkeypatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
        envs.disable_envs_cache()
        try:
            for enabled in (False, True):
                with vllm_runner(
                    model_name,
                    revision=revision,
                    dtype="bfloat16",
                    max_model_len=1024,
                    max_num_seqs=8,
                    max_num_batched_tokens=prefill_budget,
                    enable_chunked_prefill=True,
                    async_scheduling=False,
                    enable_prefix_caching=False,
                    mamba_backend="flashinfer",
                    mamba_ssm_cache_dtype="float16",
                    enable_mamba_cache_stochastic_rounding=True,
                    mamba_cache_philox_rounds=7,
                    use_replayssm=enabled,
                    replayssm_state_dtype=read_dtype if enabled else "auto",
                    replayssm_checkpoint_mode="separate" if enabled else "shared",
                    compilation_config={"cudagraph_capture_sizes": [1, 2, 8]},
                    **model_kwargs,
                ) as llm:
                    assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
                    tokenizer = llm.llm.get_tokenizer()
                    assert prefill_budget < len(tokenizer.encode(prompts[0])) <= 984
                    first = llm.generate_greedy_logprobs(
                        prompts, max_tokens=40, num_logprobs=5
                    )
                    reused = llm.generate_greedy_logprobs(
                        prompts[:1], max_tokens=40, num_logprobs=5
                    )
                    # Seven short requests leave one token of prefill capacity.
                    # Their final decode leaves another one-token prompt chunk.
                    # After they finish, the long request resumes larger chunks.
                    token = tokenizer.encode("Hello", add_special_tokens=False)[0]
                    mixed_prompts = [{"prompt_token_ids": [token]} for _ in range(7)]
                    mixed_prompts.append(
                        {"prompt_token_ids": tokenizer.encode(prompts[0])}
                    )
                    short_params = SamplingParams(
                        temperature=0, max_tokens=2, ignore_eos=True, logprobs=5
                    )
                    long_params = SamplingParams(
                        temperature=0, max_tokens=40, ignore_eos=True, logprobs=5
                    )
                    mixed = llm.llm.generate(
                        mixed_prompts, [short_params] * 7 + [long_params]
                    )
                    assert [len(result.outputs[0].token_ids) for result in mixed] == [
                        2
                    ] * 7 + [40]
                    mixed_logprobs = [
                        (
                            list(result.outputs[0].token_ids),
                            result.outputs[0].text,
                            result.outputs[0].logprobs,
                        )
                        for result in mixed
                    ]
                    outputs.append((first, reused, mixed_logprobs))
        finally:
            envs.disable_envs_cache()
    for batch in range(3):
        check_logprobs_close(
            outputs_0_lst=outputs[0][batch],
            outputs_1_lst=outputs[1][batch],
            name_0="baseline_fp16_sr",
            name_1=f"separate_{read_dtype}_batch_{batch}",
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
