# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import os
from inspect import signature

import pytest

import vllm.envs as envs
from vllm import SamplingParams
from vllm.platforms import current_platform
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
    tensor_parallel_size: int = 1,
):
    common = dict(
        max_model_len=8192,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        mamba_backend=mamba_backend,
        disable_log_stats=False,  # required for llm.get_metrics()
        tensor_parallel_size=tensor_parallel_size,
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
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0")
            envs.disable_envs_cache()
            for use_replayssm in (False, True):
                with vllm_runner(
                    model_name,
                    use_replayssm=use_replayssm,
                    replayssm_buffer_len=16,
                    # Shared prompts must not populate the baseline's cache.
                    enable_prefix_caching=use_replayssm,
                    **common,
                ) as llm:
                    config = llm.llm.llm_engine.vllm_config
                    assert config.use_v2_model_runner is use_v2
                    first_pass = llm.generate_greedy_logprobs(
                        prompts, max_tokens=32, num_logprobs=5
                    )
                    if not use_replayssm:
                        assert _prefix_cache_hits(llm) == 0
                        outputs.append(first_pass)
                        continue
                    hits_before = _prefix_cache_hits(llm)
                    before = _metric_counts(llm.llm)
                    cached = llm.generate_greedy_logprobs(
                        prompts, max_tokens=32, num_logprobs=5
                    )
                    assert _prefix_cache_hits(llm) > hits_before, (
                        f"No prefix-cache hits with use_replayssm={use_replayssm}"
                    )
                    outputs.append(cached)
                    if speculative_method is not None:
                        after = _metric_counts(llm.llm)
                        accepted = (
                            after["vllm:spec_decode_num_accepted_tokens"]
                            - before["vllm:spec_decode_num_accepted_tokens"]
                        )
                        drafted = (
                            after["vllm:spec_decode_num_draft_tokens"]
                            - before["vllm:spec_decode_num_draft_tokens"]
                        )
                        assert 0 < accepted < drafted, (accepted, drafted)
    finally:
        envs.disable_envs_cache()

    name = f"{mamba_backend}_align_{speculative_method or 'stp'}_v{2 if use_v2 else 1}"
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0=f"baseline_{name}_uncached",
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


@requires_flashinfer_replayssm_materialization
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
def test_flashinfer_replayssm_prefix_cache_tp2(vllm_runner, model_name, monkeypatch):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        use_v2=True,
        tensor_parallel_size=2,
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


def _metric_counts(llm) -> dict[str, int]:
    return {
        metric.name: metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter)
    }


def _stress_mamba_block_size(worker) -> int:
    from vllm.v1.kv_cache_interface import MambaSpec, iter_layer_specs

    sizes = {
        spec.block_size
        for group in worker.model_runner.kv_cache_config.kv_cache_groups
        for spec in iter_layer_specs(group.kv_cache_spec)
        if isinstance(spec, MambaSpec)
    }
    assert len(sizes) == 1, sizes
    return sizes.pop()


def _generate_with_preemption(llm, block_size: int):
    """Preempt active requests, then complete decode across three cache blocks."""
    engine = llm.llm_engine
    # MTP excludes one trailing block from prefix hits. Keep three reusable
    # blocks, plus that proof block and a decode margin.
    max_tokens = 4 * block_size + 64
    requests = {
        "short": ("The capital of France is", 256),
        "long": (
            "Write a detailed history of mathematics, starting in antiquity.",
            max_tokens,
        ),
    }
    prompt_length = len(llm.get_tokenizer().encode(requests["long"][0]))
    assert (
        prompt_length + max_tokens + 8 <= engine.vllm_config.model_config.max_model_len
    )
    before = _metric_counts(llm)
    # Queue both requests before scheduling either engine's initial batch.
    llm.sleep(level=0, mode="keep")
    try:
        for request_id, (prompt, length) in requests.items():
            engine.add_request(
                request_id,
                prompt,
                SamplingParams(temperature=0.0, max_tokens=length, ignore_eos=True),
            )
    finally:
        llm.wake_up(tags=["scheduling"])
    latest = {}
    preempted = False
    observed_turnover = False
    for _ in range(max_tokens + 1024):
        for output in engine.step():
            latest[output.request_id] = output
        if len(latest) != len(requests):
            continue
        short, long = latest["short"], latest["long"]
        if not preempted and len(long.outputs[0].token_ids) >= 32:
            assert not short.finished and not long.finished
            assert llm.reset_prefix_cache(reset_running_requests=True)
            preempted = True
        observed_turnover |= short.finished and not long.finished
        if all(output.finished for output in latest.values()):
            break
    assert preempted and observed_turnover
    assert all(output.finished for output in latest.values())
    after = _metric_counts(llm)
    assert after["vllm:num_preemptions"] - before["vllm:num_preemptions"] >= 2
    for request_id, (_, length) in requests.items():
        output = latest[request_id]
        completion = output.outputs[0]
        assert len(completion.token_ids) == length
        assert completion.finish_reason == "length"
        if request_id == "long":
            prompt_len = len(output.prompt_token_ids)
            crossed = (prompt_len + length - 1) // block_size - prompt_len // block_size
            assert crossed >= 3, (prompt_len, length, block_size, crossed)
            print(
                f"ReplaySSM stress: block_size={block_size}, crossings={crossed}, "
                "preemptions="
                f"{after['vllm:num_preemptions'] - before['vllm:num_preemptions']}"
            )
    long = latest["long"]
    assert long.prompt_token_ids is not None
    history = long.prompt_token_ids + list(long.outputs[0].token_ids)
    accepted = after["vllm:spec_decode_num_accepted_tokens"] - before.get(
        "vllm:spec_decode_num_accepted_tokens", 0
    )
    drafted = after["vllm:spec_decode_num_draft_tokens"] - before.get(
        "vllm:spec_decode_num_draft_tokens", 0
    )
    assert 0 < accepted < drafted, (accepted, drafted)
    return history


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize(
    ("use_v2", "async_scheduling"),
    [(False, False), (False, True), (True, True)],
    ids=["v1-sync", "v1-async", "v2-async"],
)
@pytest.mark.usefixtures("enable_pickle")
def test_flashinfer_replayssm_mtp_preemption_and_boundary_reuse(
    vllm_runner, monkeypatch, use_v2, async_scheduling
):
    """Compare accepted history after preemption with a fresh uncached reference."""
    boundary_outputs = {}
    probe_history = None
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
            envs.disable_envs_cache()
            # Retain the accepted ON history for the fresh OFF reference.
            for use_replayssm in (True, False):
                with vllm_runner(
                    MAMBA2_MTP_MODEL,
                    trust_remote_code=True,
                    max_model_len=32_768,
                    max_num_seqs=4,
                    max_num_batched_tokens=2048,
                    tensor_parallel_size=1,
                    dtype="bfloat16",
                    mamba_backend="flashinfer",
                    mamba_ssm_cache_dtype="float16",
                    mamba_cache_mode="align",
                    enable_prefix_caching=True,
                    # Retain decode checkpoints for the same-history probe.
                    prefix_cache_retention_interval=None,
                    enable_chunked_prefill=True,
                    enable_mamba_cache_stochastic_rounding=True,
                    mamba_cache_philox_rounds=5,
                    async_scheduling=async_scheduling,
                    disable_log_stats=False,
                    speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                    use_replayssm=use_replayssm,
                    replayssm_buffer_len=16,
                ) as runner:
                    llm = runner.get_llm()
                    config = llm.llm_engine.vllm_config
                    assert config.use_v2_model_runner is use_v2
                    assert config.scheduler_config.async_scheduling is async_scheduling
                    assert config.cache_config.prefix_cache_retention_interval is None
                    if use_replayssm:
                        (block_size,) = runner.collective_rpc(_stress_mamba_block_size)
                        probe_history = _generate_with_preemption(llm, block_size)
                    assert probe_history is not None
                    before_probe = _metric_counts(llm)
                    (probe,) = llm.generate(
                        [{"prompt_token_ids": probe_history}],
                        SamplingParams(
                            temperature=0.0,
                            max_tokens=8,
                            ignore_eos=True,
                            logprobs=5,
                        ),
                    )
                    after_probe = _metric_counts(llm)
                    assert probe.finished and len(probe.outputs[0].token_ids) == 8
                    assert probe.prompt_token_ids == probe_history
                    if use_replayssm:
                        assert probe.num_cached_tokens is not None
                        assert probe.num_cached_tokens >= 3 * block_size
                        assert (
                            after_probe["vllm:prefix_cache_hits"]
                            > before_probe["vllm:prefix_cache_hits"]
                        )
                    else:
                        assert probe.num_cached_tokens == 0
                    completion = probe.outputs[0]
                    boundary_outputs[use_replayssm] = [
                        (completion.token_ids, completion.text, completion.logprobs)
                    ]
                    print(
                        f"ReplaySSM={use_replayssm}: boundary probe "
                        f"history_tokens={len(probe_history)}, "
                        f"cached_tokens={probe.num_cached_tokens}, "
                        f"token_ids={completion.token_ids}, "
                        f"logprobs={completion.logprobs}"
                    )

    finally:
        envs.disable_envs_cache()

    check_logprobs_close(
        outputs_0_lst=boundary_outputs[False],
        outputs_1_lst=boundary_outputs[True],
        name_0="baseline_fresh_accepted_history",
        name_1="replayssm_cached_accepted_history",
    )
