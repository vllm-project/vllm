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
MAMBA2_PREFIX_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
MODELS = [
    pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
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
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        if expected_v2 is not None:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is expected_v2
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


@pytest.mark.parametrize("model_name", MODELS)
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
                name_1="replayssm_flashinfer",
                expected_v2=use_v2_model_runner,
            )
    finally:
        # The context restores the environment before the final cache reset.
        envs.disable_envs_cache()


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
# All-mode MTP must cache at least two full state blocks: the drafter drops
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
    mamba_cache_mode: str,
    moe_backend: str | None = None,
    use_ngram: bool,
    use_v2: bool,
    tensor_parallel_size: int,
    mamba_backend: str = "flashinfer",
):
    def run() -> None:
        # ReplaySSM materializes the exact SSM state at each cacheable block
        # boundary, so cached prefixes must match the always-materialized baseline.
        common = dict(
            max_model_len=8192,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            mamba_cache_mode=mamba_cache_mode,
            mamba_backend=mamba_backend,
            disable_log_stats=False,  # required for llm.get_metrics()
            tensor_parallel_size=tensor_parallel_size,
        )
        if moe_backend is not None:
            common["moe_backend"] = moe_backend
        if use_ngram:
            common["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 3,
            }

        with vllm_runner(model_name, **common) as llm:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
            baseline_block_size = llm.llm.llm_engine.vllm_config.cache_config.block_size
            llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            baseline_hits_before = _prefix_cache_hits(llm)
            baseline = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            baseline_hits = _prefix_cache_hits(llm)

        with vllm_runner(
            model_name, use_replayssm=True, replayssm_buffer_len=16, **common
        ) as llm:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
            replay_block_size = llm.llm.llm_engine.vllm_config.cache_config.block_size
            if mamba_backend == "flashinfer":
                # FlashInfer rings are auxiliary and cannot affect the shared page.
                assert replay_block_size == baseline_block_size
            else:
                # Triton retains the original packed five-state page. Its rings may
                # increase the attention block size needed to match that page.
                assert replay_block_size >= baseline_block_size
            llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay_hits_before = _prefix_cache_hits(llm)
            replay = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay_hits = _prefix_cache_hits(llm)
            if use_ngram:
                assert (
                    sum(
                        metric.value
                        for metric in llm.llm.get_metrics()
                        if isinstance(metric, Counter)
                        and metric.name == "vllm:spec_decode_num_drafts"
                    )
                    > 0
                )

        assert baseline_hits > baseline_hits_before
        assert replay_hits > replay_hits_before, (
            f"ReplaySSM {mamba_cache_mode}-mode run produced no prefix-cache hits; "
            "the shared prefix may be shorter than one mamba block, so prefix "
            "caching is inert"
        )
        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=replay,
            name_0=f"{mamba_backend}_baseline_{mamba_cache_mode}_pc",
            name_1=f"{mamba_backend}_replayssm_{mamba_cache_mode}_pc",
        )

    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0")
            envs.disable_envs_cache()
            run()
    finally:
        envs.disable_envs_cache()


@requires_flashinfer_replayssm_materialization
@pytest.mark.parametrize("model_name", MODELS)
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
        mamba_cache_mode="align",
        use_ngram=use_ngram,
        use_v2=use_v2,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_triton_replayssm_align_prefix_cache_matches_baseline_v1(
    vllm_runner, model_name, monkeypatch: pytest.MonkeyPatch
):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        mamba_cache_mode="align",
        use_ngram=False,
        use_v2=False,
        tensor_parallel_size=1,
        mamba_backend="triton",
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_flashinfer_replayssm_all_prefix_cache(vllm_runner, monkeypatch, use_v2: bool):
    _check_replayssm_prefix_caching(
        vllm_runner,
        MAMBA2_PREFIX_MODEL,
        monkeypatch,
        mamba_cache_mode="all",
        moe_backend="triton",
        use_ngram=False,
        use_v2=use_v2,
        tensor_parallel_size=1,
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize(
    ("use_v2", "mode"),
    [(False, "align"), (True, "align"), (True, "all")],
    ids=["v1-align", "v2-align", "v2-all"],
)
def test_flashinfer_replayssm_prefix_cache_mtp(vllm_runner, monkeypatch, use_v2, mode):
    common = dict(
        max_model_len=12288,
        max_num_seqs=4,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode=mode,
        mamba_backend="flashinfer",
        dtype="bfloat16",
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
            envs.disable_envs_cache()
            # Generic V2 all mode remains unsupported. Align materializes the
            # same canonical boundary and is the supported numerical reference.
            reference_common = {**common, "mamba_cache_mode": "align"}
            with vllm_runner(MAMBA2_MTP_MODEL, **reference_common) as llm:
                llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                baseline_hits_before = _prefix_cache_hits(llm)
                baseline = llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                assert _prefix_cache_hits(llm) > baseline_hits_before
            with vllm_runner(
                MAMBA2_MTP_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
                first_pass = llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                first_pass_hits = _prefix_cache_hits(llm)
                cached = llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                cached_hits = _prefix_cache_hits(llm)
                draft_count = sum(
                    metric.value
                    for metric in llm.llm.get_metrics()
                    if isinstance(metric, Counter)
                    and metric.name == "vllm:spec_decode_num_drafts"
                )
    finally:
        envs.disable_envs_cache()

    assert cached_hits > first_pass_hits
    assert draft_count > 0
    print(
        f"ReplaySSM v{2 if use_v2 else 1} {mode}: "
        f"prefix_hit_delta={cached_hits - first_pass_hits}, drafts={draft_count}"
    )
    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=cached,
        name_0=f"baseline_{mode}_mtp_v{2 if use_v2 else 1}_cached",
        name_1=f"replayssm_{mode}_mtp_v{2 if use_v2 else 1}_cached",
    )
    check_logprobs_close(
        outputs_0_lst=first_pass,
        outputs_1_lst=cached,
        name_0=f"replayssm_{mode}_mtp_v{2 if use_v2 else 1}_first_pass",
        name_1=f"replayssm_{mode}_mtp_v{2 if use_v2 else 1}_cached",
    )
