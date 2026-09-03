# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

from inspect import signature

import pytest

import vllm.envs as envs
from vllm.v1.metrics.reader import Counter

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

try:
    from flashinfer.mamba.checkpointing_ssu import (
        CheckpointingSSURunner,
        allocate_checkpointing_ssu_scratch,
    )

    HAS_FLASHINFER_CHECKPOINTING_SSU = callable(CheckpointingSSURunner) and callable(
        allocate_checkpointing_ssu_scratch
    )
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False

try:
    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = callable(replayssm_materialize) and (
        "active_request_indices" in signature(replayssm_materialize).parameters
    )
except ImportError:
    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = False

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


def _check_replayssm_parity(
    vllm_runner,
    model_name,
    *,
    tensor_parallel_size=1,
    mamba_backend: str = "triton",
    name_1: str = "replayssm",
    require_v2: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
):
    def run() -> None:
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
            baseline = llm.generate_greedy_logprobs(
                PROMPTS, max_tokens=32, num_logprobs=5
            )
        with vllm_runner(
            model_name, use_replayssm=True, replayssm_buffer_len=16, **common
        ) as llm:
            if require_v2:
                assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
            replay = llm.generate_greedy_logprobs(
                PROMPTS, max_tokens=32, num_logprobs=5
            )

        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=replay,
            name_0="baseline",
            name_1=name_1,
        )

    if not require_v2:
        run()
        return

    assert monkeypatch is not None
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            run()
    finally:
        envs.disable_envs_cache()


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
def test_replayssm_flashinfer_decode_matches_baseline(vllm_runner, model_name):
    pytest.importorskip("flashinfer.mamba.checkpointing_ssu")
    _check_replayssm_parity(
        vllm_runner,
        model_name,
        mamba_backend="flashinfer",
        name_1="replayssm_flashinfer",
    )


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_flashinfer_decode_matches_baseline_v2(
    vllm_runner, model_name, monkeypatch
):
    _check_replayssm_parity(
        vllm_runner,
        model_name,
        mamba_backend="flashinfer",
        name_1="replayssm_flashinfer_v2",
        require_v2=True,
        monkeypatch=monkeypatch,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_replayssm_flashinfer_decode_matches_baseline_tp2(vllm_runner, model_name):
    _check_replayssm_parity(
        vllm_runner,
        model_name,
        tensor_parallel_size=2,
        mamba_backend="flashinfer",
        name_1="replayssm_flashinfer_tp2",
    )


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
@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_flashinfer_matches_triton_replayssm(vllm_runner, model_name):
    # Both backends implement ReplaySSM; compare them directly on V1 because
    # Triton ReplaySSM is not supported on Model Runner V2.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        use_replayssm=True,
        replayssm_buffer_len=16,
    )
    with vllm_runner(model_name, mamba_backend="triton", **common) as llm:
        triton = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(model_name, mamba_backend="flashinfer", **common) as llm:
        flashinfer = llm.generate_greedy_logprobs(
            PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=triton,
        outputs_1_lst=flashinfer,
        name_0="replayssm_triton",
        name_1="replayssm_flashinfer",
    )


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
@large_gpu_mark(min_gb=40)
def test_replayssm_flashinfer_mtp_v2(vllm_runner, monkeypatch):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            with vllm_runner(MAMBA2_MTP_MODEL, **common) as llm:
                assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
                baseline = llm.generate_greedy_logprobs(
                    PROMPTS, max_tokens=32, num_logprobs=5
                )
            with vllm_runner(
                MAMBA2_MTP_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
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


def _check_flashinfer_replayssm_prefix_caching(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mamba_cache_mode: str,
    moe_backend: str | None = None,
    use_ngram: bool,
    use_v2: bool,
    tensor_parallel_size: int,
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
            mamba_backend="flashinfer",
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
            baseline = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            baseline_hits = _prefix_cache_hits(llm)

        with vllm_runner(
            model_name, use_replayssm=True, replayssm_buffer_len=16, **common
        ) as llm:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
            replay_block_size = llm.llm.llm_engine.vllm_config.cache_config.block_size
            assert replay_block_size == baseline_block_size
            llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay_hits = _prefix_cache_hits(llm)

        assert baseline_hits > 0
        assert replay_hits > 0, (
            f"ReplaySSM {mamba_cache_mode}-mode run produced no prefix-cache hits; "
            "the shared prefix may be shorter than one mamba block, so prefix "
            "caching is inert"
        )
        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=replay,
            name_0=f"flashinfer_baseline_{mamba_cache_mode}_pc",
            name_1=f"flashinfer_replayssm_{mamba_cache_mode}_pc",
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
    ("mamba_cache_mode", "use_v2", "use_ngram"),
    [
        pytest.param("align", False, False, id="align-v1-stp"),
        pytest.param("align", False, True, id="align-v1-ngram-t4"),
        pytest.param("align", True, False, id="align-v2-stp"),
    ],
)
def test_flashinfer_replayssm_prefix_cache_tp1(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    mamba_cache_mode: str,
    use_v2: bool,
    use_ngram: bool,
):
    _check_flashinfer_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        mamba_cache_mode=mamba_cache_mode,
        use_ngram=use_ngram,
        use_v2=use_v2,
        tensor_parallel_size=1,
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_flashinfer_replayssm_all_prefix_cache(vllm_runner, monkeypatch, use_v2: bool):
    _check_flashinfer_replayssm_prefix_caching(
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
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_flashinfer_replayssm_prefix_cache_v2_tp2(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
):
    _check_flashinfer_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        mamba_cache_mode="align",
        use_ngram=False,
        use_v2=True,
        tensor_parallel_size=2,
    )
