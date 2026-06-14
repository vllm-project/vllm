# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("ray")
benchmark_moe = pytest.importorskip("benchmarks.kernels.benchmark_moe")


BASE_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


@pytest.mark.parametrize(
    "message",
    [
        "PassManager::run failed",
        "error: operand #0 does not dominate this use",
        "Pipeline failed while executing [`TritonGPURemoveLayoutConversions`]",
    ],
)
def test_is_skippable_triton_compile_error(message):
    assert benchmark_moe._is_skippable_triton_compile_error(RuntimeError(message))


def test_is_skippable_triton_compile_error_rejects_runtime_errors():
    assert not benchmark_moe._is_skippable_triton_compile_error(
        RuntimeError("CUDA out of memory")
    )


def _run_tune_search(monkeypatch, configs):
    monkeypatch.setattr(benchmark_moe, "tqdm", lambda search_space: search_space)
    monkeypatch.setattr(benchmark_moe, "clear_triton_cache", lambda: None)

    return benchmark_moe._tune_search_space(
        num_tokens=16,
        num_experts=2,
        shard_intermediate_size=128,
        hidden_size=64,
        topk=2,
        dtype=benchmark_moe.torch.float16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        search_space=configs,
        block_quant_shape=None,
        use_deep_gemm=False,
    )


def test_tune_search_space_skips_triton_compile_errors(monkeypatch):
    bad_config = BASE_CONFIG.copy()
    good_config = {**BASE_CONFIG, "BLOCK_SIZE_N": 128}

    def fake_benchmark_config(config, *args, **kwargs):
        if config == bad_config:
            raise RuntimeError("PassManager::run failed")
        return 1.0

    monkeypatch.setattr(benchmark_moe, "benchmark_config", fake_benchmark_config)

    assert _run_tune_search(monkeypatch, [bad_config, good_config]) == good_config


def test_tune_search_space_returns_none_when_all_configs_fail(monkeypatch):
    configs = [BASE_CONFIG.copy(), {**BASE_CONFIG, "BLOCK_SIZE_N": 128}]

    def fake_benchmark_config(config, *args, **kwargs):
        raise RuntimeError("PassManager::run failed")

    monkeypatch.setattr(benchmark_moe, "benchmark_config", fake_benchmark_config)

    assert _run_tune_search(monkeypatch, configs) is None


def test_tune_search_space_reraises_non_compile_runtime_errors(monkeypatch):
    configs = [BASE_CONFIG.copy()]

    def fake_benchmark_config(config, *args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(benchmark_moe, "benchmark_config", fake_benchmark_config)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _run_tune_search(monkeypatch, configs)
