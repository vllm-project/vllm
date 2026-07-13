# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from benchmarks.kernels import benchmark_rapid_sampling as rapid_bench
from vllm.v1.sample.ops import topk_topp_sampler


def test_run_config_records_indexed_penalty_diagnostics(monkeypatch) -> None:
    config = rapid_bench.BenchmarkConfig(
        batch_size=3,
        vocab_size=8,
        top_p=0.9,
        top_k=4,
        logit_type=1,
    )
    real_full = torch.full
    real_zeros = torch.zeros
    logits = real_zeros((config.batch_size, config.vocab_size), dtype=torch.float32)
    penalty_indices = torch.arange(config.batch_size * 2, dtype=torch.int32)[::2]
    assert not penalty_indices.is_contiguous()

    monkeypatch.setattr(rapid_bench, "create_logits", lambda _config, seed: logits)
    monkeypatch.setattr(rapid_bench, "rapid_sample_input_supported", lambda _: True)
    monkeypatch.setattr(
        rapid_bench,
        "make_rapid_args",
        lambda _config: (
            real_full((config.batch_size,), config.top_k, dtype=torch.int32),
            real_full((config.batch_size,), config.top_p, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        rapid_bench,
        "make_non_contiguous_penalty_indices",
        lambda _config: (penalty_indices, config.batch_size * 2, "stride2-test"),
    )

    def cpu_full(*args, **kwargs):
        kwargs.pop("device", None)
        return real_full(*args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(rapid_bench.torch, "full", cpu_full)
    monkeypatch.setattr(rapid_bench.torch, "zeros", cpu_zeros)

    def fake_rapid_sample(
        logits,
        top_k,
        top_p,
        *,
        penalties=None,
        penalty_indices=None,
        **_kwargs,
    ):
        assert penalty_indices is penalty_indices_arg
        assert penalties.shape == (config.batch_size * 2, config.vocab_size)
        topk_topp_sampler._record_rapid_penalty_index_stats(
            rows=logits.shape[0],
            vocab_size=logits.shape[1],
        )
        return real_zeros((config.batch_size,), dtype=torch.int32)

    penalty_indices_arg = penalty_indices
    monkeypatch.setattr(rapid_bench, "rapid_sample", fake_rapid_sample)

    def fake_benchmark_cuda_call(fn, warmup_iters, benchmark_iters):
        assert warmup_iters == 0
        assert benchmark_iters == 1
        fn()
        return 0.25

    monkeypatch.setattr(rapid_bench, "benchmark_cuda_call", fake_benchmark_cuda_call)

    result = rapid_bench.run_config(
        config,
        providers=["rapid_penalty_indexed"],
        warmup_iters=0,
        benchmark_iters=1,
    )

    assert result["rapid_penalty_index_pattern"] == "stride2-test"
    assert result["rapid_penalty_index_rows"] == config.batch_size * 2
    assert result["rapid_penalty_index_first_values"] == [0, 2, 4]
    assert result["rapid_penalty_indexed_ms"] == 0.25
    assert result["rapid_penalty_index_stats"] == {
        "indexed_calls": 1,
        "indexed_rows": 3,
        "indexed_vocab_elements": 24,
    }


def test_run_config_rejects_legacy_indexed_penalty_provider() -> None:
    config = rapid_bench.BenchmarkConfig(
        batch_size=3,
        vocab_size=8,
        top_p=0.9,
        top_k=4,
        logit_type=1,
    )
    with pytest.raises(ValueError, match="rapid_penalty_indexed_legacy"):
        rapid_bench.run_config(
            config,
            providers=["rapid_penalty_indexed_legacy"],
            warmup_iters=0,
            benchmark_iters=1,
        )
