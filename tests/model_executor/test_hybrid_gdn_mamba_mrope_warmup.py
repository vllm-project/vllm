# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.warmup import kernel_warmup
from vllm.v1.worker.gpu.warmup import warmup_kernels


def test_kernel_warmup_runs_hybrid_and_zeroer_warmups(monkeypatch) -> None:
    calls: list[str] = []

    model = object()

    def fake_hybrid_warmup(*args, **kwargs) -> None:
        assert args == (model,)
        assert kwargs == {"model_dtype": torch.bfloat16}
        calls.append("hybrid")

    class FakeZeroer:
        def warmup(self) -> None:
            calls.append("zeroer")

    def fake_dummy_run(**kwargs) -> None:
        assert kwargs == {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
        calls.append("dummy")

    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)
    monkeypatch.setattr(
        kernel_warmup,
        "has_hybrid_gdn_mamba_mrope",
        lambda model: True,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "hybrid_gdn_mamba_mrope_warmup",
        fake_hybrid_warmup,
    )

    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
        model_runner=SimpleNamespace(
            dtype=torch.bfloat16,
            _kv_block_zeroer=FakeZeroer(),
            _dummy_run=fake_dummy_run,
            is_pooling_model=True,
            attn_groups=[],
        ),
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == ["hybrid", "zeroer"]


def test_kernel_warmup_runs_runtime_dummy_for_hybrid_models(monkeypatch) -> None:
    calls: list[str] = []

    model = object()

    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)
    monkeypatch.setattr(
        kernel_warmup,
        "has_hybrid_gdn_mamba_mrope",
        lambda model: True,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "hybrid_gdn_mamba_mrope_warmup",
        lambda *args, **kwargs: calls.append("hybrid"),
    )

    def fake_scheduler_warmup(model_runner, execute_model, sample_tokens) -> None:
        assert model_runner.num_speculative_steps == 0
        assert hasattr(model_runner, "kv_connector")
        assert model_runner.is_last_pp_rank is True
        calls.append("scheduler")

    monkeypatch.setattr(kernel_warmup, "warmup_kernels", fake_scheduler_warmup)
    monkeypatch.setattr(
        kernel_warmup,
        "_warmup_single_request_decode_kernels",
        lambda worker: calls.append("single"),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    def fake_dummy_run(**kwargs) -> None:
        assert kwargs == {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
        calls.append("dummy")

    model_runner = SimpleNamespace(
        dtype=torch.bfloat16,
        _kv_block_zeroer=None,
        _dummy_run=fake_dummy_run,
        is_pooling_model=False,
        attn_groups=[],
    )
    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
        model_runner=model_runner,
        execute_model=object(),
        sample_tokens=object(),
        use_v2_model_runner=False,
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == ["hybrid", "dummy", "scheduler", "single"]
    assert not hasattr(model_runner, "num_speculative_steps")
    assert not hasattr(model_runner, "kv_connector")
    assert not hasattr(model_runner, "is_last_pp_rank")


def test_kernel_warmup_skips_scheduler_warmup_for_v2_runner(monkeypatch) -> None:
    calls: list[str] = []

    model = object()

    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)
    monkeypatch.setattr(
        kernel_warmup,
        "has_hybrid_gdn_mamba_mrope",
        lambda model: True,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "hybrid_gdn_mamba_mrope_warmup",
        lambda *args, **kwargs: calls.append("hybrid"),
    )
    monkeypatch.setattr(
        kernel_warmup,
        "warmup_kernels",
        lambda *args, **kwargs: calls.append("scheduler"),
    )
    monkeypatch.setattr(
        kernel_warmup,
        "_warmup_single_request_decode_kernels",
        lambda worker: calls.append("single"),
    )

    def fake_dummy_run(**kwargs) -> None:
        calls.append("dummy")

    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
        model_runner=SimpleNamespace(
            dtype=torch.bfloat16,
            _kv_block_zeroer=None,
            _dummy_run=fake_dummy_run,
            is_pooling_model=False,
            attn_groups=[],
        ),
        execute_model=object(),
        sample_tokens=object(),
        use_v2_model_runner=True,
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == ["hybrid", "dummy"]


def test_single_request_warmup_builds_prefill_decode_cleanup() -> None:
    outputs = []
    samples = []

    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        model_runner=SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                num_blocks=16,
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
                    SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
                ],
            ),
        ),
        execute_model=lambda scheduler_output: outputs.append(scheduler_output),
        sample_tokens=lambda grammar_output: samples.append(grammar_output),
    )

    kernel_warmup._warmup_single_request_decode_kernels(worker)

    assert len(outputs) == 3
    assert samples == [None, None]

    prefill_output, decode_output, cleanup_output = outputs
    assert len(prefill_output.scheduled_new_reqs) == 1
    assert prefill_output.total_num_scheduled_tokens == 64
    assert prefill_output.num_common_prefix_blocks == [0, 0]
    assert prefill_output.scheduled_new_reqs[0].block_ids == (
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    )

    cached_reqs = decode_output.scheduled_cached_reqs
    assert cached_reqs.req_ids == ["_hybrid_single_request_warmup_"]
    assert cached_reqs.num_computed_tokens == [64]
    assert cached_reqs.num_output_tokens == [1]
    assert cached_reqs.new_block_ids == [([9], [10])]
    assert decode_output.total_num_scheduled_tokens == 1
    assert cleanup_output.finished_req_ids == {"_hybrid_single_request_warmup_"}


def test_single_request_warmup_disables_kv_connector_on_failure() -> None:
    connector_calls = []

    class FakeKVConnector:
        def set_disabled(self, disabled: bool) -> None:
            connector_calls.append(disabled)

    model_runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=16,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            ],
        ),
        kv_connector=FakeKVConnector(),
    )
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        model_runner=model_runner,
        execute_model=lambda _scheduler_output: (_ for _ in ()).throw(
            RuntimeError("single warmup failed")
        ),
        sample_tokens=lambda _grammar_output: None,
    )

    with pytest.raises(RuntimeError, match="single warmup failed"):
        kernel_warmup._warmup_single_request_decode_kernels(worker)

    assert connector_calls == [True, False]


def test_scheduler_warmup_reenables_kv_connector_on_failure() -> None:
    connector_calls = []

    class FakeKVConnector:
        def set_disabled(self, disabled: bool) -> None:
            connector_calls.append(disabled)

    model_runner = SimpleNamespace(
        num_speculative_steps=0,
        kv_cache_config=SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            ],
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=16,
        ),
        kv_connector=FakeKVConnector(),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 128),
    )

    def fail_execute_model(_scheduler_output):
        raise RuntimeError("warmup failed")

    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup_kernels(model_runner, fail_execute_model, lambda _grammar: None)

    assert connector_calls == [True, False]
