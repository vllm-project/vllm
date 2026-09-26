# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for sparse-only tuning and executable warmup buckets."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as warmup

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        warmup.current_platform, "is_device_capability_family", lambda _: True
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4, 16, 64]),
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=True),
        use_v2_model_runner=False,
    )
    backend = SimpleNamespace(get_name=lambda: "FLASHINFER_MLA_SPARSE_SM120")
    runner = SimpleNamespace(
        vllm_config=config,
        max_num_reqs=64,
        uniform_decode_query_len=1,
        decode_query_len=1,
        ubatch_runner=None,
        is_pooling_model=False,
        attn_groups=[[SimpleNamespace(backend=backend)]],
        _dummy_run=Mock(),
    )
    return SimpleNamespace(
        vllm_config=config,
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        execute_model=Mock(),
        sample_tokens=Mock(),
    )


@pytest.mark.parametrize(
    ("is_v2", "query_len", "max_reqs", "capture_sizes", "expected"),
    [
        (False, 1, 64, [128, 64, 1, 32, 32, 0, 65, -4], (1, 32, 64)),
        (True, 1, 8, [1, 8, 16, 64], (1, 8)),
        (False, 1, 8, [1, 8, 16, 64], (1, 8)),
        (False, 3, 2, [1, 3, 4, 6, 7], (1, 3, 4, 6)),
        (True, 3, 2, [1, 3, 4, 6, 7], (3, 6)),
        (True, 3, 64, [63, 64, 66], (63,)),
        (False, 1, 64, [], ()),
        (True, 1, 64, None, ()),
    ],
)
def test_sparse_mla_refine_tokens(
    worker, is_v2, query_len, max_reqs, capture_sizes, expected
):
    worker.vllm_config.use_v2_model_runner = is_v2
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = capture_sizes
    worker.model_runner.max_num_reqs = max_reqs
    worker.model_runner.uniform_decode_query_len = query_len
    worker.model_runner.decode_query_len = query_len
    assert warmup._sparse_mla_refine_tokens(worker) == expected


def test_refine_tokens_respect_token_budget(worker):
    worker.scheduler_config.max_num_batched_tokens = 8
    assert warmup._sparse_mla_refine_tokens(worker) == (1, 4)


@pytest.mark.parametrize("is_v2", [False, True])
def test_warmup_scopes_mixed_and_decode_calls(worker, monkeypatch, is_v2):
    worker.vllm_config.use_v2_model_runner = is_v2
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", " fp4_gemm, ,test_op ")
    active = False
    events = []

    @contextmanager
    def scoped(*, skip_ops):
        nonlocal active
        assert skip_ops == {"fp4_gemm", "test_op"}
        active = True
        try:
            yield
        finally:
            active = False

    def record(tokens):
        assert active
        events.append(tokens)
        return True

    def dummy(**kwargs):
        assert kwargs["is_profile"] and kwargs["skip_eplb"]
        assert kwargs.get("uniform_decode", False) == bool(events)
        if not events:
            assert kwargs["create_mixed_batch"]
        if is_v2:
            assert kwargs["skip_attn"] is False
        else:
            assert kwargs["force_attention"] and not kwargs["allow_microbatching"]
        record(kwargs["num_tokens"])

    worker.model_runner._dummy_run.side_effect = dummy
    monkeypatch.setattr(warmup, "autotune_sparse_mla_only", scoped)
    monkeypatch.setattr(
        warmup,
        "run_mixed_prefill_decode_warmup",
        lambda runner, execute, sample, tokens, **kwargs: record(tokens),
    )
    warmup.flashinfer_sparse_mla_decode_autotune_warmup(worker)
    assert events == [16, 1, 4, 16, 64]
    assert not active


@pytest.mark.parametrize(
    "gate", ["backend", "disabled", "missing", "platform", "v2_ubatch"]
)
def test_gates_precede_bucket_selection(worker, monkeypatch, gate):
    runner = worker.model_runner
    del runner.uniform_decode_query_len, runner.decode_query_len, runner.max_num_reqs
    monkeypatch.setattr(warmup, "has_flashinfer", lambda: gate != "missing")
    monkeypatch.setattr(
        warmup.current_platform,
        "is_device_capability_family",
        lambda _: gate != "platform",
    )
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = gate != "disabled"
    if gate == "backend":
        runner.attn_groups = []
    if gate == "v2_ubatch":
        worker.vllm_config.use_v2_model_runner = True
        runner.ubatch_runner = object()
    warmup.flashinfer_sparse_mla_decode_autotune_warmup(worker)
    runner._dummy_run.assert_not_called()
