# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the SM120 sparse-MLA warmup's per-bucket refinement."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.utils.flashinfer as fi_utils
from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as warmup

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr(warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        warmup.current_platform, "is_device_capability_family", lambda _: True
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4, 16, 64]),
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=True),
        parallel_config=SimpleNamespace(
            enable_elastic_ep=False, pipeline_parallel_size=1
        ),
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
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    monkeypatch.setattr(
        warmup, "resolve_flashinfer_autotune_file", lambda _: cache_path
    )
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_v2_root", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(AutoTuner=SimpleNamespace(get=lambda: Mock())),
    )
    return SimpleNamespace(
        vllm_config=config,
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        execute_model=Mock(),
        sample_tokens=Mock(),
    )


def _world(monkeypatch, *, rank, world_size):
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        broadcast_object=Mock(return_value=b"cached"),
        barrier=Mock(),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_node_count", lambda: 1)
    return world


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


def _record_runs(worker, is_v2, context_state):
    runs = []

    def dummy(**kwargs):
        assert kwargs["is_profile"] and kwargs["skip_eplb"]
        if kwargs.get("uniform_decode"):
            if is_v2:
                assert kwargs["skip_attn"] is False
            else:
                assert kwargs["force_attention"]
                assert kwargs["allow_microbatching"] is False
        runs.append((kwargs["num_tokens"], context_state["tuning"]))

    worker.model_runner._dummy_run.side_effect = dummy
    return runs


def _tracking_context(context_state):
    @contextmanager
    def context(*args, **kwargs):
        assert not context_state["tuning"]
        context_state["tuning"] = True
        try:
            yield
        finally:
            context_state["tuning"] = False

    return Mock(side_effect=context)


@pytest.mark.parametrize("is_v2", [False, True])
def test_managed_path_tunes_buckets_on_every_rank(worker, monkeypatch, is_v2):
    worker.vllm_config.use_v2_model_runner = is_v2
    worker.model_runner.max_num_reqs = 1  # no mixed batch: plain dummy run
    _world(monkeypatch, rank=0, world_size=1)
    state = {"tuning": False}
    managed = _tracking_context(state)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=managed, autotune_v2_reload=Mock()),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: True)
    runs = _record_runs(worker, is_v2, state)

    warmup.flashinfer_sparse_mla_decode_autotune_warmup(worker)

    assert runs == [(16, True), (1, True)]
    assert managed.call_count == 2


@pytest.mark.parametrize("rank", [0, 1])
def test_legacy_path_runs_buckets_after_mixed_warmup(worker, monkeypatch, rank):
    _world(monkeypatch, rank=rank, world_size=2)
    state = {"tuning": False}
    legacy = _tracking_context(state)
    monkeypatch.setattr(warmup, "flashinfer_autotune", legacy)
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: False)
    runs = _record_runs(worker, False, state)

    warmup.flashinfer_sparse_mla_decode_autotune_warmup(worker)

    leader = rank == 0
    assert runs == [(16, leader), (1, leader), (4, leader), (16, leader), (64, leader)]
    assert legacy.call_count == (2 if leader else 0)


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
