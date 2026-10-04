# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import threading
from collections import defaultdict
from contextlib import nullcontext
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from vllm.model_executor.warmup import kernel_warmup as warmup
from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
    flashinfer_autotune,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend),
            attention_config=SimpleNamespace(hisparse_config=None),
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ],
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 32, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(skip_attn):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)

    assert get_buckets.call_args_list == [call(8192), call(128)]
    assert autotune.call_args_list == [
        call(tuning_buckets=max_buckets),
        call(tuning_buckets=deferred_buckets),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


class _AutotuneGroup:
    def __init__(self, run, ranks):
        self.run = run
        self.ranks = tuple(ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = self.ranks.index(run.rank)
        self.cpu_group = self

    def record(self, operation):
        self.run.collectives[self.ranks][self.run.rank].append(operation)

    def broadcast_object(self, obj, src=0):
        assert src == 0
        self.record(("broadcast", src))
        if self.rank_in_group == src:
            self.run.broadcasts[self.ranks] = obj
        return self.run.broadcasts[self.ranks]

    def barrier(self):
        self.record(("barrier",))


class _AutotuneTuner:
    def __init__(self, run):
        self.run = run
        self.cache = {}
        self.loaded = None

    def load_configs(self, path):
        self.loaded = json.loads(Path(path).read_text())
        self.cache.update(self.loaded)

    def save_configs(self, path):
        self.run.saves.append((self.run.rank, Path(path), dict(self.cache)))
        Path(path).write_text(json.dumps(self.cache))

    def profile(self, operation):
        if operation in self.cache:
            return
        group = self.run.tuning_group
        self.run.profile_groups[self.run.rank].append(
            None if group is None else group.ranks
        )
        for tactic in range(2):
            if group is not None:
                group.record(("all_reduce", operation, tactic))
        self.cache[operation] = self.run.rank // self.run.tp


class _AutotuneRun:
    def __init__(self, pp, tp):
        self.pp, self.tp = pp, tp
        self.rank = 0
        self.tuning_group = None
        self.collectives: dict[tuple[int, ...], dict[int, list[tuple[Any, ...]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.broadcasts = {}
        self.tuners = {}
        self.saves = []
        self.profile_groups = defaultdict(list)

    def world(self):
        return _AutotuneGroup(self, range(self.pp * self.tp))

    def tensor_group(self):
        start = self.rank // self.tp * self.tp
        return _AutotuneGroup(self, range(start, start + self.tp))

    def pipeline_group(self):
        return SimpleNamespace(world_size=self.pp)

    def set_group(self, group):
        self.tuning_group = group

    def dummy_runs(self, runner, *, skip_attn=False):
        self.tuners[self.rank].profile("shared_gemm")
        if self.pp > 1 and self.rank // self.tp == 0:
            self.tuners[self.rank].profile("pp0_extra_gemm")

    def execute(self):
        for rank in range(self.pp * self.tp):
            self.rank = rank
            self.tuners[rank] = _AutotuneTuner(self)
            flashinfer_autotune(_make_runner([]))
        return self

    def assert_collectives_match(self):
        for ranks, traces in self.collectives.items():
            assert set(traces) == set(ranks)
            expected = traces[ranks[0]]
            assert all(trace == expected for trace in traces.values()), dict(traces)


@pytest.fixture
def autotune_run(monkeypatch, tmp_path):
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    def make_run(*, pp=2, tp=4):
        run = _AutotuneRun(pp, tp)
        autotuner = ModuleType("flashinfer.autotuner")
        monkeypatch.setattr(
            autotuner,
            "AutoTuner",
            SimpleNamespace(get=lambda: run.tuners[run.rank]),
            raising=False,
        )
        monkeypatch.setattr(
            autotuner, "set_autotune_process_group", run.set_group, raising=False
        )
        monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        monkeypatch.setattr(parallel_state, "get_world_group", run.world)
        monkeypatch.setattr(parallel_state, "get_tp_group", run.tensor_group)
        monkeypatch.setattr(parallel_state, "get_pp_group", run.pipeline_group)
        monkeypatch.setattr(fi_utils, "autotune", lambda **kwargs: nullcontext())
        monkeypatch.setattr(
            warmup,
            "resolve_flashinfer_autotune_file",
            lambda runner: tmp_path / "autotune_configs.json",
        )
        monkeypatch.setattr(
            warmup, "_flashinfer_autotune_skip_ops", lambda runner: None
        )
        monkeypatch.setattr(
            warmup, "_run_flashinfer_autotune_dummy_runs", run.dummy_runs
        )
        monkeypatch.setattr(warmup, "replayssm_autotune_warmup", lambda runner: None)
        monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", lambda model: None)
        return run

    return make_run


@pytest.mark.parametrize("tp", [1, 4])
def test_heterogeneous_pp_stages_have_compatible_collectives(autotune_run, tp):
    run = autotune_run(tp=tp).execute()
    run.assert_collectives_match()
    for rank, groups in run.profile_groups.items():
        expected = tuple(range(rank // tp * tp, (rank // tp + 1) * tp))
        assert groups and all(
            group == (expected if tp > 1 else None) for group in groups
        )


def test_pp_stage_cache_roundtrip_isolated_and_asymmetric_hits_safe(
    autotune_run, tmp_path
):
    legacy = tmp_path / "autotune_configs.json"
    legacy.write_text('{"legacy_world_cache": 99}')
    cold = autotune_run().execute()
    assert [rank for rank, _, _ in cold.saves] == [0, 4]
    paths = [path for _, path, _ in cold.saves]
    assert len(set(paths)) == 2 and legacy not in paths
    assert json.loads(legacy.read_text()) == {"legacy_world_cache": 99}
    assert cold.saves[0][2] == {"shared_gemm": 0, "pp0_extra_gemm": 0}
    assert cold.saves[1][2] == {"shared_gemm": 1}
    cold.assert_collectives_match()
    warm = autotune_run().execute()
    warm.assert_collectives_match()
    assert not warm.profile_groups
    for rank, tuner in warm.tuners.items():
        assert tuner.loaded == cold.saves[rank // 4][2]
    paths[1].unlink()
    mixed = autotune_run().execute()
    mixed.assert_collectives_match()
    assert set(mixed.profile_groups) == {4, 5, 6, 7}
    assert all(mixed.tuners[rank].loaded is not None for rank in range(4))
    assert all(mixed.tuners[rank].loaded is None for rank in range(4, 8))


def test_pp1_retains_world_synchronization_and_existing_cache_name(autotune_run):
    run = autotune_run(pp=1, tp=4).execute()
    run.assert_collectives_match()
    assert [(rank, path.name) for rank, path, _ in run.saves] == [
        (0, "autotune_configs.json")
    ]
    assert all(groups == [(0, 1, 2, 3)] for groups in run.profile_groups.values())


class _CacheOnlyGroup:
    """Per-rank view of a group whose collectives rendezvous across threads."""

    def __init__(self, run, ranks):
        self.run = run
        self.ranks = tuple(ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = self.ranks.index(run.rank)
        self.cpu_group = self

    def _exchange(self, operation, value):
        self.run.collectives[self.ranks][self.run.rank].append(operation)
        if self.world_size == 1:
            return [value]
        slots, barrier = self.run.rendezvous(self.ranks)
        slots[self.rank_in_group] = value
        barrier.wait()
        values = list(slots)
        barrier.wait()
        return values

    def broadcast_object(self, obj, src=0):
        return self._exchange("broadcast", obj)[src]

    def all_gather_object(self, output, obj):
        output[:] = self._exchange("all_gather", obj)


class _CacheOnlyTuner:
    def __init__(self, result):
        self.result = result
        self.loaded = None

    def load_configs(self, path):
        self.loaded = Path(path).read_bytes()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _CacheOnlyRun:
    def __init__(self, pp, tp, load_results):
        self.pp, self.tp = pp, tp
        self.load_results = load_results
        self.local = threading.local()
        self.lock = threading.Lock()
        self.rendezvous_points = {}
        self.collectives: dict[tuple[int, ...], dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.tuners = {}
        self.errors = {}
        self.autotuned = []

    @property
    def rank(self):
        return self.local.rank

    def rendezvous(self, ranks):
        with self.lock:
            if ranks not in self.rendezvous_points:
                self.rendezvous_points[ranks] = (
                    [None] * len(ranks),
                    threading.Barrier(len(ranks), timeout=5),
                )
            return self.rendezvous_points[ranks]

    def world(self):
        return _CacheOnlyGroup(self, range(self.pp * self.tp))

    def tensor_group(self):
        start = self.rank // self.tp * self.tp
        return _CacheOnlyGroup(self, range(start, start + self.tp))

    def _run_rank(self, rank):
        self.local.rank = rank
        self.tuners[rank] = _CacheOnlyTuner(self.load_results.get(rank, True))
        try:
            flashinfer_autotune(_make_runner([]))
        except Exception as exc:
            self.errors[rank] = exc

    def execute(self):
        threads = [
            threading.Thread(target=self._run_rank, args=(rank,))
            for rank in range(self.pp * self.tp)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in threads), "ranks hung"
        assert not any(
            isinstance(error, threading.BrokenBarrierError)
            for error in self.errors.values()
        ), "ranks issued mismatched collectives"
        for ranks, traces in self.collectives.items():
            assert set(traces) == set(ranks)
            assert all(trace == traces[ranks[0]] for trace in traces.values())
        return self

    def assert_all_failed(self, *fragments):
        assert set(self.errors) == set(range(self.pp * self.tp))
        messages = {str(error) for error in self.errors.values()}
        assert len(messages) == 1, messages
        message = messages.pop()
        assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY" in message
        for fragment in fragments:
            assert fragment in message


@pytest.fixture
def cache_only_run(monkeypatch, tmp_path):
    import torch

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY", "1")

    def make_run(*, pp=1, tp=2, load_results=None):
        run = _CacheOnlyRun(pp, tp, load_results or {})
        autotuner = ModuleType("flashinfer.autotuner")
        monkeypatch.setattr(
            autotuner,
            "AutoTuner",
            SimpleNamespace(get=lambda: run.tuners[run.rank]),
            raising=False,
        )
        monkeypatch.setattr(
            autotuner,
            "set_autotune_process_group",
            lambda group: run.autotuned.append(run.rank),
            raising=False,
        )
        monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda output, obj, group: group.all_gather_object(output, obj),
        )
        monkeypatch.setattr(parallel_state, "get_world_group", run.world)
        monkeypatch.setattr(parallel_state, "get_tp_group", run.tensor_group)
        monkeypatch.setattr(
            parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=pp)
        )
        monkeypatch.setattr(fi_utils, "autotune", lambda **kwargs: nullcontext())
        monkeypatch.setattr(
            warmup,
            "resolve_flashinfer_autotune_file",
            lambda runner: tmp_path / "autotune_configs.json",
        )
        monkeypatch.setattr(
            warmup, "_flashinfer_autotune_skip_ops", lambda runner: None
        )
        monkeypatch.setattr(
            warmup,
            "_run_flashinfer_autotune_dummy_runs",
            lambda runner, **kwargs: run.autotuned.append(run.rank),
        )
        return run

    return make_run


@pytest.mark.parametrize("tp", [1, 2])
def test_cache_only_loads_cache_on_every_rank_and_skips_autotune(
    cache_only_run, tmp_path, tp
):
    cache = b'{"gemm": ["Runner", 3]}'
    (tmp_path / "autotune_configs.json").write_bytes(cache)
    run = cache_only_run(tp=tp).execute()
    assert not run.errors
    assert all(tuner.loaded == cache for tuner in run.tuners.values())
    assert not run.autotuned
    assert list(tmp_path.iterdir()) == [tmp_path / "autotune_configs.json"]


@pytest.mark.parametrize(
    ("cache", "fragment"),
    [
        (None, "FileNotFoundError"),
        (b"", "is empty"),
        ("directory", "IsADirectoryError"),
    ],
)
def test_cache_only_read_failure_fails_every_rank_without_loading(
    cache_only_run, tmp_path, cache, fragment
):
    path = tmp_path / "autotune_configs.json"
    if cache == "directory":
        path.mkdir()
    elif cache is not None:
        path.write_bytes(cache)
    run = cache_only_run(tp=4).execute()
    run.assert_all_failed("rank 0:", fragment)
    assert all(tuner.loaded is None for tuner in run.tuners.values())
    assert not run.autotuned


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        (False, "FlashInfer rejected the cache"),
        (ValueError("bad tactic"), "ValueError: bad tactic"),
    ],
)
@pytest.mark.parametrize("failing_rank", [0, 3])
def test_cache_only_load_failure_on_any_rank_fails_every_rank(
    cache_only_run, tmp_path, result, fragment, failing_rank
):
    (tmp_path / "autotune_configs.json").write_bytes(b"{}")
    run = cache_only_run(tp=4, load_results={failing_rank: result}).execute()
    run.assert_all_failed(f"rank {failing_rank}: {fragment}")
    assert not run.autotuned


def test_cache_only_pp_stages_load_own_cache_and_fail_together(
    cache_only_run, tmp_path
):
    stage_caches = {0: b'{"stage": 0}', 1: b'{"stage": 1}'}
    for stage, cache in stage_caches.items():
        ranks = f"{2 * stage}-{2 * stage + 1}"
        (tmp_path / f"autotune_configs_tp_{ranks}.json").write_bytes(cache)
    run = cache_only_run(pp=2, tp=2).execute()
    assert not run.errors
    for rank, tuner in run.tuners.items():
        assert tuner.loaded == stage_caches[rank // 2]
    assert set(run.collectives) == {(0, 1), (2, 3), (0, 1, 2, 3)}
    assert run.collectives[(0, 1, 2, 3)][0] == ["all_gather"]

    (tmp_path / "autotune_configs_tp_2-3.json").unlink()
    run = cache_only_run(pp=2, tp=2).execute()
    run.assert_all_failed("rank 2:", "autotune_configs_tp_2-3.json")
    assert "rank 0:" not in str(run.errors[0])


def test_cache_only_disabled_keeps_autotune_path(autotune_run, monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY", "0")
    run = autotune_run(pp=1, tp=2).execute()
    run.assert_collectives_match()
    assert set(run.profile_groups) == {0, 1}
    assert [rank for rank, _, _ in run.saves] == [0]
