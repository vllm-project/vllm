# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
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
            load_config=SimpleNamespace(load_format="auto"),
            kernel_config=SimpleNamespace(linear_backend=linear_backend),
            attention_config=SimpleNamespace(hisparse_config=None),
            parallel_config=SimpleNamespace(data_parallel_rank=0),
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

    def all_gather_object(self, out, obj):
        # Ranks run sequentially: snapshot every rank's file before any saves.
        self.record(("all_gather",))
        if isinstance(obj, bool):
            out[:] = [self.run.loads(rank) for rank in self.ranks]
            assert out[self.rank_in_group] == obj
            return
        out[:] = self.run.gathered.setdefault(
            self.ranks,
            [
                warmup._autotune_cache_fingerprint(self.run.cache_path(rank))
                for rank in self.ranks
            ],
        )

    def barrier(self):
        self.record(("barrier",))


class _AutotuneTuner:
    def __init__(self, run):
        self.run = run
        self.cache = {}
        self.loaded = None
        self._dirty = False

    def load_configs(self, path):
        if self.run.rank in self.run.rejected_load_ranks:
            return False
        self.loaded = json.loads(Path(path).read_text())
        self.cache.update(self.loaded)
        return True

    def clear_cache(self):
        self.cache.clear()
        self._dirty = False

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
        self._dirty = True


class _AutotuneRun:
    def __init__(self, pp, tp, cache_dir):
        self.pp, self.tp = pp, tp
        self.cache_dir = cache_dir
        self.rank = 0
        self.tuning_group = None
        self.collectives: dict[tuple[int, ...], dict[int, list[tuple[Any, ...]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.gathered = {}
        self.rejected_load_ranks: set[int] = set()
        self.tuners = {}
        self.saves = []
        self.profile_groups = defaultdict(list)

    def cache_path(self, rank):
        return self.cache_dir / f"autotune_configs_dp0_rank{rank}.json"

    def loads(self, rank):
        """Whether ``rank``'s tuner will accept the configs it is offered."""
        return rank not in self.rejected_load_ranks

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
        run = _AutotuneRun(pp, tp, tmp_path)
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
            "torch.distributed.all_gather_object",
            lambda out, obj, group: group.all_gather_object(out, obj),
        )
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
    cold.assert_collectives_match()
    assert [rank for rank, _, _ in cold.saves] == list(range(8))
    assert [path for _, path, _ in cold.saves] == [cold.cache_path(r) for r in range(8)]
    assert json.loads(legacy.read_text()) == {"legacy_world_cache": 99}
    stage_caches = [{"shared_gemm": 0, "pp0_extra_gemm": 0}, {"shared_gemm": 1}]
    assert all(cache == stage_caches[r // 4] for r, _, cache in cold.saves)
    warm = autotune_run().execute()
    warm.assert_collectives_match()
    assert not warm.profile_groups and not warm.saves
    for rank, tuner in warm.tuners.items():
        assert tuner.loaded == stage_caches[rank // 4]
    cold.cache_path(5).unlink()
    mixed = autotune_run().execute()
    mixed.assert_collectives_match()
    assert set(mixed.profile_groups) == {4, 5, 6, 7}
    assert all(mixed.tuners[rank].loaded is not None for rank in range(4))
    assert all(mixed.tuners[rank].loaded is None for rank in range(4, 8))


def test_pp1_tunes_world_group_and_saves_per_rank(autotune_run):
    run = autotune_run(pp=1, tp=4).execute()
    run.assert_collectives_match()
    assert [(rank, path) for rank, path, _ in run.saves] == [
        (rank, run.cache_path(rank)) for rank in range(4)
    ]
    assert all(groups == [(0, 1, 2, 3)] for groups in run.profile_groups.values())


def test_mismatched_rank_caches_are_ignored_by_every_rank(autotune_run):
    """One rank loading while others tune deadlocks the per-tactic reduce."""
    cold = autotune_run(pp=1, tp=4).execute()
    stale = cold.cache_path(2)
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "old_gemm": 0}))
    rerun = autotune_run(pp=1, tp=4).execute()
    rerun.assert_collectives_match()
    assert all(tuner.loaded is None for tuner in rerun.tuners.values())
    assert set(rerun.profile_groups) == {0, 1, 2, 3}


@pytest.mark.parametrize("pp, tp", [(1, 1), (1, 4), (2, 4)])
def test_rejected_cache_retunes_every_rank_in_its_tuning_group(autotune_run, pp, tp):
    """Matching files can still be incompatible with one rank's runtime."""
    autotune_run(pp=pp, tp=tp).execute()
    rerun = autotune_run(pp=pp, tp=tp)
    rerun.rejected_load_ranks = {tp - 1}
    rerun.execute()
    rerun.assert_collectives_match()
    assert set(rerun.profile_groups) == set(range(tp))
    assert {rank for rank, _, _ in rerun.saves} == set(range(tp))


def _serve_from_daemon(run, monkeypatch, daemon):
    """Back flashinfer_autotune's daemon cache with ``daemon``, keyed by rank."""
    held = set(daemon)

    def put(key, table):
        daemon[key] = table
        return True

    def load(table):
        run.tuners[run.rank].cache.update(json.loads(table))
        return True

    monkeypatch.setattr(
        warmup.DaemonArtifactCache,
        "from_vllm_config",
        lambda config: SimpleNamespace(get=daemon.get, put=put),
    )
    monkeypatch.setattr(
        warmup,
        "flashinfer_autotune_artifact_key",
        lambda runner, skip_ops, rank_id: rank_id,
    )
    monkeypatch.setattr(warmup, "try_load_flashinfer_autotune_configs", load)
    monkeypatch.setattr(
        warmup,
        "dump_flashinfer_autotune_configs",
        lambda: json.dumps(run.tuners[run.rank].cache).encode(),
    )
    # Ranks run one after another, so the gather predicts every rank's load
    # from the tables the daemons held before any rank published.
    monkeypatch.setattr(run, "loads", lambda rank: f"dp0_rank{rank}" in held)


def test_pp_stages_adopt_daemon_tables_independently(autotune_run, monkeypatch):
    """A stage adopting its tables from the weight cache daemons while another
    stage tunes must still meet it at the world barrier, and each rank must
    only ever see the table tuned for it."""
    run = autotune_run()
    daemon = {
        f"dp0_rank{rank}": json.dumps({"shared_gemm": 1, "rank": rank}).encode()
        for rank in range(4, 8)
    }
    _serve_from_daemon(run, monkeypatch, daemon)

    run.execute()
    run.assert_collectives_match()
    assert set(run.profile_groups) == {0, 1, 2, 3}
    for rank in range(4, 8):
        assert run.tuners[rank].cache == {"shared_gemm": 1, "rank": rank}
    assert json.loads(daemon["dp0_rank0"]) == {"shared_gemm": 0, "pp0_extra_gemm": 0}


def test_daemon_tables_are_adopted_only_if_every_rank_has_one(
    autotune_run, monkeypatch
):
    """FlashInfer keys MoE entries by rank, so each rank adopts its own table,
    and a rank that skipped the pass would leave a tuning one blocked in the
    per-tactic reduce: one missing table retunes the whole group."""
    run = autotune_run(pp=1)
    daemon = {
        f"dp0_rank{rank}": json.dumps({"shared_gemm": 9}).encode() for rank in range(3)
    }
    _serve_from_daemon(run, monkeypatch, daemon)

    run.execute()
    run.assert_collectives_match()
    assert set(run.profile_groups) == {0, 1, 2, 3}
    for rank in range(4):
        assert json.loads(daemon[f"dp0_rank{rank}"]) == {"shared_gemm": 0}


def test_daemon_hit_still_enables_kimi_k3_projection_overlap(autotune_run, monkeypatch):
    """Kimi K3's QKVG warmup also raises a limit on the model that the tuned
    table does not carry, so adopting the table must not skip it."""
    run = autotune_run(pp=1)
    daemon = {
        f"dp0_rank{rank}": json.dumps({"shared_gemm": 0}).encode() for rank in range(4)
    }
    _serve_from_daemon(run, monkeypatch, daemon)
    warmed = []
    monkeypatch.setattr(
        warmup, "_autotune_kimi_k3_kda_qkvg", lambda model: warmed.append(run.rank)
    )

    run.execute()
    assert not run.profile_groups
    assert warmed == [0, 1, 2, 3]


def _autotune_key(skip_ops=None, *, config_hash="cfg", workspace="ws", rank_id=""):
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        flashinfer_autotune_artifact_key,
    )

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda include_version: config_hash)
    )
    with patch(
        "vllm.model_executor.warmup.flashinfer_autotune_cache._flashinfer_workspace_id",
        return_value=workspace,
    ):
        return flashinfer_autotune_artifact_key(runner, skip_ops, rank_id)


def test_flashinfer_autotune_key_distinguishes_skipped_ops():
    """A hit lets an engine skip the autotune pass outright, so a table tuned
    with ops skipped must not be adopted by an engine that tunes them -- those
    ops would silently stay on their heuristics."""
    assert _autotune_key() != _autotune_key({"fp4_gemm"})
    assert _autotune_key({"fp4_gemm"}) == _autotune_key(["fp4_gemm"])


def test_flashinfer_autotune_key_tracks_config_build_and_rank():
    """Tactics are only valid for the graph they were measured on, the build
    that measured them, and (for MoE entries) the rank they were tuned on."""
    assert _autotune_key() != _autotune_key(config_hash="other")
    assert _autotune_key() != _autotune_key(workspace="other")
    assert _autotune_key(rank_id="dp0_rank0") != _autotune_key(rank_id="dp0_rank1")
