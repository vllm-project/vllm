# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch

import vllm.model_executor.warmup.kernel_warmup as warmup
import vllm.utils.flashinfer as fi_utils
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
            parallel_config=SimpleNamespace(
                enable_elastic_ep=False, nnodes=1, pipeline_parallel_size=1
            ),
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


@pytest.mark.parametrize(
    "has_v2,elastic_ep,node_count",
    [(False, False, 1), (True, False, 1), (True, True, 1), (True, False, 2)],
    ids=["v1", "v2", "elastic-v1", "multi-node-v1"],
)
@pytest.mark.parametrize(
    "hisparse,world_size,rank,skip_ops,failure",
    [
        (False, 1, 0, [], None),
        (False, 2, 0, [], None),
        (False, 2, 1, [], None),
        (True, 2, 0, ["fp4_gemm", "bmm_fp8"], None),
        (False, 2, 1, ["fp4_gemm"], "dummy"),
        (False, 2, 1, [], "replayssm"),
        (True, 2, 0, [], "kimi"),
        (True, 2, 1, ["bmm_fp8"], "bf16"),
    ],
)
def test_flashinfer_autotune_lifecycle(
    monkeypatch,
    tmp_path,
    has_v2,
    elastic_ep,
    node_count,
    hisparse,
    world_size,
    rank,
    skip_ops,
    failure,
):
    use_v2 = has_v2 and not elastic_ep and node_count == 1
    runner = _make_runner([])
    runner.vllm_config.parallel_config.enable_elastic_ep = elastic_ep
    runner.vllm_config.attention_config.hisparse_config = object() if hisparse else None
    model = runner.get_model()
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    events = Mock()
    events.broadcast.return_value = b"cached"
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        cpu_group=object(),
        broadcast_object=events.broadcast,
        barrier=events.barrier,
    )

    in_autotune = False

    @contextmanager
    def autotune_context(**kwargs):
        nonlocal in_autotune
        assert torch.is_inference_mode_enabled()
        in_autotune = True
        events.enter()
        try:
            yield
        finally:
            in_autotune = False
            events.exit()

    def bf16_warmup(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not in_autotune
        if failure == "bf16":
            raise RuntimeError("warmup failed")

    events.autotune.side_effect = autotune_context
    if failure is not None:
        getattr(events, failure).side_effect = RuntimeError("warmup failed")
    events.bf16.side_effect = bf16_warmup
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(
            AutoTuner=SimpleNamespace(get=lambda: events.tuner),
            set_autotune_process_group=events.set_group,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=events.autotune, autotune_v2_reload=events.reload),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_node_count", lambda: node_count
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: has_v2)
    monkeypatch.setattr(fi_utils, "autotune", events.autotune)
    monkeypatch.setattr(warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", skip_ops)
    resolve_file = Mock(return_value=cache_path)
    resolve_root = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_file", resolve_file)
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_v2_root", resolve_root)
    monkeypatch.setattr(warmup, "write_flashinfer_autotune_cache", events.write)
    monkeypatch.setattr(warmup, "_run_flashinfer_autotune_dummy_runs", events.dummy)
    monkeypatch.setattr(warmup, "replayssm_autotune_warmup", events.replayssm)
    monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", events.kimi)
    monkeypatch.setattr(warmup, "_run_flashinfer_bf16_autotune_dummy_run", events.bf16)
    monkeypatch.setattr(
        warmup, "autotune_hisparse_flashinfer_attention", events.hisparse
    )

    if failure is None:
        warmup.flashinfer_autotune(runner)
    else:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.flashinfer_autotune(runner)

    kwargs = {"skip_ops": set(skip_ops)} if skip_ops else {}
    expected = []
    if use_v2:
        resolve_file.assert_not_called()
        resolve_root.assert_called_once_with()
        expected.append(call.autotune(mode="tune", cache_root=str(tmp_path), **kwargs))
    else:
        resolve_root.assert_not_called()
        resolve_file.assert_called_once_with(runner)
        expected.extend(
            [
                call.broadcast(b"cached" if rank == 0 else None, src=0),
                call.write(cache_path, b"cached"),
                call.barrier(),
                call.tuner.load_configs(str(cache_path)),
                call.autotune(tune_mode=True, **kwargs),
            ]
        )
    expected.extend(
        [call.set_group(world.cpu_group if world_size > 1 else None), call.enter()]
    )
    if hisparse:
        expected.append(call.hisparse(runner))
    for name, arg, kwargs in [
        ("dummy", runner, {"skip_attn": hisparse}),
        ("replayssm", runner, {}),
        ("kimi", model, {}),
    ]:
        expected.append(getattr(call, name)(arg, **kwargs))
        if failure == name:
            break
    expected.append(call.exit())
    if failure in (None, "bf16"):
        expected.append(
            call.bf16(runner, skip_ops=set(skip_ops) or None, skip_attn=hisparse)
        )
    expected.append(call.set_group(None))
    if failure is None:
        if world_size > 1:
            expected.append(call.barrier())
        if use_v2 and world_size > 1:
            expected.extend([call.reload(), call.barrier()])
        elif not use_v2 and rank == 0:
            expected.append(call.tuner.save_configs(str(cache_path)))
    assert events.mock_calls == expected


@pytest.mark.parametrize(
    "node_count,elastic_ep,pp_size",
    [(1, False, 1), (2, False, 1), (1, True, 1), (1, False, 2)],
)
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("model_runner_v2,max_reqs", [(False, 2), (True, 1), (True, 2)])
def test_sparse_autotune_respects_cache_selection(
    monkeypatch,
    tmp_path,
    node_count,
    elastic_ep,
    pp_size,
    rank,
    model_runner_v2,
    max_reqs,
):
    import vllm.model_executor.warmup.flashinfer_sparse_mla_warmup as sparse

    runner = _make_runner([])
    runner.max_num_reqs = max_reqs
    runner.vllm_config.use_v2_model_runner = model_runner_v2
    runner.vllm_config.kernel_config.enable_flashinfer_autotune = True
    runner.vllm_config.parallel_config.enable_elastic_ep = elastic_ep
    runner.vllm_config.parallel_config.pipeline_parallel_size = pp_size
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=runner.vllm_config,
        execute_model=Mock(),
        sample_tokens=Mock(),
    )
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=2,
        broadcast_object=Mock(return_value=b"cached"),
        barrier=Mock(),
    )
    managed = Mock(return_value=nullcontext())
    legacy = Mock(return_value=nullcontext())
    reload = Mock()
    tuner = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=managed, autotune_v2_reload=reload),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(AutoTuner=SimpleNamespace(get=lambda: tuner)),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_node_count", lambda: node_count
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: True)
    monkeypatch.setattr(sparse, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        sparse.current_platform, "is_device_capability_family", lambda _: True
    )
    monkeypatch.setattr(
        sparse, "_flashinfer_sparse_mla_decode_label", lambda *_: "DSv3.2"
    )
    monkeypatch.setattr(sparse, "resolve_flashinfer_autotune_v2_root", lambda: tmp_path)
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    monkeypatch.setattr(
        sparse, "resolve_flashinfer_autotune_file", lambda _: cache_path
    )
    monkeypatch.setattr(sparse, "flashinfer_autotune", legacy)

    def mixed_warmup(*args, mixed_step_context=None, **kwargs):
        context = (
            mixed_step_context if mixed_step_context is not None else nullcontext()
        )
        with context:
            assert torch.is_inference_mode_enabled()
        return True

    mixed = Mock(side_effect=mixed_warmup)
    monkeypatch.setattr(sparse, "run_mixed_prefill_decode_warmup", mixed)
    assert sparse._run_flashinfer_sparse_mla_decode_autotune(worker, 16, frozenset())

    if node_count == 1 and not elastic_ep and pp_size == 1:
        managed.assert_called_once_with(mode="tune", cache_root=tmp_path)
        legacy.assert_not_called()
        world.broadcast_object.assert_not_called()
        tuner.load_configs.assert_not_called()
        reload.assert_called_once_with()
        assert world.barrier.call_count == 2
    else:
        managed.assert_not_called()
        reload.assert_not_called()
        assert legacy.call_count == (1 if rank == 0 else 0)
        world.broadcast_object.assert_called_once_with(
            b"cached" if rank == 0 else None, src=0
        )
        tuner.load_configs.assert_called_once_with(str(cache_path))
    assert mixed.call_count == (1 if model_runner_v2 and max_reqs >= 2 else 0)
    assert runner._dummy_run.call_count == (0 if mixed.called else 1)


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
            runner = _make_runner([])
            runner.vllm_config.parallel_config.pipeline_parallel_size = self.pp
            flashinfer_autotune(runner)
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
        monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: pp > 1)
        monkeypatch.setattr(parallel_state, "get_node_count", lambda: 1)
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
