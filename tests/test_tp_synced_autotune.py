# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VLLM_TRITON_SYNC_AUTOTUNE_TP (CPU-only, no kernel launch)."""

from types import SimpleNamespace

import pytest

from vllm.triton_utils import HAS_TRITON, tl, triton

if not HAS_TRITON:
    pytest.skip("triton not available", allow_module_level=True)

from vllm.triton_utils import tp_synced_autotune  # noqa: E402


class _FakeAutotuner:
    """Just the state check_disk_cache touches."""

    def __init__(self, disk_cache: bool):
        self._vllm_disk_cache = disk_cache
        self.cache: dict = {}
        self.configs_timings: dict = {}


CONFIGS = [triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})]


@pytest.fixture
def hooked(monkeypatch: pytest.MonkeyPatch):
    """check_disk_cache wrapped around a fake Triton one that records its calls."""
    calls: list[str] = []

    def triton_check_disk_cache(self, tuning_key, configs, bench_fn):
        calls.append("disk")
        self.cache[tuning_key] = configs[1]
        self.configs_timings = {configs[0]: [2.0], configs[1]: [1.0]}
        return True

    monkeypatch.setattr(tp_synced_autotune, "_layout_ok", True)
    return tp_synced_autotune._check_disk_cache(triton_check_disk_cache), calls


def _bench_fn(autotuner, calls, fastest):
    def bench():
        calls.append("bench")
        autotuner.cache["k"] = fastest
        autotuner.configs_timings = {c: [1.0 if c is fastest else 2.0] for c in CONFIGS}

    return bench


@pytest.mark.parametrize("disk_cache, expected", [(True, "disk"), (False, "bench")])
def test_outside_tp_resolves_locally(hooked, monkeypatch, disk_cache, expected):
    check, calls = hooked
    monkeypatch.setattr(tp_synced_autotune, "_sync_group", lambda: None)
    tuner = _FakeAutotuner(disk_cache)
    check(tuner, "k", CONFIGS, _bench_fn(tuner, calls, CONFIGS[0]))
    assert calls == [expected]


def _tp_group(monkeypatch, rank, payload=None):
    import torch.distributed as dist

    group = SimpleNamespace(ranks=[0, 1], cpu_group=object(), world_size=2)
    monkeypatch.setattr(tp_synced_autotune, "_sync_group", lambda: group)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    sent = []

    def fake_broadcast(box, src, group):
        assert src == 0
        if rank == 0:
            sent.append(box[0])
        else:
            box[0] = payload

    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast)
    return sent


@pytest.mark.parametrize("disk_cache", [True, False])
def test_rank0_resolves_and_broadcasts(hooked, monkeypatch, disk_cache):
    check, calls = hooked
    sent = _tp_group(monkeypatch, rank=0)
    tuner = _FakeAutotuner(disk_cache)
    check(tuner, "k", CONFIGS, _bench_fn(tuner, calls, CONFIGS[1]))
    assert calls == ["disk" if disk_cache else "bench"]
    assert sent == [([str(c) for c in CONFIGS], 1, [[2.0], [1.0]])]


@pytest.mark.parametrize("disk_cache", [True, False])
def test_other_rank_adopts_without_tuning(hooked, monkeypatch, disk_cache):
    check, calls = hooked
    payload = ([str(c) for c in CONFIGS], 0, [[1.0], [2.0]])
    _tp_group(monkeypatch, rank=1, payload=payload)
    tuner = _FakeAutotuner(disk_cache)
    assert check(tuner, "k", CONFIGS, _bench_fn(tuner, calls, CONFIGS[1])) is True
    assert calls == []  # neither its disk cache nor a benchmark
    assert tuner.cache["k"] is CONFIGS[0]
    assert tuner.configs_timings == {CONFIGS[0]: [1.0], CONFIGS[1]: [2.0]}


def test_other_rank_with_other_candidates_tunes_locally(hooked, monkeypatch):
    check, calls = hooked
    payload = (["something else"], 0, [[1.0]])
    _tp_group(monkeypatch, rank=1, payload=payload)
    tuner = _FakeAutotuner(disk_cache=False)
    check(tuner, "k", CONFIGS, _bench_fn(tuner, calls, CONFIGS[1]))
    assert calls == ["bench"]
    assert tuner.cache["k"] is CONFIGS[1]


def test_install_routes_new_keys_through_check_disk_cache(monkeypatch):
    monkeypatch.setattr(triton.knobs.autotuning, "cache", False)
    Autotuner = triton.runtime.autotuner.Autotuner
    monkeypatch.setattr(tp_synced_autotune, "_installed", False)
    monkeypatch.setattr(Autotuner, "__init__", Autotuner.__init__)
    monkeypatch.setattr(Autotuner, "check_disk_cache", Autotuner.check_disk_cache)
    tp_synced_autotune.install()
    hooked_init = Autotuner.__init__
    tp_synced_autotune.install()  # idempotent
    assert Autotuner.__init__ is hooked_init

    @triton.jit
    def kernel(x, BLOCK: tl.constexpr):
        pass

    tuner = Autotuner(kernel, kernel.arg_names, CONFIGS, ["x"], None, None)
    assert tuner.cache_results is True
    assert tuner._vllm_disk_cache is False


def _layout(ep=False, sp_moe=False, dcp=1, pcp=1):
    return SimpleNamespace(
        enable_expert_parallel=ep,
        use_sequence_parallel_moe=sp_moe,
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
    )


@pytest.mark.parametrize(
    "layout, ok",
    [
        (_layout(), True),
        (_layout(ep=True), False),
        (_layout(sp_moe=True), False),
        (_layout(dcp=2), False),
        (_layout(pcp=2), False),
    ],
)
def test_configure_disables_rank_dependent_layouts(monkeypatch, layout, ok):
    monkeypatch.setattr(tp_synced_autotune, "_layout_ok", None)
    tp_synced_autotune.configure(layout)
    assert tp_synced_autotune._layout_ok is ok


def test_no_sync_before_configure(monkeypatch):
    monkeypatch.setattr(tp_synced_autotune, "_layout_ok", None)
    assert tp_synced_autotune._sync_group() is None
