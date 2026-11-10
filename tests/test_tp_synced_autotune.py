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


def test_outside_tp_is_tritons_own(hooked, monkeypatch):
    check, calls = hooked
    monkeypatch.setattr(tp_synced_autotune, "_sync_group", lambda: None)
    tuner = _FakeAutotuner(disk_cache=False)
    check(tuner, "k", CONFIGS, _bench_fn(tuner, calls, CONFIGS[0]))
    assert calls == ["disk"]  # Triton reaches it only when its disk cache is on


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


@pytest.fixture
def real_tuner(monkeypatch: pytest.MonkeyPatch):
    """A real Triton Autotuner over a fake launch, with install() applied and a fake
    two-rank broadcast; yields (tuner, state) where state toggles the forward pass."""
    import torch.distributed as dist

    Autotuner = triton.runtime.autotuner.Autotuner
    monkeypatch.setattr(triton.knobs.autotuning, "cache", False)
    monkeypatch.setattr(tp_synced_autotune, "_installed", False)
    monkeypatch.setattr(tp_synced_autotune, "_layout_ok", True)
    monkeypatch.setattr(Autotuner, "run", Autotuner.run)
    monkeypatch.setattr(Autotuner, "check_disk_cache", Autotuner.check_disk_cache)
    state = SimpleNamespace(in_forward=False, benched=[], broadcasts=0)
    monkeypatch.setattr(
        Autotuner,
        "_bench",
        lambda self, *a, config, **k: (
            state.benched.append(config.kwargs["BLOCK"])
            or [1.0 if config.kwargs["BLOCK"] == 128 else 2.0]
        ),
    )
    group = SimpleNamespace(ranks=[0, 1], cpu_group=object(), world_size=2)
    monkeypatch.setattr(
        tp_synced_autotune,
        "_sync_group",
        lambda: group if state.in_forward else None,
    )
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def fake_broadcast(box, src, group):
        state.broadcasts += 1

    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast)
    tp_synced_autotune.install()

    @triton.jit
    def kernel(n, BLOCK: tl.constexpr):
        pass

    monkeypatch.setattr(kernel, "run", lambda *a, **k: k["BLOCK"])
    yield Autotuner(kernel, kernel.arg_names, CONFIGS, ["n"], None, None), state


def test_install_is_idempotent(real_tuner):
    Autotuner = triton.runtime.autotuner.Autotuner
    hooked = Autotuner.run
    tp_synced_autotune.install()
    assert Autotuner.run is hooked


def test_key_tuned_outside_forward_is_synced_on_first_tp_use(real_tuner):
    tuner, state = real_tuner
    assert tuner.run(1000) == 128  # outside a forward: Triton's own autotune
    assert state.benched == [64, 128] and state.broadcasts == 0
    state.in_forward = True
    state.benched.clear()
    assert tuner.run(1000) == 128  # the local cache hit must not skip the sync
    assert state.benched == [64, 128] and state.broadcasts == 1
    assert tuner.run(1000) == 128  # synced key: no further broadcast
    assert state.broadcasts == 1


def test_synced_run_restores_local_state(real_tuner):
    tuner, state = real_tuner
    local_cache = tuner.cache
    state.in_forward = True
    tuner.run(1000)
    assert tuner.cache is local_cache and not local_cache
    assert tuner.cache_results is False
    assert list(tuner._vllm_synced_cache) == [(1000,)]


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
