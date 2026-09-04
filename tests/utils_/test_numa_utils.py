# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig
from vllm.utils import numa_utils


@pytest.fixture(autouse=True)
def _disable_pct_by_default(monkeypatch):
    """Force PCT detection OFF unless a test opts in via ``_patch_pct_gates``.

    The CI / dev machines themselves can be Xeon 6776P with PCT enabled, so a
    plain ``cache_clear`` would let the gate auto-detect ``True`` from the
    live filesystem and silently re-route "baseline" tests through the PCT
    path. Stub ``/proc/cpuinfo`` and ``acpi_cppc/highest_perf`` to a state
    that fails the gate; ``_patch_pct_gates`` re-stubs on top when needed.
    """
    from io import StringIO

    real_open = open

    def _no_pct_open(path, *args, **kwargs):
        if path == numa_utils._PROC_CPUINFO_PATH:
            return StringIO("processor\t: 0\nmodel name\t: Generic Test CPU\n")
        if path == numa_utils._PCT_HIGHEST_PERF_PATH:
            raise OSError("PCT disabled by autouse fixture")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _no_pct_open)
    numa_utils._pct_sku_config.cache_clear()
    yield
    numa_utils._pct_sku_config.cache_clear()


def _make_config(**parallel_kwargs):
    parallel_defaults = dict(
        numa_bind=False,
        numa_bind_nodes=None,
        numa_bind_cpus=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    parallel_defaults.update(parallel_kwargs)
    parallel_config = SimpleNamespace(**parallel_defaults)
    return SimpleNamespace(parallel_config=parallel_config)


def test_get_numactl_args_with_node_binding():
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[0, 1])
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--cpunodebind=1 --membind=1"
    )


def test_get_numactl_args_with_cpu_binding():
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 1],
        numa_bind_cpus=["0-3", "4-7"],
    )
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--physcpubind=4-7 --membind=1"
    )


def _patch_pct_gates(
    monkeypatch,
    *,
    model_match: bool,
    highest_perf: int | None,
    cpulist: str | None = "0-31,64-95",
    cpulist_by_node: dict[int, str | None] | None = None,
    sku: str = "6776P",
):
    """Force `_pct_sku_config` and node cpulist read to deterministic state.

    ``cpulist`` is the default returned for any node not present in
    ``cpulist_by_node``. ``cpulist_by_node`` lets a test return different
    cpulists for different NUMA nodes (e.g. node 0 vs node 1). ``sku`` lets
    the test pick which Granite Rapids SKU appears in the fake
    ``/proc/cpuinfo`` ``model name`` (only used when ``model_match=True``).
    """
    import pathlib
    from io import StringIO

    import regex as re

    cpuinfo = (
        f"processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum {sku} CPU @ 2.40GHz\n"
        if model_match
        else "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum 8480+\n"
    )

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == numa_utils._PROC_CPUINFO_PATH:
            return StringIO(cpuinfo)
        if path == numa_utils._PCT_HIGHEST_PERF_PATH:
            if highest_perf is None:
                raise OSError("missing")
            return StringIO(f"{highest_perf}\n")
        return real_open(path, *args, **kwargs)

    real_read_text = pathlib.Path.read_text
    cpulist_by_node = cpulist_by_node or {}

    def fake_read_text(self, *args, **kwargs):
        path_str = str(self)
        if path_str.endswith("/cpulist") and "/sys/devices/system/node" in path_str:
            match = re.search(r"/node(\d+)/cpulist$", path_str)
            if match:
                node_id = int(match.group(1))
                if node_id in cpulist_by_node:
                    val = cpulist_by_node[node_id]
                    if val is None:
                        raise OSError(f"missing cpulist for node{node_id}")
                    return val
            if cpulist is None:
                raise OSError("missing cpulist")
            return cpulist
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr("pathlib.Path.read_text", fake_read_text)
    numa_utils._pct_sku_config.cache_clear()


def test_pct_binding_filters_cpus(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46)
    assert numa_utils._maybe_get_pct_cpu_binding([0]) == [0, 1, 16, 17, 64, 65, 80, 81]


@pytest.mark.parametrize(
    "sku,expected_cpus",
    [
        # 64-core SKUs (stride 16): cpus from "0-31,64-95" with cpu_id % 16
        # in (0, 1) -> 0, 1, 16, 17, 64, 65, 80, 81.
        ("6776P", [0, 1, 16, 17, 64, 65, 80, 81]),
        ("6774P", [0, 1, 16, 17, 64, 65, 80, 81]),
        # 72-core SKU (stride 18): cpus from "0-31,64-95" with cpu_id % 18
        # in (0, 1) -> 0, 1, 18, 19, 72, 73, 90, 91.
        ("6962P", [0, 1, 18, 19, 72, 73, 90, 91]),
    ],
)
def test_pct_binding_fires_on_every_capable_sku(monkeypatch, sku, expected_cpus):
    """Each SKU in ``_PCT_CAPABLE_SKUS`` engages the gate at its own
    expected ``highest_perf`` and uses its own priority-core stride."""
    sku_config = numa_utils._PCT_CAPABLE_SKUS[sku]
    _patch_pct_gates(
        monkeypatch,
        model_match=True,
        highest_perf=sku_config.highest_perf,
        sku=sku,
    )
    assert numa_utils._maybe_get_pct_cpu_binding([0]) == expected_cpus


def test_pct_binding_fails_closed_when_sku_perf_mismatch(monkeypatch):
    """6962P with 6776P's highest_perf (46 vs expected 44) must fail closed."""
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46, sku="6962P")
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_pct_binding_disabled_when_cpu_model_mismatch(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=False, highest_perf=46)
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_pct_binding_disabled_when_highest_perf_does_not_match(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=42)
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_pct_binding_disabled_when_files_missing(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=None)
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_pct_binding_returns_none_when_node_cpulist_filter_empty(monkeypatch):
    _patch_pct_gates(
        monkeypatch,
        model_match=True,
        highest_perf=46,
        cpulist="2-15,18-31",
    )
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_pct_binding_returns_none_when_node_cpulist_missing(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46, cpulist=None)
    assert numa_utils._maybe_get_pct_cpu_binding([0]) is None


def test_get_numactl_args_uses_pct_when_user_did_not_specify_cpus(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46)
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[0, 1])
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--physcpubind=0,1,16,17,64,65,80,81 --membind=1"
    )


def test_get_numactl_args_engine_core_baseline_single_node_shard():
    """Baseline (no PCT): single-NUMA shard -> single-node bind."""
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[0, 1])
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == "--cpunodebind=0 --membind=0"
    )


def test_get_numactl_args_engine_core_baseline_spans_shard_numa_nodes():
    """Baseline (no PCT): a TP=4 shard spanning both NUMA nodes -> bind to both."""
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        tensor_parallel_size=4,
    )
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == "--cpunodebind=0,1 --membind=0,1"
    )


def test_get_numactl_args_engine_core_pct_spans_shard_numa_nodes(monkeypatch):
    """PCT: EngineCore for a multi-NUMA shard binds to the union of priority
    cores across all shard nodes, so worker `--physcpubind` is always a
    subset of EngineCore's `cpus_allowed`."""
    _patch_pct_gates(
        monkeypatch,
        model_match=True,
        highest_perf=46,
        cpulist_by_node={0: "0-31,128-159", 1: "64-95,192-223"},
    )
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        tensor_parallel_size=4,
    )
    assert numa_utils._get_numactl_enginecore_args(
        vllm_config.parallel_config, local_rank=0
    ) == (
        "--physcpubind="
        "0,1,16,17,64,65,80,81,128,129,144,145,192,193,208,209"
        " --membind=0,1"
    )


def test_get_numactl_args_engine_core_pct_dp_shard_picks_local_nodes(monkeypatch):
    """With DP=2, each shard's EngineCore binds only to its own NUMA nodes."""
    _patch_pct_gates(
        monkeypatch,
        model_match=True,
        highest_perf=46,
        cpulist_by_node={0: "0-31,128-159", 1: "64-95,192-223"},
    )
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        tensor_parallel_size=2,
        data_parallel_rank_local=1,
    )
    # Shard 1 owns gpu_indices 2 and 3 -> nodes [1, 1] -> {1}.
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == "--physcpubind=64,65,80,81,192,193,208,209 --membind=1"
    )


def test_get_numactl_args_engine_core_pct_external_launcher_spans_local_nodes(
    monkeypatch,
):
    """external_launcher (or multi-node-within-DP, or Ray) hits the
    fallback branch. EngineCore must still span every local NUMA node
    so it can mp-spawn its local workers without ``--physcpubind``
    strict-validation failures."""
    _patch_pct_gates(
        monkeypatch,
        model_match=True,
        highest_perf=46,
        cpulist_by_node={0: "0-31,128-159", 1: "64-95,192-223"},
    )
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 0, 0, 1, 1, 1, 1],
        distributed_executor_backend="external_launcher",
        tensor_parallel_size=8,
    )
    assert numa_utils._get_numactl_enginecore_args(
        vllm_config.parallel_config, local_rank=0
    ) == (
        "--physcpubind="
        "0,1,16,17,64,65,80,81,128,129,144,145,192,193,208,209"
        " --membind=0,1"
    )


def test_get_numactl_args_engine_core_baseline_multi_node_within_dp_spans_locals():
    """Multi-node-within-DP fallback: bind EngineCore to all local NUMA
    nodes that the visible ``numa_bind_nodes`` reference."""
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 0, 0, 1, 1, 1, 1],
        nnodes_within_dp=2,
        tensor_parallel_size=8,
    )
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == "--cpunodebind=0,1 --membind=0,1"
    )


def test_get_numactl_args_engine_core_skips_user_cpu_list(monkeypatch):
    """EngineCore ignores ``--numa-bind-cpus``.

    Those are per-worker lists; binding EngineCore to any of them would
    shrink its ``cpus_allowed`` below the strict-superset workers'
    ``--physcpubind`` spawns need. We fall back to ``--cpunodebind`` over
    the shard's NUMA nodes instead. PCT auto-detect is also bypassed when
    the user is explicit (its priority-core union may not be a superset
    of the user's per-worker cores)."""
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46)
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        numa_bind_cpus=["0-3", "4-7", "64-67", "68-71"],
        tensor_parallel_size=4,
    )
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == "--cpunodebind=0,1 --membind=0,1"
    )


def test_get_numactl_args_user_cpus_override_pct(monkeypatch):
    _patch_pct_gates(monkeypatch, model_match=True, highest_perf=46)
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 1],
        numa_bind_cpus=["0-3", "4-7"],
    )
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--physcpubind=4-7 --membind=1"
    )


def test_get_numactl_args_uses_dp_offset():
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1],
        data_parallel_rank_local=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
    )
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--cpunodebind=1 --membind=1"
    )


def test_get_numactl_args_requires_detectable_nodes(monkeypatch):
    vllm_config = _make_config(numa_bind=True)
    monkeypatch.setattr(numa_utils, "get_auto_numa_nodes", lambda: None)
    with pytest.raises(RuntimeError):
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=0)


def test_configure_subprocess_rejects_unknown_process_kind():
    """configure_subprocess only knows 'worker' and 'EngineCore'; anything
    else must raise ValueError instead of silently routing to the worker
    path."""
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[0])
    with (
        pytest.raises(ValueError, match="process_kind"),
        numa_utils.configure_subprocess(
            vllm_config, local_rank=0, process_kind="bogus"
        ),
    ):
        pass


def test_log_numactl_show(monkeypatch):
    log_lines = []

    def fake_debug(msg, *args):
        log_lines.append(msg % args)

    monkeypatch.setattr(numa_utils.logger, "debug", fake_debug)
    monkeypatch.setattr(
        numa_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="policy: bind\nphyscpubind: 0 1 2 3\n", returncode=0
        ),
    )

    assert numa_utils._log_numactl_show("Worker_0") is True
    assert log_lines == [
        "Worker_0 affinity: policy: bind, physcpubind: 0 1 2 3",
    ]


def test_get_numactl_executable_points_to_fixed_wrapper(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/numactl")
    executable, debug_str = numa_utils._get_numactl_executable()
    assert executable.endswith("/vllm/utils/numa_wrapper.sh")
    assert "_VLLM_INTERNAL_NUMACTL_ARGS" in debug_str


def test_set_numa_wrapper_env_restores_previous_values():
    os.environ[numa_utils._NUMACTL_ARGS_ENV] = "old-args"
    os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] = "old-python"

    with numa_utils._set_numa_wrapper_env("new-args", "new-python"):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "new-args"
        assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "new-python"

    assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "old-args"
    assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "old-python"


def test_set_numa_wrapper_env_clears_values_when_unset():
    os.environ.pop(numa_utils._NUMACTL_ARGS_ENV, None)
    os.environ.pop(numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV, None)

    with numa_utils._set_numa_wrapper_env("new-args", "new-python"):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "new-args"
        assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "new-python"

    assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV not in os.environ


def test_parallel_config_validates_numa_bind_nodes():
    with pytest.raises(ValueError, match="non-negative"):
        ParallelConfig(numa_bind_nodes=[0, -1])


@pytest.mark.parametrize("cpuset", ["", "abc", "1-", "4-1", "1,,2", "1:2"])
def test_parallel_config_rejects_invalid_numa_bind_cpus(cpuset):
    with pytest.raises(ValueError, match="numa_bind_cpus"):
        ParallelConfig(numa_bind_cpus=[cpuset])


def _fake_numactl_run(rejected_args):
    """Fake ``numactl`` that fails when any of ``rejected_args`` is present."""

    def run(cmd, *args, **kwargs):
        arg_str = " ".join(cmd[1:-1])
        ok = not any(bad in arg_str for bad in rejected_args)
        return SimpleNamespace(returncode=0 if ok else 1)

    return run


def test_configure_subprocess_numa_fallback(monkeypatch):
    import multiprocessing

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/numactl")
    monkeypatch.setattr(numa_utils.envs, "VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    node_config = _make_config(numa_bind=True, numa_bind_nodes=[0])

    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    with numa_utils.configure_subprocess(node_config, local_rank=0):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=0 --membind=0"

    membind_fails = _fake_numactl_run(["--membind="])
    monkeypatch.setattr(numa_utils.subprocess, "run", membind_fails)
    with numa_utils.configure_subprocess(node_config, local_rank=0):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=0"

    cpu_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0],
        numa_bind_cpus=["0-3"],
    )
    with numa_utils.configure_subprocess(cpu_config, local_rank=0):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--physcpubind=0-3"

    before = multiprocessing.spawn.get_executable()
    monkeypatch.setattr(
        numa_utils.subprocess,
        "run",
        _fake_numactl_run(["--cpunodebind=", "--membind="]),
    )
    with numa_utils.configure_subprocess(node_config, local_rank=0):
        assert multiprocessing.spawn.get_executable() == before
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
