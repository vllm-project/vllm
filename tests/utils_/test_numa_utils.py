# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import Any

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
    parallel_defaults: dict[str, Any] = dict(
        numa_bind=False,
        numa_bind_nodes=None,
        numa_bind_cpus=None,
        assigned_physical_gpu_ids=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        data_parallel_size_local=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    parallel_defaults.update(parallel_kwargs)
    parallel_defaults.setdefault(
        "local_world_size",
        parallel_defaults["pipeline_parallel_size"]
        * parallel_defaults["tensor_parallel_size"]
        * parallel_defaults["prefill_context_parallel_size"]
        // parallel_defaults["nnodes_within_dp"],
    )
    parallel_config = SimpleNamespace(**parallel_defaults)
    parallel_config.world_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
        * parallel_config.pipeline_parallel_size
    )
    parallel_config.get_local_gpu_index = partial(
        ParallelConfig.get_local_gpu_index, parallel_config
    )
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


@pytest.mark.parametrize("pcp,pp", [(1, 1), (2, 1), (2, 2)])
def test_get_numactl_args_uses_dp_offset(pcp, pp):
    workers_per_dp = 2 * pcp * pp
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0] * workers_per_dp + [1] * workers_per_dp,
        data_parallel_rank_local=1,
        pipeline_parallel_size=pp,
        tensor_parallel_size=2,
        prefill_context_parallel_size=pcp,
    )
    assert (
        numa_utils._get_numactl_worker_args(vllm_config.parallel_config, local_rank=1)
        == "--cpunodebind=1 --membind=1"
    )


@pytest.mark.parametrize(
    "local_dp_rank,global_dp_rank,launch_dp_rank",
    [(1, 3, None), (None, 1, None), (0, 0, 1)],
    ids=["local-rank", "global-fallback", "launch-override"],
)
def test_pcp_worker_cpu_binding_uses_dp_gpu_index(
    local_dp_rank, global_dp_rank, launch_dp_rank
):
    """CPU overrides use the same PCP-aware GPU index as device selection."""
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0] * 4 + [1] * 4,
        numa_bind_cpus=[f"{4 * i}-{4 * i + 3}" for i in range(8)],
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_rank_local=local_dp_rank,
        data_parallel_index=global_dp_rank,
    )
    assert (
        numa_utils._get_numactl_worker_args(
            vllm_config.parallel_config, local_rank=1, dp_local_rank=launch_dp_rank
        )
        == "--physcpubind=20-23 --membind=1"
    )


@pytest.mark.parametrize("dp_rank,expected_nodes", [(0, "0,1"), (1, "2,3")])
def test_pcp_engine_core_covers_all_shard_workers(dp_rank, expected_nodes):
    """EngineCore includes every PCP worker's NUMA node in its DP shard."""
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 0, 1, 1, 2, 2, 3, 3],
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_rank_local=dp_rank,
    )
    assert (
        numa_utils._get_numactl_enginecore_args(
            vllm_config.parallel_config, local_rank=0
        )
        == f"--cpunodebind={expected_nodes} --membind={expected_nodes}"
    )


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"distributed_executor_backend": "ray"},
        {"distributed_executor_backend": "external_launcher"},
        {"data_parallel_backend": "ray"},
        {"nnodes_within_dp": 2},
    ],
    ids=["ray-executor", "external-launcher", "ray-dp", "multi-node-dp"],
)
def test_pcp_worker_keeps_local_gpu_index_without_dp_offset(parallel_kwargs):
    """Node-local launcher ranks do not acquire a colocated DP offset."""
    vllm_config = _make_config(
        numa_bind=True,
        numa_bind_nodes=[0, 1],
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_rank_local=1,
        **parallel_kwargs,
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


def test_parallel_config_numa_lists_imply_numa_bind():
    assert ParallelConfig().numa_bind is None
    assert ParallelConfig(numa_bind_nodes=[0, 1]).numa_bind is True
    assert ParallelConfig(numa_bind_cpus=["0-3"]).numa_bind is True
    with pytest.raises(ValueError, match="require numa_bind"):
        ParallelConfig(numa_bind=False, numa_bind_nodes=[0, 1])


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
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
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


def _patch_auto_numa(
    monkeypatch, start_method, detect, visible_gpus=1, node_cpus="0-7"
):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/numactl")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", start_method)
    monkeypatch.setattr(numa_utils, "_is_auto_numa_available", lambda **kwargs: True)
    monkeypatch.setattr(numa_utils, "get_auto_numa_nodes", detect)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.device_count", lambda: visible_gpus
    )
    real_read_text = numa_utils.Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path).startswith("/sys/devices/system/node/node"):
            return node_cpus
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(numa_utils.Path, "read_text", read_text)


def test_configure_subprocess_auto_binds_and_records_result(monkeypatch):
    """Default mode resolves once in the launcher and stores the result, so
    spawned children bind with it instead of re-checking eligibility."""
    _patch_auto_numa(monkeypatch, "spawn", lambda: [0])
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))

    vllm_config = _make_config(numa_bind=None)
    with numa_utils.configure_subprocess(
        vllm_config, local_rank=0, process_kind="EngineCore"
    ):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=0 --membind=0"
        child_config = deepcopy(vllm_config)
    assert vllm_config.parallel_config.numa_bind is True
    assert child_config.parallel_config.numa_bind_nodes == [0]

    monkeypatch.setattr(numa_utils, "get_auto_numa_nodes", _raise_probe_error)
    monkeypatch.setattr(numa_utils, "_is_auto_numa_available", lambda **kwargs: False)
    with numa_utils.configure_subprocess(child_config, local_rank=0):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=0 --membind=0"


def test_auto_numa_redetects_when_visible_devices_change(monkeypatch):
    """Independent launchers must not reuse another visible GPU set's mapping."""
    _patch_auto_numa(
        monkeypatch, "spawn", numa_utils.get_auto_numa_nodes, visible_gpus=2
    )
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    detections = []

    def detect():
        devices = os.environ["CUDA_VISIBLE_DEVICES"]
        detections.append(devices)
        return {"0,1": [0, 0], "2,3": [1, 1]}[devices]

    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_all_device_numa_nodes",
        detect,
        raising=False,
    )
    for devices, node in [("0,1", 0), ("2,3", 1)]:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
        vllm_config = _make_config(numa_bind=None, tensor_parallel_size=2)
        with numa_utils.configure_subprocess(
            vllm_config, local_rank=0, process_kind="EngineCore"
        ):
            numactl_args = os.environ[numa_utils._NUMACTL_ARGS_ENV]
            assert numactl_args == f"--cpunodebind={node} --membind={node}"
        assert vllm_config.parallel_config.numa_bind_nodes == [node, node]
    assert detections == ["0,1", "2,3"]


@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
@pytest.mark.parametrize(
    "policy",
    [0, 1, 2, 3, 4, 5, None],
    ids=[
        "default",
        "preferred",
        "bind",
        "interleave",
        "local",
        "preferred-many",
        "unreadable",
    ],
)
def test_configure_subprocess_respects_existing_mempolicy(
    monkeypatch, policy, explicit
):
    """Only explicit enablement may replace an existing non-default policy."""
    is_auto_numa_available = numa_utils._is_auto_numa_available
    _patch_auto_numa(monkeypatch, "spawn", numa_utils.get_auto_numa_nodes)
    monkeypatch.setattr(numa_utils, "_is_auto_numa_available", is_auto_numa_available)
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda_alike", lambda: True)
    monkeypatch.setattr(numa_utils.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(numa_utils.psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        numa_utils.psutil,
        "Process",
        lambda pid: SimpleNamespace(cpu_affinity=lambda: [0, 1, 2, 3]),
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_all_device_numa_nodes",
        lambda: [0],
        raising=False,
    )

    def get_mempolicy(mode, *_args):
        if policy is None:
            return -1
        mode._obj.value = policy
        return 0

    monkeypatch.setattr(
        numa_utils,
        "get_libnuma",
        lambda: SimpleNamespace(numa_available=lambda: 0, get_mempolicy=get_mempolicy),
    )
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    vllm_config = _make_config(numa_bind=True if explicit else None)
    if explicit and policy is None:
        with (
            pytest.raises(RuntimeError, match="could not detect"),
            numa_utils.configure_subprocess(vllm_config, local_rank=0),
        ):
            pass
        return

    expected_bind = policy is not None and (policy == 0 or explicit)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        numactl_args = os.environ.get(numa_utils._NUMACTL_ARGS_ENV)
        assert numactl_args == (
            "--cpunodebind=0 --membind=0" if expected_bind else None
        )
    assert vllm_config.parallel_config.numa_bind is expected_bind


def _raise_probe_error():
    raise RuntimeError("amdsmi init failed")


@pytest.mark.parametrize(
    ("start_method", "detect", "assigned_gpus"),
    [
        ("fork", lambda: [0], None),
        ("spawn", lambda: None, None),
        ("spawn", lambda: [], None),
        ("spawn", _raise_probe_error, None),
        ("spawn", lambda: [0, 1], [1]),
    ],
    ids=["fork", "no-topology", "no-visible-gpus", "probe-raises", "device-ids"],
)
def test_configure_subprocess_auto_falls_back_to_unbound(
    monkeypatch, start_method, detect, assigned_gpus
):
    """Default mode never fails startup or forces spawn; when binding cannot
    engage (or would pick the wrong node) it starts unbound."""
    import multiprocessing

    _patch_auto_numa(monkeypatch, start_method, detect)

    vllm_config = _make_config(numa_bind=None, assigned_physical_gpu_ids=assigned_gpus)
    before = multiprocessing.spawn.get_executable()
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert multiprocessing.spawn.get_executable() == before
    assert vllm_config.parallel_config.numa_bind is False


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"distributed_executor_backend": "ray"},
        {"data_parallel_backend": "ray"},
        {"distributed_executor_backend": "external_launcher"},
        {"prefill_context_parallel_size": 2},
    ],
    ids=["ray-executor", "ray-dp", "external-launcher", "pcp"],
)
def test_auto_numa_skips_unsupported_launchers(monkeypatch, parallel_kwargs):
    detections = []

    def detect():
        detections.append(True)
        return [0, 1]

    _patch_auto_numa(monkeypatch, "spawn", detect, visible_gpus=2)
    vllm_config = _make_config(numa_bind=None, **parallel_kwargs)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert vllm_config.parallel_config.numa_bind is False
    assert detections == []


@pytest.mark.parametrize("nodes", [[0], [0, 1, 2], [0, -1], [0, None]])
def test_auto_numa_rejects_incomplete_or_invalid_topology(monkeypatch, nodes):
    _patch_auto_numa(monkeypatch, "spawn", lambda: nodes, visible_gpus=2)
    vllm_config = _make_config(numa_bind=None, tensor_parallel_size=2)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert vllm_config.parallel_config.numa_bind is False
    assert vllm_config.parallel_config.numa_bind_nodes is None


def test_auto_numa_checks_all_local_dp_workers_before_first_engine(monkeypatch):
    _patch_auto_numa(monkeypatch, "spawn", lambda: [0, 1], visible_gpus=2)
    vllm_config = _make_config(
        numa_bind=None, tensor_parallel_size=2, data_parallel_size_local=2
    )
    with numa_utils.configure_subprocess(
        vllm_config, local_rank=0, dp_local_rank=0, process_kind="EngineCore"
    ):
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert vllm_config.parallel_config.numa_bind is False


def test_auto_numa_multi_node_topology_is_local(monkeypatch):
    _patch_auto_numa(monkeypatch, "spawn", lambda: [2, 3], visible_gpus=2)
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    vllm_config = _make_config(
        numa_bind=None, tensor_parallel_size=4, nnodes_within_dp=2
    )
    with numa_utils.configure_subprocess(
        vllm_config, local_rank=0, process_kind="EngineCore"
    ):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == (
            "--cpunodebind=2,3 --membind=2,3"
        )
        child_config = deepcopy(vllm_config)
    with numa_utils.configure_subprocess(child_config, local_rank=1):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=3 --membind=3"


@pytest.mark.parametrize("node_cpus", [None, "", "invalid"])
def test_auto_numa_skips_nodes_without_usable_cpus(monkeypatch, node_cpus):
    _patch_auto_numa(monkeypatch, "spawn", lambda: [0], node_cpus=node_cpus)
    vllm_config = _make_config(numa_bind=None)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert vllm_config.parallel_config.numa_bind is False


@pytest.mark.parametrize(
    "rejected_args, expected_args",
    [("--membind=", "--cpunodebind=0"), ("--cpunodebind=", None)],
)
def test_auto_numa_records_effective_policy(
    monkeypatch, caplog, rejected_args, expected_args
):
    caplog.set_level("INFO", logger=numa_utils.logger.name)
    _patch_auto_numa(monkeypatch, "spawn", lambda: [0])
    monkeypatch.setattr(
        numa_utils.subprocess, "run", _fake_numactl_run([rejected_args])
    )
    vllm_config = _make_config(numa_bind=None)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert os.environ.get(numa_utils._NUMACTL_ARGS_ENV) == expected_args
        assert vllm_config.parallel_config.numa_bind is (expected_args is not None)
    launch_logs = [r.getMessage() for r in caplog.records if "Launching" in r.message]
    if expected_args is None:
        assert launch_logs == []
        assert vllm_config.parallel_config.numa_bind_nodes is None
    else:
        assert len(launch_logs) == 1
        assert launch_logs[0].endswith(f"numactl {expected_args}")


def test_auto_numa_does_not_swallow_launch_errors(monkeypatch):
    import multiprocessing

    _patch_auto_numa(monkeypatch, "spawn", lambda: [0])
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    monkeypatch.delenv(numa_utils._NUMACTL_ARGS_ENV, raising=False)
    monkeypatch.delenv(numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV, raising=False)
    original_executable = multiprocessing.spawn.get_executable()
    with (
        pytest.raises(RuntimeError, match="worker startup failed"),
        numa_utils.configure_subprocess(_make_config(numa_bind=None), local_rank=0),
    ):
        raise RuntimeError("worker startup failed")
    assert multiprocessing.spawn.get_executable() == original_executable
    assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV not in os.environ


@pytest.mark.parametrize("missing_numactl", [False, True])
def test_auto_numa_skips_unavailable_environment(monkeypatch, missing_numactl):
    _patch_auto_numa(monkeypatch, "spawn", lambda: [0])
    if missing_numactl:
        monkeypatch.setattr("shutil.which", lambda name: None)
    else:
        monkeypatch.setattr(
            numa_utils, "_is_auto_numa_available", lambda **kwargs: False
        )
    vllm_config = _make_config(numa_bind=None)
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert numa_utils._NUMACTL_ARGS_ENV not in os.environ
    assert vllm_config.parallel_config.numa_bind is False


def test_explicit_numa_nodes_bypass_auto_eligibility(monkeypatch):
    _patch_auto_numa(monkeypatch, "spawn", _raise_probe_error)
    monkeypatch.setattr(numa_utils, "_is_auto_numa_available", lambda **kwargs: False)
    monkeypatch.setattr(numa_utils.subprocess, "run", _fake_numactl_run([]))
    vllm_config = _make_config(numa_bind=True, numa_bind_nodes=[7])
    with numa_utils.configure_subprocess(vllm_config, local_rank=0):
        assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=7 --membind=7"


def test_explicit_numa_requires_detectable_topology(monkeypatch):
    _patch_auto_numa(monkeypatch, "spawn", lambda: None)
    with (
        pytest.raises(RuntimeError, match="could not detect"),
        numa_utils.configure_subprocess(_make_config(numa_bind=True), local_rank=0),
    ):
        pass


@pytest.mark.parametrize("numa_bind", [False, True])
def test_explicit_numa_handles_missing_numactl(monkeypatch, numa_bind):
    monkeypatch.setattr("shutil.which", lambda name: None)
    vllm_config = _make_config(numa_bind=numa_bind, numa_bind_nodes=[0])
    if numa_bind:
        with (
            pytest.raises(RuntimeError, match="numactl is required"),
            numa_utils.configure_subprocess(vllm_config, local_rank=0),
        ):
            pass
    else:
        with numa_utils.configure_subprocess(vllm_config, local_rank=0):
            assert numa_utils._NUMACTL_ARGS_ENV not in os.environ


@pytest.mark.parametrize("affinity", [[0, 1], None, [3, 2, 1, 0]])
def test_auto_numa_checks_cpu_affinity(monkeypatch, affinity):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda_alike", lambda: True)
    monkeypatch.setattr(numa_utils.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(numa_utils.psutil, "cpu_count", lambda: 4)
    process = SimpleNamespace()
    if affinity is not None:
        process.cpu_affinity = lambda: affinity
    monkeypatch.setattr(numa_utils.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(numa_utils, "_get_mempolicy", lambda: 0)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_all_device_numa_nodes",
        lambda: [0, 1],
        raising=False,
    )
    assert numa_utils._is_auto_numa_available(verbose=False, strict=True) is (
        affinity == [3, 2, 1, 0]
    )
