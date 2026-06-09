# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VLLM_RAY_DP_PLACEMENT_NODE_IPS."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.v1.engine.utils as utils
from vllm.v1.engine.utils import CoreEngineActorManager


def _vllm_config(
    *, dp_size, dp_local, master_ip, world_size=1, all2all_backend="naive"
):
    parallel = SimpleNamespace(
        data_parallel_master_ip=master_ip,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_local,
        world_size=world_size,
        all2all_backend=all2all_backend,
    )
    return SimpleNamespace(parallel_config=parallel)


def _resources(node_gpus: dict[str, int]):
    # node_gpus: {ip: gpu_count}; plus a CPU-only head node.
    res = {
        f"id-{ip}": {"GPU": float(g), f"node:{ip}": 1.0} for ip, g in node_gpus.items()
    }
    res["id-head"] = {
        "CPU": 8.0,
        "node:__internal_head__": 1.0,
        "node:10.9.9.9": 1.0,
    }
    return res


def _run(cfg, resources):
    created = []

    def fake_pg(name, strategy, bundles):
        created.append({"name": name, "strategy": strategy, "bundles": bundles})
        return object()

    with (
        patch(
            "ray._private.state.available_resources_per_node",
            return_value=resources,
        ),
        patch.object(utils, "current_platform", SimpleNamespace(ray_device_key="GPU")),
        patch("ray.util.placement_group", side_effect=fake_pg),
    ):
        pgs, local_ranks = CoreEngineActorManager.create_dp_placement_groups(cfg)
    return pgs, local_ranks, created


def _pinned_ips(created):
    return {
        key.split(":", 1)[1]
        for pg in created
        for bundle in pg["bundles"]
        for key in bundle
        if key.startswith("node:")
    }


def test_allowlist_confines_dp_to_listed_nodes(monkeypatch):
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.1,10.0.0.3")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8, "10.0.0.3": 8, "10.0.0.4": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    pgs, _, created = _run(cfg, resources)

    assert len(pgs) == 16  # 8 on .1 (master) + 8 on .3
    assert _pinned_ips(created) <= {"10.0.0.1", "10.0.0.3"}


def test_empty_allowlist_is_noop(monkeypatch):
    monkeypatch.delenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", raising=False)
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    pgs, _, created = _run(cfg, resources)

    assert len(pgs) == 16
    assert _pinned_ips(created) == {"10.0.0.1", "10.0.0.2"}  # all nodes used


def test_master_auto_added_with_warning(monkeypatch):
    # Allowlist omits the master; vLLM must still keep it and warn.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.3")
    resources = _resources({"10.0.0.1": 8, "10.0.0.3": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")
    _, _, created = _run(cfg, resources)

    assert _pinned_ips(created) == {"10.0.0.1", "10.0.0.3"}


def test_allowlist_isolates_two_engines(monkeypatch):
    # Engine B is confined to .2/.4, so it can never touch engine A's master .1.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.2,10.0.0.4")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8, "10.0.0.3": 8, "10.0.0.4": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.2")

    _, _, created = _run(cfg, resources)

    assert _pinned_ips(created) <= {"10.0.0.2", "10.0.0.4"}


def test_allowlist_too_small_raises(monkeypatch):
    # Master alone can't hold all ranks and no other node is allowed.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.1")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    with pytest.raises(ValueError):  # not enough placement groups created
        _run(cfg, resources)
