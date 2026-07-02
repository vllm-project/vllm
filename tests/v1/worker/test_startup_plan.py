# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.worker import startup_plan
from vllm.v1.worker.startup_plan import (
    applicable_kv_cache_memory_bytes,
    load_startup_plan,
    maybe_apply_startup_plan,
    save_startup_plan,
)

GiB = 1 << 30


def _mock_platform(name="NVIDIA H100 PCIe", total_mem=80 * GiB, cap=(9, 0)):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: total_mem,
        get_device_capability=lambda device_id=0: cap,
    )


def _fingerprint(config_hash="abc123", rank=0, world_size=1, **platform_kwargs):
    vllm_config = SimpleNamespace(compute_hash=lambda: config_hash)
    platform = _mock_platform(**platform_kwargs)
    with patch.object(startup_plan, "current_platform", platform):
        return startup_plan.compute_plan_fingerprint(vllm_config, rank, world_size)


def test_fingerprint_stable_and_sensitive():
    base = _fingerprint()
    assert base == _fingerprint(), "same inputs must give the same fingerprint"
    assert base != _fingerprint(config_hash="different")
    assert base != _fingerprint(name="NVIDIA A100-SXM4-80GB")
    assert base != _fingerprint(total_mem=40 * GiB)
    assert base != _fingerprint(rank=1, world_size=2)


def test_save_load_round_trip(tmp_path):
    save_startup_plan(str(tmp_path), "deadbeef00000000", 50 * GiB, 78 * GiB)
    plan = load_startup_plan(str(tmp_path), "deadbeef00000000")
    assert plan is not None
    assert plan["kv_cache_memory_bytes"] == 50 * GiB
    assert plan["free_memory_baseline"] == 78 * GiB


def test_load_missing_or_corrupt_returns_none(tmp_path):
    assert load_startup_plan(str(tmp_path), "0000000000000000") is None

    bad = tmp_path / "startup_plan_1111111111111111.json"
    bad.write_text("{not json")
    assert load_startup_plan(str(tmp_path), "1111111111111111") is None

    wrong_fp = tmp_path / "startup_plan_2222222222222222.json"
    wrong_fp.write_text(json.dumps({"schema": 1, "fingerprint": "mismatch"}))
    assert load_startup_plan(str(tmp_path), "2222222222222222") is None


def test_free_memory_gate():
    plan = {"kv_cache_memory_bytes": 50 * GiB, "free_memory_baseline": 78 * GiB}
    # Enough free memory: apply.
    assert applicable_kv_cache_memory_bytes(plan, 78 * GiB) == 50 * GiB
    assert applicable_kv_cache_memory_bytes(plan, 79 * GiB) == 50 * GiB
    # Less free memory than when measured: refuse, re-profile.
    assert applicable_kv_cache_memory_bytes(plan, 77 * GiB) is None


def test_gate_rejects_malformed_plans():
    assert applicable_kv_cache_memory_bytes({}, 80 * GiB) is None
    assert (
        applicable_kv_cache_memory_bytes(
            {"kv_cache_memory_bytes": -1, "free_memory_baseline": 1}, 80 * GiB
        )
        is None
    )
    assert (
        applicable_kv_cache_memory_bytes(
            {"kv_cache_memory_bytes": "50", "free_memory_baseline": 1}, 80 * GiB
        )
        is None
    )


def test_maybe_apply_end_to_end(tmp_path):
    vllm_config = SimpleNamespace(compute_hash=lambda: "abc123")
    with patch.object(startup_plan, "current_platform", _mock_platform()):
        fp = startup_plan.compute_plan_fingerprint(vllm_config, 0, 1)
        save_startup_plan(str(tmp_path), fp, 50 * GiB, 78 * GiB)

        # Same fingerprint + enough memory: applied.
        assert (
            maybe_apply_startup_plan(str(tmp_path), vllm_config, 0, 1, 78 * GiB)
            == 50 * GiB
        )
        # Same fingerprint, less memory: refused.
        assert (
            maybe_apply_startup_plan(str(tmp_path), vllm_config, 0, 1, 60 * GiB) is None
        )
        # Different config: fingerprint miss.
        other = SimpleNamespace(compute_hash=lambda: "zzz999")
        assert maybe_apply_startup_plan(str(tmp_path), other, 0, 1, 78 * GiB) is None
