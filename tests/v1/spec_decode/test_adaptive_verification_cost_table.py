# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode import (
    adaptive_verification_cost_table as cost_table,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    build_cost_tables_from_curves,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

DUMP = "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP"
LOAD = "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD"

PROFILED = {"draft": [(1, 0.5), (8, 0.9)], "verify": [(1, 2.0), (64, 3.0)]}
STORED = {"draft": [(1, 0.4), (8, 0.8)], "verify": [(1, 1.5), (64, 2.5)]}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(DUMP, raising=False)
    monkeypatch.delenv(LOAD, raising=False)


@pytest.fixture
def warnings(monkeypatch):
    """The warnings the cost-table helpers log (the vLLM logger does not propagate)."""
    logged: list[str] = []
    monkeypatch.setattr(
        cost_table.logger,
        "warning",
        lambda msg, *args, **_: logged.append(msg % args),
    )
    return logged


def make_manager(
    rank: int = 0, cudagraph_limit: int = 64
) -> AdaptiveVerificationManager:
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = 3
    manager.req_states = SimpleNamespace(max_num_reqs=8, max_num_batched_tokens=128)
    manager._cudagraph_limit = cudagraph_limit
    manager.cost_tables = None
    return manager


@pytest.fixture
def tp_group(monkeypatch):
    """A TP group that returns what it is given: rank 0, or another rank on request."""
    group = SimpleNamespace(rank_in_group=0, broadcast_object=lambda obj, src=0: obj)
    monkeypatch.setattr(adaptive_module, "get_tp_group", lambda: group)
    return group


@pytest.fixture
def fixed_fingerprint(monkeypatch):
    """A constant fingerprint, so the tests do not depend on this machine's GPU."""
    monkeypatch.setattr(
        cost_table, "cost_table_fingerprint", lambda manager: {"cudagraph_limit": 64}
    )


def test_round_trip(tmp_path):
    path = str(tmp_path / "curves.json")
    fingerprint = {"gpu_name": "H100", "tp_size": 4}
    cost_table.dump_cost_curves(path, fingerprint, PROFILED)

    loaded = cost_table.load_cost_curves(path, fingerprint, ["draft", "verify"])

    assert loaded == PROFILED
    # The stored document carries the fingerprint and its hash for humans to compare.
    document = json.loads(Path(path).read_text())
    assert document["fingerprint"] == fingerprint
    assert document["fingerprint_hash"]
    # An atomic write leaves no temporary file behind.
    assert [p.name for p in tmp_path.iterdir()] == ["curves.json"]


def test_mismatching_fingerprint_keeps_profiled_curves(tmp_path, warnings):
    path = str(tmp_path / "curves.json")
    cost_table.dump_cost_curves(path, {"tp_size": 4}, PROFILED)

    assert (
        cost_table.load_cost_curves(path, {"tp_size": 8}, ["draft", "verify"]) is None
    )
    assert len(warnings) == 1
    assert "profiled for another configuration" in warnings[0]
    assert "tp_size" in warnings[0]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"format": 99, "fingerprint": {}, "curves": PROFILED}),
        json.dumps({"format": 1, "fingerprint": {}, "curves": {"draft": [[1, 1.0]]}}),
        json.dumps(
            {
                "format": 1,
                "fingerprint": {},
                "curves": {"draft": [[1, 1.0]], "verify": [[1, float("nan")]]},
            }
        ),
        json.dumps(
            {
                "format": 1,
                "fingerprint": {},
                "curves": {"draft": [], "verify": [[1, 1.0]]},
            }
        ),
    ],
    ids=["corrupt", "format", "curve-names", "non-finite", "empty-curve"],
)
def test_unusable_file_keeps_profiled_curves(tmp_path, warnings, content):
    path = tmp_path / "curves.json"
    path.write_text(content)

    assert cost_table.load_cost_curves(str(path), {}, ["draft", "verify"]) is None
    assert len(warnings) == 1
    assert "could not be used" in warnings[0]


def test_missing_file_keeps_profiled_curves(tmp_path, warnings):
    path = str(tmp_path / "absent.json")

    assert cost_table.load_cost_curves(path, {}, ["draft", "verify"]) is None
    assert len(warnings) == 1


def test_both_paths_set_fails_at_construction(monkeypatch):
    monkeypatch.setenv(DUMP, "/tmp/a.json")
    monkeypatch.setenv(LOAD, "/tmp/b.json")

    # The check runs before the constructor touches any argument.
    with pytest.raises(RuntimeError, match="both set"):
        AdaptiveVerificationManager(None, None, 1, 1)  # type: ignore[arg-type]


def test_unset_leaves_curves_untouched(monkeypatch, tp_group):
    def fail(*_args, **_kwargs):
        raise AssertionError("the cost-table helpers must not run when unset")

    monkeypatch.setattr(adaptive_module, "dump_or_load_cost_curves", fail)
    manager = make_manager()

    manager.set_cost_curves(PROFILED["draft"], PROFILED["verify"])

    expected = build_cost_tables_from_curves(
        PROFILED["draft"], PROFILED["verify"], 8, 128, 64
    )
    assert manager.cost_tables is not None
    for table, reference in zip(manager.cost_tables, expected):
        np.testing.assert_array_equal(table, reference)


def test_dump_then_load_pins_the_stored_curves(
    monkeypatch, tmp_path, tp_group, fixed_fingerprint
):
    path = str(tmp_path / "curves.json")

    monkeypatch.setenv(DUMP, path)
    first = make_manager()
    first.set_cost_curves(PROFILED["draft"], PROFILED["verify"])
    # Dumping does not change what the manager uses.
    reference = build_cost_tables_from_curves(
        PROFILED["draft"], PROFILED["verify"], 8, 128, 64
    )
    assert first.cost_tables is not None
    np.testing.assert_array_equal(first.cost_tables[1], reference[1])

    monkeypatch.delenv(DUMP)
    monkeypatch.setenv(LOAD, path)
    second = make_manager()
    # A different profile on this boot is replaced by the stored one.
    second.set_cost_curves(STORED["draft"], STORED["verify"])
    pinned = build_cost_tables_from_curves(
        PROFILED["draft"], PROFILED["verify"], 8, 128, 64
    )
    assert second.cost_tables is not None
    for table, reference in zip(second.cost_tables, pinned):
        np.testing.assert_array_equal(table, reference)


def test_mismatch_on_load_uses_this_boots_profile(
    monkeypatch, tmp_path, tp_group, warnings
):
    path = str(tmp_path / "curves.json")
    monkeypatch.setattr(
        cost_table, "cost_table_fingerprint", lambda manager: {"cudagraph_limit": 64}
    )
    cost_table.dump_cost_curves(path, {"cudagraph_limit": 32}, STORED)
    monkeypatch.setenv(LOAD, path)

    manager = make_manager()
    manager.set_cost_curves(PROFILED["draft"], PROFILED["verify"])

    expected = build_cost_tables_from_curves(
        PROFILED["draft"], PROFILED["verify"], 8, 128, 64
    )
    assert manager.cost_tables is not None
    np.testing.assert_array_equal(manager.cost_tables[1], expected[1])
    assert any("another configuration" in w for w in warnings)


def test_only_rank_zero_reads_the_file(
    monkeypatch, tmp_path, tp_group, fixed_fingerprint
):
    path = str(tmp_path / "curves.json")
    cost_table.dump_cost_curves(path, {"cudagraph_limit": 64}, STORED)
    monkeypatch.setenv(LOAD, path)
    tp_group.rank_in_group = 1

    def fail(*_args, **_kwargs):
        raise AssertionError("only rank 0 may read the stored curves")

    monkeypatch.setattr(adaptive_module, "dump_or_load_cost_curves", fail)
    manager = make_manager()

    # Rank 1 hands its own curves to the broadcast, which overwrites them.
    manager.set_cost_curves(PROFILED["draft"], PROFILED["verify"])

    assert manager.cost_tables is not None


def test_dump_failure_does_not_stop_startup(monkeypatch, tmp_path, tp_group, warnings):
    monkeypatch.setattr(
        cost_table, "cost_table_fingerprint", lambda manager: {"cudagraph_limit": 64}
    )
    monkeypatch.setenv(DUMP, str(tmp_path / "missing-dir" / "curves.json"))
    manager = make_manager()

    manager.set_cost_curves(PROFILED["draft"], PROFILED["verify"])

    assert manager.cost_tables is not None
    assert any("could not be written" in w for w in warnings)


def test_fingerprint_changes_with_the_profiled_shapes():
    base = cost_table.cost_table_fingerprint(make_manager())

    assert cost_table.cost_table_fingerprint(make_manager()) == base
    assert cost_table.cost_table_fingerprint(make_manager(cudagraph_limit=128)) != base
    # It survives a JSON round trip, which is how a stored fingerprint is compared.
    assert json.loads(json.dumps(base)) == base
