# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checksum merging, comparison, and cross-replica consistency, without a
model or GPU."""

import asyncio
import json

import pytest
from fastapi import HTTPException

from vllm.config import ParallelConfig, VllmConfig
from vllm.entrypoints.serve.dev.rlhf.weight_checker import handle_weight_checker
from vllm.utils.weight_checksum import (
    combine_weight_checksums,
    compare_weight_checksum_reports,
    split_checksum_key,
)
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _RankZeroGroup:
    rank_in_group = 0


# Rank-qualified checksum entries, as the handler receives them from a worker.
_DP0_W_A = {"dp0:pp0:pcp0:tp0:ep0:w": "a"}
_DP1_W_A = {"dp1:pp0:pcp0:tp0:ep0:w": "a"}


@pytest.fixture
def single_rank_groups(monkeypatch):
    """Make tp/pp/pcp/ep look like one rank, so the prefix is DP-only."""
    for name in ("get_tp_group", "get_pp_group", "get_pcp_group"):
        monkeypatch.setattr(gpu_worker, name, _RankZeroGroup)
    return _RankZeroGroup


@pytest.mark.parametrize(
    "current, mismatches",
    [
        ({"rank0:w": "a", "rank1:w": "b"}, []),
        ({"rank0:w": "changed", "rank1:w": "b"}, ["rank0:w"]),
        ({"rank0:w": "a"}, ["rank1:w"]),
        ({"rank0:w": "a", "rank1:w": "b", "rank2:w": "c"}, ["rank2:w"]),
        ({}, ["rank0:w", "rank1:w"]),
    ],
)
def test_compare_detects_changed_missing_and_extra_tensors(current, mismatches):
    """The two-report form still behaves as a baseline diff."""
    baseline = {"rank0:w": "a", "rank1:w": "b"}
    match, reported, _ = compare_weight_checksum_reports([baseline, current])
    assert (match, reported) == (not mismatches, mismatches)


def test_compare_does_not_mutate_its_inputs():
    baseline = {"rank0:w": "a"}
    current = {"rank0:w": "b"}
    compare_weight_checksum_reports([baseline, current])
    assert baseline == {"rank0:w": "a"}
    assert current == {"rank0:w": "b"}


def test_merge_rejects_duplicate_rank_keys_even_when_values_match():
    with pytest.raises(RuntimeError, match="Duplicate weight checksum keys"):
        combine_weight_checksums([{"dp0:tp0:w": "a"}, {"dp0:tp0:w": "a"}])


def test_merge_preserves_identically_named_weights_on_distinct_ranks():
    shards = [{"dp0:tp0:w": "a"}, {"dp1:tp0:w": "b"}]
    assert combine_weight_checksums(shards) == {"dp0:tp0:w": "a", "dp1:tp0:w": "b"}


def test_dense_dp_ranks_get_distinct_key_prefixes(single_rank_groups):
    """Dense DP ranks must not collide, or merging rejects their results.

    reconfigure_for_independent_dp_rank() zeroes data_parallel_rank for dense
    models, which is why the prefix has to come from data_parallel_index.
    """

    def prefix_for(dp_rank: int) -> str:
        worker = object.__new__(Worker)
        config = VllmConfig()
        config.parallel_config = ParallelConfig(
            data_parallel_size=2, data_parallel_rank=dp_rank
        )
        worker.vllm_config = config
        # Dense: a DP rank is reconfigured into an independent engine.
        config.parallel_config.reconfigure_for_independent_dp_rank()
        assert config.parallel_config.data_parallel_rank == 0
        return Worker._weight_checksum_key_prefix(worker)

    first, second = prefix_for(0), prefix_for(1)
    assert first != second, f"dense DP ranks share a key prefix: {first}"


@pytest.mark.parametrize(
    "key, prefix, name",
    [
        ("dp0:pp0:pcp0:tp0:ep0:w", "dp0:pp0:pcp0:tp0:ep0:", "w"),
        (
            "dp1:pp2:pcp3:tp4:ep5:model.layers.0.w",
            "dp1:pp2:pcp3:tp4:ep5:",
            "model.layers.0.w",
        ),
        # Splits on the first five colons, so a colon in the name is kept.
        ("dp0:pp0:pcp0:tp0:ep0:a:b.c", "dp0:pp0:pcp0:tp0:ep0:", "a:b.c"),
        ("unqualified", "", "unqualified"),
    ],
)
def test_split_checksum_key_keeps_the_name_whole(key, prefix, name):
    assert split_checksum_key(key) == (prefix, name)


@pytest.mark.parametrize(
    "reports, mismatches",
    [
        # Identical replicas.
        ([_DP0_W_A, _DP0_W_A, _DP0_W_A], []),
        # Digests match, but the second report only reached one rank. The
        # uncovered key is not "equal", so it must not pass as consistent.
        (
            [
                {"dp0:pp0:pcp0:tp0:ep0:w": "a", "dp0:pp0:pcp0:tp0:ep0:v": "b"},
                _DP0_W_A,
            ],
            ["dp0:pp0:pcp0:tp0:ep0:v"],
        ),
        # Same rank prefix, different digest: the replicas disagree.
        (
            [_DP0_W_A, {"dp0:pp0:pcp0:tp0:ep0:w": "c"}],
            ["dp0:pp0:pcp0:tp0:ep0:w"],
        ),
        # The same tensor name on another rank is another shard, so it is
        # reported as uncovered rather than as a disagreement.
        (
            [_DP0_W_A, {"dp1:pp0:pcp0:tp0:ep0:w": "c"}],
            ["dp0:pp0:pcp0:tp0:ep0:w", "dp1:pp0:pcp0:tp0:ep0:w"],
        ),
        ([_DP0_W_A, {}], ["dp0:pp0:pcp0:tp0:ep0:w"]),
        ([{}, {}], []),
    ],
)
def test_compare_separates_disagreement_from_missing_ranks(reports, mismatches):
    """A key only some reports carry is not "equal", it is uncovered."""
    match, reported, _ = compare_weight_checksum_reports(reports)
    assert reported == mismatches
    assert match is not mismatches


@pytest.mark.parametrize(
    "reports, ranks",
    [
        ([_DP0_W_A, _DP0_W_A], ["dp0:pp0:pcp0:tp0:ep0:"]),
        (
            [_DP0_W_A, _DP1_W_A],
            ["dp0:pp0:pcp0:tp0:ep0:", "dp1:pp0:pcp0:tp0:ep0:"],
        ),
        (
            [_DP0_W_A, {"dp0:pp0:pcp0:tp1:ep0:w": "a"}],
            ["dp0:pp0:pcp0:tp0:ep0:", "dp0:pp0:pcp0:tp1:ep0:"],
        ),
        ([], []),
    ],
)
def test_compare_reports_the_rank_prefixes_it_covered(reports, ranks):
    """Coverage is what tells a caller the reports reached the right ranks."""
    _, _, covered = compare_weight_checksum_reports(reports)
    assert covered == ranks


def test_compare_does_not_mutate_its_reports():
    reports = [{"dp0:pp0:pcp0:tp0:ep0:w": "a"}, {"dp0:pp0:pcp0:tp0:ep0:w": "b"}]
    compare_weight_checksum_reports(reports)
    assert reports == [
        {"dp0:pp0:pcp0:tp0:ep0:w": "a"},
        {"dp0:pp0:pcp0:tp0:ep0:w": "b"},
    ]


class _FakeClient:
    """Stand in for an engine client that is reachable and not paused."""

    def __init__(self, checksums: dict[str, str] | None = None):
        self._checksums = _DP0_W_A if checksums is None else checksums

    async def is_paused(self) -> bool:
        return False

    async def compute_weight_checksums_all(self) -> list[dict[str, str]]:
        return [self._checksums]


def _run(body: dict, client: _FakeClient | None = None) -> dict:
    return asyncio.run(handle_weight_checker(body, client or _FakeClient()))


def test_handler_compare_matches_the_live_engine_against_the_baseline():
    response = _run({"action": "compare", "baseline": _DP0_W_A})
    assert response == {
        "match": True,
        "mismatches": [],
        "ranks": ["dp0:pp0:pcp0:tp0:ep0:"],
    }
    # The response has to survive the JSON round trip the endpoint performs.
    assert json.loads(json.dumps(response)) == response


def test_handler_compare_accepts_the_callers_other_reports():
    """With extra reports, compare holds every report to every other one."""
    response = _run(
        {"action": "compare", "baseline": _DP0_W_A, "checksums": [_DP0_W_A]}
    )
    assert response["match"] is True

    diverged = _run(
        {
            "action": "compare",
            "baseline": _DP0_W_A,
            "checksums": [{"dp0:pp0:pcp0:tp0:ep0:w": "b"}],
        }
    )
    assert diverged["match"] is False
    assert diverged["mismatches"] == ["dp0:pp0:pcp0:tp0:ep0:w"]


def test_handler_compare_flags_a_rank_the_baseline_does_not_cover():
    """An engine holding a rank the baseline lacks is a mismatch, not a pass."""
    response = _run({"action": "compare", "baseline": _DP1_W_A})
    assert response["match"] is False
    assert response["mismatches"] == [
        "dp0:pp0:pcp0:tp0:ep0:w",
        "dp1:pp0:pcp0:tp0:ep0:w",
    ]


@pytest.mark.parametrize("checksums", [{}, "report", [1], ["x"]])
def test_handler_compare_rejects_a_malformed_report_list(checksums):
    """A malformed list is rejected rather than silently ignored."""
    with pytest.raises(HTTPException) as excinfo:
        _run({"action": "compare", "baseline": _DP0_W_A, "checksums": checksums})
    assert excinfo.value.status_code == 400


def test_handler_compare_requires_a_baseline():
    with pytest.raises(HTTPException) as excinfo:
        _run({"action": "compare"})
    assert excinfo.value.status_code == 400


def test_handler_no_longer_has_a_separate_consistency_action():
    with pytest.raises(HTTPException) as excinfo:
        _run({"action": "consistency", "checksums": [_DP0_W_A]})
    assert excinfo.value.status_code == 400
