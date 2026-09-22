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
    are_weight_checksums_consistent,
    combine_weight_checksums,
    compare_weight_checksums,
    split_checksum_key,
)
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _RankZeroGroup:
    rank_in_group = 0


# A rank-qualified checksum entry, as the handler receives it from a worker.
_DP0_W_A = {"dp0:pp0:pcp0:tp0:ep0:w": "a"}


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
    baseline = {"rank0:w": "a", "rank1:w": "b"}
    assert compare_weight_checksums(baseline, current) == (not mismatches, mismatches)


def test_compare_does_not_mutate_its_inputs():
    baseline = {"rank0:w": "a"}
    current = {"rank0:w": "b"}
    compare_weight_checksums(baseline, current)
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
def test_consistency_separates_disagreement_from_missing_ranks(reports, mismatches):
    consistent, reported, _ = are_weight_checksums_consistent(reports)
    assert reported == mismatches
    assert consistent is not mismatches


@pytest.mark.parametrize(
    "reports, ranks",
    [
        ([_DP0_W_A, _DP0_W_A], ["dp0:pp0:pcp0:tp0:ep0:"]),
        (
            [_DP0_W_A, {"dp1:pp0:pcp0:tp0:ep0:w": "a"}],
            ["dp0:pp0:pcp0:tp0:ep0:", "dp1:pp0:pcp0:tp0:ep0:"],
        ),
        (
            [_DP0_W_A, {"dp0:pp0:pcp0:tp1:ep0:w": "a"}],
            ["dp0:pp0:pcp0:tp0:ep0:", "dp0:pp0:pcp0:tp1:ep0:"],
        ),
        ([], []),
    ],
)
def test_consistency_reports_the_rank_prefixes_it_covered(reports, ranks):
    _, _, covered = are_weight_checksums_consistent(reports)
    assert covered == ranks


def test_consistency_does_not_mutate_its_inputs():
    reports = [{"dp0:pp0:pcp0:tp0:ep0:w": "a"}, {"dp0:pp0:pcp0:tp0:ep0:w": "b"}]
    are_weight_checksums_consistent(reports)
    assert reports == [
        {"dp0:pp0:pcp0:tp0:ep0:w": "a"},
        {"dp0:pp0:pcp0:tp0:ep0:w": "b"},
    ]


class _AwakeClient:
    """Stand in for an engine client that is reachable and not paused."""

    async def is_paused(self) -> bool:
        return False


def _run(body: dict) -> dict:
    return asyncio.run(handle_weight_checker(body, _AwakeClient()))


def test_handler_consistency_reports_agreeing_replicas():
    checksums = {"dp0:pp0:pcp0:tp0:ep0:w": "a"}
    response = _run({"action": "consistency", "checksums": [checksums, checksums]})
    assert response == {
        "consistent": True,
        "mismatches": [],
        "reports": 2,
        "ranks": ["dp0:pp0:pcp0:tp0:ep0:"],
    }
    # The response has to survive the JSON round trip the endpoint performs.
    assert json.loads(json.dumps(response)) == response


def test_handler_consistency_flags_a_diverged_replica():
    response = _run(
        {
            "action": "consistency",
            "checksums": [{"dp0:pp0:pcp0:tp0:ep0:w": "a"}] * 2
            + [{"dp0:pp0:pcp0:tp0:ep0:w": "b"}],
        }
    )
    assert response["consistent"] is False
    assert response["mismatches"] == ["dp0:pp0:pcp0:tp0:ep0:w"]
    assert response["reports"] == 3


@pytest.mark.parametrize("checksums", [None, {}, "report", [1], ["x"]])
def test_handler_consistency_requires_reports(checksums):
    with pytest.raises(HTTPException) as excinfo:
        _run({"action": "consistency", "checksums": checksums})
    assert excinfo.value.status_code == 400
