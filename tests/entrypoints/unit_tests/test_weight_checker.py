# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checksum merging and comparison, without a model or GPU."""

import pytest

from vllm.config import ParallelConfig, VllmConfig
from vllm.entrypoints.serve.dev.rlhf.weight_checker import compare_weight_checksums
from vllm.utils.weight_checksum import combine_weight_checksums
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _RankZeroGroup:
    rank_in_group = 0


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
