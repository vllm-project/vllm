# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checksum merging and one-shot baseline state, without a model or GPU."""

import pytest

from vllm.entrypoints.serve.dev.rlhf.weight_checker import _WeightCheckerState
from vllm.v1.worker.utils import combine_weight_checksums

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


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
    state = _WeightCheckerState()
    baseline = {"rank0:w": "a", "rank1:w": "b"}
    assert state.store_if_absent(baseline)
    baseline["rank0:w"] = "caller mutation"
    assert not state.store_if_absent(current)
    assert state.compare(current) == (not mismatches, mismatches)
    assert not state.has_baseline()
    with pytest.raises(RuntimeError, match="No checksum baseline"):
        state.compare(current)
    assert state.store_if_absent(current)
    assert state.compare(current) == (True, [])


def test_merge_rejects_duplicate_rank_keys_even_when_values_match():
    with pytest.raises(RuntimeError, match="Duplicate weight checksum keys"):
        combine_weight_checksums([{"dp0:tp0:w": "a"}, {"dp0:tp0:w": "a"}])


def test_merge_preserves_identically_named_weights_on_distinct_ranks():
    shards = [{"dp0:tp0:w": "a"}, {"dp1:tp0:w": "b"}]
    assert combine_weight_checksums(shards) == {"dp0:tp0:w": "a", "dp1:tp0:w": "b"}
