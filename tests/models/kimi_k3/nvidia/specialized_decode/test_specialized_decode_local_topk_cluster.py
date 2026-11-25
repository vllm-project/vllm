# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the DCP local top-N over a CTA cluster per row
(`cutedsl.dsa.indexer` LOCAL_CLUSTER, `IndexerLocalTopk(clusters=C)`).

- Policy (no GPU): C = 4 at DCP16 and 8 rows only, one CTA per row for every other
  DCP size and row count; a cluster needs the exchange push; the split threshold
  keeps rows below LOCAL_SPLIT_FROM vectors on CTA 0 and gives every share of a
  split row more than n vectors (each share's own bound needs n thread maxima).

The kernel's bits against one CTA per row: `test_specialized_decode_fp4_indexer.py`
(`test_local_topk_cluster_bitwise`).
"""

import pytest

from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import dcp_merge_local_topk
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import indexer


@pytest.mark.parametrize("world", [2, 4, 8, 16, 32])
def test_local_cluster_table(world):
    for rows in range(1, 65):
        want = 4 if (world, rows) == (16, 8) else 1
        assert indexer.local_cluster(world, rows) == want, (world, rows)


@pytest.mark.parametrize("world", [8, 16])
def test_local_topk_split_threshold(world):
    n = dcp_merge_local_topk(2048, world)
    assert indexer.IndexerLocalTopk(8, n, 512, world).clusters == 1
    for clusters in (2, 4, indexer.CLUSTER_MAX):
        local = indexer.IndexerLocalTopk(8, n, 512, world, clusters)
        assert local.clusters == clusters
        want = max(indexer.LOCAL_SPLIT_FROM, clusters * (n + clusters))
        assert local.split_from == want
        assert local.split_from // clusters > n  # every share: more than n vectors
    with pytest.raises(AssertionError):  # a cluster pushes to the exchange
        indexer.IndexerLocalTopk(8, n, 512, 0, 4)
    with pytest.raises(AssertionError):
        indexer.IndexerLocalTopk(8, n, 512, world, indexer.CLUSTER_MAX * 2)
