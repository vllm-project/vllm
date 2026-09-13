# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4_1.common.pipeline import (
    get_sharing_dependencies,
    validate_local_sharing,
)


def _config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
    )


@pytest.mark.parametrize(
    "ranges", [[(0, 40)], [(0, 20), (20, 40)], [(0, 8), (8, 14), (14, 20), (20, 40)]]
)
def test_group_aligned_pipeline_cuts_keep_all_sources_local(ranges):
    validate_local_sharing(get_sharing_dependencies(_config(), ranges))


def test_equal_pp4_reports_cross_stage_kv_index_and_candidates():
    dependencies = get_sharing_dependencies(
        _config(), [(0, 10), (10, 20), (20, 30), (30, 40)]
    )
    cross = {
        (d.kind, d.source_layer, d.consumer_layer)
        for d in dependencies
        if d.source_stage != d.consumer_stage
    }
    assert {("kv", 8, 10), ("index", 28, 30), ("candidate", 20, 32)} <= cross
    with pytest.raises(NotImplementedError, match="source layer 8.*stage 0.*stage 1"):
        validate_local_sharing(dependencies)


@pytest.mark.parametrize("values", [[8, 2], [2, 2], [0, 2], [-1, 2], [2, 40]])
def test_invalid_source_lists_fail_before_layer_construction(values):
    config = _config()
    config.kv_source_layer_ids = values
    with pytest.raises(ValueError, match="kv_source_layer_ids"):
        get_sharing_dependencies(config, [(0, 40)])


def test_compressed_consumer_requires_an_earlier_source():
    config = _config()
    config.kv_source_layer_ids = [8, 14, 20]
    with pytest.raises(ValueError, match="layer 2 has no preceding kv source"):
        get_sharing_dependencies(config, [(0, 40)])


def test_candidate_source_must_publish_indices():
    config = _config()
    config.candidate_source_layer_id = 21
    with pytest.raises(ValueError, match="candidate source must be an index source"):
        get_sharing_dependencies(config, [(0, 40)])
