# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 rejects IndexCache keys passed as top-level --hf-overrides."""

import pytest
import regex as re

from vllm.config import ModelConfig
from vllm.models.minimax_m3.amd.model import _should_skip_index_topk
from vllm.models.minimax_m3.nvidia.model import (
    _should_skip_index_topk as _should_skip_index_topk_nv,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() or current_platform.is_cuda()),
    reason="MiniMax-M3 AMD/NVIDIA models",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("minimax_m3")
    MiniMaxM3Config(
        architectures=["MiniMaxM3SparseForConditionalGeneration"]
    ).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize(
    ("hf_overrides", "expected_error"),
    [
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 2,
                "index_skip_topk_offset": 1,
            },
            (
                "top-level use_index_cache, index_topk_freq, index_skip_topk_offset "
                'would be ignored. Pass them as --hf-overrides \'{"text_config": '
                '{"use_index_cache": true, "index_topk_freq": 2, '
                '"index_skip_topk_offset": 1}}\'.'
            ),
        ),
        (
            {"use_index_cache": True, "text_config": {"index_topk_freq": 2}},
            (
                "top-level use_index_cache would be ignored. Pass them as "
                '--hf-overrides \'{"text_config": {"use_index_cache": true}}\'.'
            ),
        ),
    ],
    ids=["top_level", "mixed"],
)
def test_top_level_overrides_are_rejected(checkpoint, hf_overrides, expected_error):
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        ModelConfig(checkpoint, hf_overrides=hf_overrides)


@pytest.mark.parametrize(
    ("overrides", "expected_skip"),
    [
        # Sparse layers 3..59; every 4th one recomputes the top-k.
        (
            {"use_index_cache": True, "index_topk_freq": 4},
            set(range(3, 60)) - set(range(3, 60, 4)),
        ),
        # The offset counts sparse-layer ordinals (the AMD-path convention,
        # not raw layer ids), so the first three sparse layers all compute
        # before the every-4th cadence starts.
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 4,
                "index_skip_topk_offset": 2,
            },
            set(range(3, 60)) - {3, 4, 5} - set(range(9, 60, 4)),
        ),
    ],
    ids=["default_offset", "offset_2"],
)
def test_text_config_overrides_take_effect(checkpoint, overrides, expected_skip):
    text_config = ModelConfig(
        checkpoint, hf_overrides={"text_config": overrides}
    ).hf_text_config

    # The AMD and NVIDIA models must pick the same compute layers.
    expected = sorted(expected_skip)
    assert [i for i in range(60) if _should_skip_index_topk(text_config, i)] == expected
    assert [
        i for i in range(60) if _should_skip_index_topk_nv(text_config, i)
    ] == expected
