# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 rejects IndexCache keys passed as top-level --hf-overrides."""

import pytest
import regex as re

from vllm.config import ModelConfig
from vllm.models.minimax_m3.amd.model import _should_skip_index_topk
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 AMD model"
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


def test_text_config_overrides_take_effect(checkpoint):
    text_config = ModelConfig(
        checkpoint,
        hf_overrides={"text_config": {"use_index_cache": True, "index_topk_freq": 4}},
    ).hf_text_config

    # Sparse layers 3..59; every 4th one recomputes the top-k.
    assert [i for i in range(60) if _should_skip_index_topk(text_config, i)] == sorted(
        set(range(3, 60)) - set(range(3, 60, 4))
    )
