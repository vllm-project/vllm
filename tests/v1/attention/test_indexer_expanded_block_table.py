# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import AttentionGroup


@pytest.mark.parametrize(
    ("max_num_blocks", "block_size", "kernel_block_size", "expected_width"),
    [(1875, 64, 64, 1876), (235, 256, 64, 940)],
)
def test_get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int,
    expected_width: int,
):
    assert (
        get_block_table_width(max_num_blocks, block_size, kernel_block_size)
        == expected_width
    )


def test_attention_group_passes_final_width_to_builder():
    class WidthBuilder:
        requires_block_table_width = True

        def __init__(self, kv_cache_spec, *args, block_table_width: int):
            self.kv_cache_spec = kv_cache_spec
            self.block_table_width = block_table_width

    class WidthBackend:
        @staticmethod
        def get_builder_cls():
            return WidthBuilder

    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1200),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    group = AttentionGroup(WidthBackend, ["dummy"], kv_cache_spec, 0)

    group.create_metadata_builders(
        vllm_config, torch.device("cpu"), kernel_block_size=64
    )

    builder = group.get_metadata_builder()
    assert builder.block_table_width == 12
    assert builder.kv_cache_spec.block_size == 64
