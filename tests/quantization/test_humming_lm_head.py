# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for prepare_humming_layer on ParallelLMHead.

ParallelLMHead subclasses VocabParallelEmbedding, not LinearBase, so it has
none of LinearBase's sharding attributes. prepare_humming_layer previously
read layer.output_partition_sizes and layer.has_bias unconditionally, so any
checkpoint that quantizes lm_head through humming (e.g. a compressed-tensors
"re:.*lm_head$" FP8 target group, as used by third-party REAP/NVFP4 dumps of
large MoE models) crashed with AttributeError before a single token could be
generated. This mirrors the existing input_size_per_partition guard for the
same class of layer.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    prepare_humming_layer,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


@pytest.fixture
def dist_init():
    from tests.utils import ensure_current_vllm_config

    temp_file = tempfile.mkstemp()[1]
    with ensure_current_vllm_config():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield


def test_prepare_humming_layer_on_parallel_lm_head_without_bias(dist_init):
    layer = ParallelLMHead(
        num_embeddings=64,
        embedding_dim=8,
        bias=False,
        params_dtype=torch.float16,
    )
    assert not hasattr(layer, "output_partition_sizes")
    assert not hasattr(layer, "has_bias")

    captured = {}
    weight_schema = MagicMock()
    weight_schema.convert_humming.side_effect = (
        lambda tensors, shape_n_stacks, shape_k_stacks, param_dtype: (
            captured.update(
                shape_n_stacks=shape_n_stacks, shape_k_stacks=shape_k_stacks
            )
            or (MagicMock(), tensors)
        )
    )

    with (
        patch("vllm.utils.humming.BaseWeightSchema") as mock_base_schema,
        patch("vllm.utils.humming.HummingInputSchema"),
        patch("vllm.utils.humming.HummingMethod") as mock_method,
    ):
        mock_base_schema.from_config.return_value = weight_schema
        mock_method.prepare_layer_meta.side_effect = lambda **kwargs: captured.update(
            shape_n=kwargs["shape_n"], has_bias=kwargs["has_bias"]
        )
        mock_method.transform_humming_layer.return_value = None

        prepare_humming_layer(layer, quant_config={})

    assert captured["shape_n_stacks"] == [layer.num_embeddings_per_partition]
    assert captured["shape_n"] == layer.num_embeddings_per_partition
    assert captured["has_bias"] is False


def test_prepare_humming_layer_on_parallel_lm_head_with_bias(dist_init):
    layer = ParallelLMHead(
        num_embeddings=64,
        embedding_dim=8,
        bias=True,
        params_dtype=torch.float16,
    )
    assert layer.bias is not None

    captured = {}
    weight_schema = MagicMock()
    weight_schema.convert_humming.side_effect = (
        lambda tensors, shape_n_stacks, shape_k_stacks, param_dtype: (
            MagicMock(),
            tensors,
        )
    )

    with (
        patch("vllm.utils.humming.BaseWeightSchema") as mock_base_schema,
        patch("vllm.utils.humming.HummingInputSchema"),
        patch("vllm.utils.humming.HummingMethod") as mock_method,
    ):
        mock_base_schema.from_config.return_value = weight_schema
        mock_method.prepare_layer_meta.side_effect = lambda **kwargs: captured.update(
            has_bias=kwargs["has_bias"]
        )
        mock_method.transform_humming_layer.return_value = None

        prepare_humming_layer(layer, quant_config={})

    assert captured["has_bias"] is True
