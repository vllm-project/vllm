# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

pytestmark = pytest.mark.skip_global_cleanup


def _draft_vocab_padding_config(unpadded_vocab_size: int):
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type="draft"),
        speculative_config=SimpleNamespace(
            allow_draft_model_vocab_padding=True,
            draft_model_unpadded_vocab_size=unpadded_vocab_size,
        ),
    )


def _new_vocab_layer(
    layer_cls: type[VocabParallelEmbedding],
    *,
    rank: int = 0,
    tp_size: int = 1,
    vocab_size: int = 20,
    hidden_size: int = 3,
    loaded_vocab_size: int | None = None,
):
    current_config = (
        None
        if loaded_vocab_size is None
        else _draft_vocab_padding_config(loaded_vocab_size)
    )
    with (
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_current_vllm_config_or_none",
            return_value=current_config,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=rank,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
    ):
        return layer_cls(vocab_size, hidden_size, padding_size=8)


@pytest.mark.parametrize("layer_cls", [VocabParallelEmbedding, ParallelLMHead])
def test_short_loaded_vocab_tensor_requires_opt_in(layer_cls):
    layer = _new_vocab_layer(layer_cls)
    loaded_weight = torch.empty(14, 3)

    with pytest.raises(AssertionError):
        layer.weight_loader(layer.weight, loaded_weight)


@pytest.mark.parametrize("layer_cls", [VocabParallelEmbedding, ParallelLMHead])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_short_loaded_vocab_tensor_is_zero_padded(layer_cls, tp_size):
    vocab_size = 20
    loaded_vocab_size = 14
    hidden_size = 3
    loaded_weight = torch.arange(
        loaded_vocab_size * hidden_size, dtype=torch.float32
    ).view(loaded_vocab_size, hidden_size)

    for rank in range(tp_size):
        layer = _new_vocab_layer(
            layer_cls,
            rank=rank,
            tp_size=tp_size,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            loaded_vocab_size=loaded_vocab_size,
        )

        layer.weight_loader(layer.weight, loaded_weight)

        shard = layer.shard_indices
        local_start = min(shard.org_vocab_start_index, loaded_vocab_size)
        local_end = min(shard.org_vocab_end_index, loaded_vocab_size)
        copied_rows = local_end - local_start
        expected = torch.zeros_like(layer.weight)
        if copied_rows > 0:
            expected[:copied_rows] = loaded_weight[local_start:local_end]

        torch.testing.assert_close(layer.weight, expected)


@pytest.mark.parametrize("bad_loaded_vocab_size", [13, 15])
def test_opt_in_loaded_vocab_size_must_match_recorded_size(bad_loaded_vocab_size):
    layer = _new_vocab_layer(
        VocabParallelEmbedding,
        loaded_vocab_size=14,
    )
    loaded_weight = torch.empty(bad_loaded_vocab_size, 3)

    with pytest.raises(ValueError, match="unpadded draft vocab size"):
        layer.weight_loader(layer.weight, loaded_weight)


def test_draft_vocab_padding_rejects_packed_vocab_weight():
    layer = _new_vocab_layer(VocabParallelEmbedding, loaded_vocab_size=14)
    layer.weight.packed_dim = 0

    with pytest.raises(ValueError, match="packed weights"):
        layer.weight_loader(layer.weight, torch.empty(14, 3))


def test_draft_vocab_padding_rejects_nonzero_output_dim():
    layer = _new_vocab_layer(VocabParallelEmbedding, loaded_vocab_size=14)
    param = torch.nn.Parameter(torch.empty(3, layer.num_embeddings_per_partition))
    param.output_dim = 1

    with pytest.raises(ValueError, match="output_dim=0"):
        layer.weight_loader(param, torch.empty(3, 14))


def test_full_logits_mask_padded_draft_vocab_rows():
    layer = _new_vocab_layer(
        VocabParallelEmbedding,
        vocab_size=6,
        hidden_size=1,
        loaded_vocab_size=4,
    )
    layer.weight.data.zero_()
    layer.weight.data[:4, 0] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    processor = LogitsProcessor(vocab_size=6)

    with (
        patch(
            "vllm.model_executor.layers.logits_processor.tensor_model_parallel_gather",
            side_effect=lambda logits: logits,
        ),
        patch(
            "vllm.model_executor.layers.logits_processor."
            "tensor_model_parallel_all_gather",
            side_effect=lambda logits: logits,
        ),
    ):
        logits = processor(layer, torch.ones(1, 1))

    assert logits is not None
    assert torch.isneginf(logits[0, 4:6]).all()
    torch.testing.assert_close(logits[0, :4], torch.tensor([-4.0, -3.0, -2.0, -1.0]))


def test_local_argmax_masks_padded_draft_vocab_rows():
    layer = _new_vocab_layer(
        VocabParallelEmbedding,
        vocab_size=6,
        hidden_size=1,
        loaded_vocab_size=4,
    )
    layer.weight.data.zero_()
    layer.weight.data[:4, 0] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    processor = LogitsProcessor(vocab_size=6)

    with patch(
        "vllm.model_executor.layers.logits_processor."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        top_tokens = processor.get_top_tokens(layer, torch.ones(1, 1))

    assert top_tokens.item() == 3
