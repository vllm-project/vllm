# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check sparse scoring and standard router checkpoint loading."""

import pytest
import torch

from vllm.model_executor.layers.hierarchical_lm_head import HierarchicalLMHead
from vllm.model_executor.models.utils import AutoWeightsLoader


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_dynamic_routing_matches_selected_dense_logits(device, dtype):
    head = HierarchicalLMHead(4, 8, 4, 2).to(device=device, dtype=dtype)
    with torch.no_grad():
        head.centroids.weight.copy_(torch.eye(4, device=device, dtype=dtype))
        head.token_ordering.copy_(torch.tensor([7, 2, 5, 0, 6, 1, 4, 3], device=device))
    weight = torch.arange(32, dtype=dtype, device=device).reshape(8, 4)
    for batch in (1, 3, 7):
        hidden = torch.tensor(
            [[1.0, 4.0, 2.0, 3.0]], device=device, dtype=dtype
        ).repeat(batch, 1)
        logits = head(hidden, weight)
        selected = torch.tensor([5, 0, 4, 3], device=device)
        reference = torch.full_like(logits, torch.finfo(logits.dtype).min)
        reference[:, selected] = (hidden @ weight.T)[:, selected]
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            head.get_top_tokens(hidden, weight), reference.argmax(-1)
        )
        if device == "cuda":
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = head.get_top_tokens(hidden, weight)
            graph.replay()
            torch.testing.assert_close(captured, reference.argmax(-1))


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_router_loads_through_auto_weights_loader(default_device):
    # Both models use masked_embedding; exercise the actual recursive loader,
    # including separate groups of checkpoint entries and its persistent buffer.
    model = torch.nn.Module()
    model.masked_embedding = HierarchicalLMHead(4, 8, 4, 2)
    model.other = torch.nn.Linear(4, 4, bias=False)
    ordering = torch.tensor([3, 7, 0, 5, 2, 6, 1, 4])
    weights = [
        ("masked_embedding.centroids.weight", torch.eye(4)),
        ("other.weight", torch.eye(4)),
        ("masked_embedding.token_ordering", ordering),
    ]
    with torch.device(default_device):
        loaded = AutoWeightsLoader(model).load_weights(iter(weights))
    torch.testing.assert_close(model.masked_embedding.centroids.weight, torch.eye(4))
    torch.testing.assert_close(model.masked_embedding.token_ordering, ordering)
    assert "masked_embedding.token_ordering" in loaded


@pytest.mark.parametrize("invalid", ["duplicate", "negative", "dtype", "shape"])
def test_router_checkpoint_rejects_invalid_ordering(invalid):
    head = HierarchicalLMHead(4, 8, 4, 2)
    ordering = torch.arange(8)
    if invalid == "duplicate":
        ordering[0] = 1
    elif invalid == "negative":
        ordering[0] = -1
    elif invalid == "dtype":
        ordering = ordering.float()
    else:
        ordering = ordering[:-1]
    with pytest.raises(ValueError, match="permutation|shape"):
        head.load_weights([("token_ordering", ordering)])


@pytest.mark.parametrize("top_k,expected_token", [(1, 5), (4, 7)])
def test_checkpoint_config_controls_selected_vocabulary(top_k, expected_token):
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        use_ordered_embeddings=True,
        num_centroids=4,
        centroid_intermediate_top_k=top_k,
    )
    head = HierarchicalLMHead.from_config(config, hidden_size=4, vocab_size=8)
    assert head is not None
    head.load_weights(
        [
            ("centroids.weight", torch.eye(4)),
            ("token_ordering", torch.tensor([7, 2, 5, 0, 6, 1, 4, 3])),
        ]
    )
    hidden = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    assert head.get_top_tokens(hidden, weight).item() == expected_token


@pytest.mark.parametrize(
    "num_centroids,top_k,error",
    [(4, 0, "top-k"), (4, 5, "top-k"), (3, 1, "divisible")],
)
def test_invalid_checkpoint_routing_config(num_centroids, top_k, error):
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        use_ordered_embeddings=True,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=top_k,
    )
    with pytest.raises(ValueError, match=error):
        HierarchicalLMHead.from_config(config, hidden_size=4, vocab_size=8)


def test_checkpoint_without_router_keeps_dense_head():
    from transformers import PreTrainedConfig

    assert (
        HierarchicalLMHead.from_config(PreTrainedConfig(), hidden_size=4, vocab_size=8)
        is None
    )
