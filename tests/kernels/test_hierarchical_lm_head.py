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
    model.masked_embedding.validate_loaded_weights(loaded)
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


def test_router_checkpoint_requires_ordering_buffer():
    head = HierarchicalLMHead(4, 8, 4, 2)
    loaded = head.load_weights([("centroids.weight", torch.eye(4))])
    with pytest.raises(ValueError, match="token_ordering"):
        head.validate_loaded_weights({"masked_embedding." + name for name in loaded})


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_sidecar_router_infers_clusters_and_preserves_model_config(tmp_path, suffix):
    from safetensors.torch import save_file
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(hidden_size=4, vocab_size=8)
    path = tmp_path / ("router" + suffix)
    state = {
        "masked_embedding.centroids.weight": torch.eye(4),
        "masked_embedding.token_ordering": torch.arange(7, -1, -1),
    }
    if suffix == ".pt":
        torch.save({**state, "metadata": {"step": 100}}, path)
    else:
        save_file(state, path)
    before = config.to_dict()
    head = HierarchicalLMHead.from_config(
        config, hidden_size=4, vocab_size=8, router_path=str(path), top_k=2
    )
    assert head is not None
    loaded = head.load_sidecar_weights()
    head.validate_loaded_weights(loaded)
    assert head.num_centroids == 4 and head.num_selected == 4
    assert config.to_dict() == before
    torch.testing.assert_close(head.centroids.weight, torch.eye(4))
    torch.testing.assert_close(head.token_ordering, torch.arange(7, -1, -1))
    with pytest.raises(ValueError, match="explicit top-k"):
        HierarchicalLMHead.from_config(
            config, hidden_size=4, vocab_size=8, router_path=str(path)
        )
    with pytest.raises(ValueError, match="hidden size"):
        HierarchicalLMHead.from_config(
            config, hidden_size=5, vocab_size=8, router_path=str(path), top_k=2
        )
    with pytest.raises(ValueError, match="top-k"):
        HierarchicalLMHead.from_config(
            config, hidden_size=4, vocab_size=8, router_path=str(path), top_k=5
        )


def test_embedded_router_top_k_override_and_dense_default():
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        use_ordered_embeddings=True, num_centroids=4, centroid_intermediate_top_k=2
    )
    default = HierarchicalLMHead.from_config(config, hidden_size=4, vocab_size=8)
    override = HierarchicalLMHead.from_config(
        config, hidden_size=4, vocab_size=8, top_k=3
    )
    assert default is not None and override is not None
    assert default.centroid_intermediate_top_k == 2
    assert override.centroid_intermediate_top_k == 3
    assert config.centroid_intermediate_top_k == 2
    dense = PreTrainedConfig()
    assert HierarchicalLMHead.from_config(dense, hidden_size=4, vocab_size=8) is None
    with pytest.raises(ValueError, match="requires a hierarchical router"):
        HierarchicalLMHead.from_config(dense, hidden_size=4, vocab_size=8, top_k=2)
