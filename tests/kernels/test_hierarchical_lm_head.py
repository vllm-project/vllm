# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check routing, masking, dynamic scoring, and training sidecar compatibility."""

import pytest
import torch

from vllm.model_executor.layers.hierarchical_lm_head import (
    HierarchicalLMHead,
    _sorted_expert_assignment,
)


@pytest.mark.parametrize(
    "device,backend", [("cpu", "compiled"), ("cuda", "compiled"), ("cuda", "moe")]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_dynamic_routing_matches_selected_dense_logits(device, backend, dtype):
    head = HierarchicalLMHead(4, 8, 4, 2, backend=backend).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        head.centroids.weight.copy_(torch.eye(4, device=device, dtype=dtype))
        head.token_ordering.copy_(torch.tensor([7, 2, 5, 0, 6, 1, 4, 3], device=device))
    weight = torch.arange(32, dtype=dtype, device=device).reshape(8, 4)
    head.prepare_weights(weight)
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
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = head.get_top_tokens(hidden, weight)
            graph.replay()
            torch.testing.assert_close(captured, reference.argmax(-1))


def test_training_sidecar_loads_balanced_assignment(tmp_path):
    head = HierarchicalLMHead(4, 8, 4, 2)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": {
                "cluster_signature": torch.eye(4),
                "token_to_cluster": torch.tensor([1, 3, 2, 0, 3, 1, 2, 0]),
            }
        },
        path,
    )
    head.load_sidecar(str(path))
    torch.testing.assert_close(head.centroids.weight, torch.eye(4))
    torch.testing.assert_close(
        head.token_ordering, torch.tensor([3, 7, 0, 5, 2, 6, 1, 4])
    )


@pytest.mark.parametrize("default_device", ["meta", "cuda"])
def test_sidecar_validation_ignores_default_device(tmp_path, default_device):
    """CPU checkpoint validation must not allocate on the model's default device."""
    head = HierarchicalLMHead(4, 8, 4, 2)
    centroids = torch.eye(4)
    ordering = torch.arange(7, -1, -1)
    path = tmp_path / "router.pt"
    torch.save({"centroids.weight": centroids, "token_ordering": ordering}, path)
    with torch.device(default_device):
        head.load_sidecar(str(path))
    torch.testing.assert_close(head.centroids.weight, centroids)
    torch.testing.assert_close(head.token_ordering, ordering)


def test_moe_preparation_preserves_order_and_refills_in_place():
    head = HierarchicalLMHead(4, 8, 4, 2, backend="moe")
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    with torch.inference_mode():
        head.prepare_weights(weight)
    address = head.centroid_weight.data_ptr()
    head.token_ordering.copy_(torch.arange(7, -1, -1))
    head.prepare_weights(weight + 1)
    assert head.centroid_weight.data_ptr() == address
    torch.testing.assert_close(head.centroid_weight.flatten(0, 1), (weight + 1).flip(0))
    assert "centroid_weight" not in head.state_dict()


@pytest.mark.parametrize("routing", ["auto", "unsorted", "sorted"])
@torch.inference_mode()
def test_moe_large_cluster_count_matches_dense_selected_logits(routing):
    # Cover all routing policies at C4096 below the auto grouping threshold.
    head = HierarchicalLMHead(
        4, 8192, 4096, 2, backend="moe", moe_routing=routing
    ).cuda()
    head.centroids.weight.zero_()
    head.centroids.weight[:, 0].copy_(torch.arange(4096, device="cuda"))
    # Small integers make the dot products exact even with TF32 input precision.
    weight = (
        torch.arange(32768, device="cuda", dtype=torch.float32)
        .remainder_(128)
        .reshape(8192, 4)
    )
    head.prepare_weights(weight)
    hidden = torch.tensor([[1.0, 4.0, 2.0, 3.0]], device="cuda").repeat(3, 1)
    reference = torch.full((3, 8192), torch.finfo(torch.float32).min, device="cuda")
    reference[:, -4:] = (hidden @ weight.T)[:, -4:]
    torch.testing.assert_close(head(hidden, weight), reference, atol=0, rtol=0)
    torch.testing.assert_close(
        head.get_top_tokens(hidden, weight), reference.argmax(-1)
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_logits = head(hidden, weight)
    hidden[:, 0].fill_(-1)
    graph.replay()
    reference.fill_(torch.finfo(torch.float32).min)
    reference[:, :4] = (hidden @ weight.T)[:, :4]
    torch.testing.assert_close(captured_logits, reference, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_experts", [4, 4096])
@pytest.mark.parametrize("repeats", [0, 1, 9])
def test_sorted_assignment_groups_and_pads_original_route_ids(
    device, num_experts, repeats
):
    routes = torch.tensor(
        [[num_experts - 1, 1], [1, 0], [num_experts - 1, 0]],
        device=device,
        dtype=torch.int32,
    ).repeat(repeats, 1)

    def assign():
        return _sorted_expert_assignment(routes, 4, num_experts)

    def check(result):
        sorted_ids, experts, total = result
        reference_ids, reference_experts = [], []
        flat = routes.flatten().tolist()
        for expert in range(num_experts):
            ids = [i for i, cluster in enumerate(flat) if cluster == expert]
            padded = ((len(ids) + 3) // 4) * 4
            reference_ids.extend(ids + [len(flat)] * (padded - len(ids)))
            reference_experts.extend([expert] * (padded // 4))
        assert total.item() == len(reference_ids)
        assert sorted_ids[: len(reference_ids)].tolist() == reference_ids
        assert experts[: len(reference_experts)].tolist() == reference_experts
        assert torch.all(sorted_ids[len(reference_ids) :] == len(flat))

    check(assign())
    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = assign()
        routes.copy_((routes + 1) % num_experts)
        graph.replay()
        check(captured)
