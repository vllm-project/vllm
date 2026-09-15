import torch
from unittest.mock import MagicMock

from vllm.model_executor.models.llama4 import Llama4ForCausalLM


def test_llama4_moe_weight_loading_contiguity():
    """Verify that non-contiguous CPU tensors become contiguous before weight_loader is called."""
    # Create non-contiguous CPU tensor via transpose
    raw_tensor = torch.randn(4, 32, 64, device="cpu")
    non_contiguous_weight = raw_tensor.transpose(-1, -2)
    assert not non_contiguous_weight.is_contiguous()

    received_tensors = []

    def mock_weight_loader(param, loaded_weight, param_name, shard_id=None, expert_id=None):
        received_tensors.append(loaded_weight)

    param = torch.nn.Parameter(torch.empty(32, 64, device="cpu"))
    param.weight_loader = mock_weight_loader

    params_dict = {
        "model.layers.0.feed_forward.experts.0.gate_up_proj.weight": param
    }

    expert_params_mapping = [
        ("model.layers.0.feed_forward.experts.0.gate_up_proj.", "experts.0.gate_up_proj.", 0, "w1")
    ]

    model_stub = MagicMock(spec=Llama4ForCausalLM)
    model_stub.layers = [MagicMock()]
    model_stub.layers[0].feed_forward.experts.expert_map = None

    loaded_params = set()
    result = Llama4ForCausalLM._load_moe_matrix(
        model_stub,
        name="model.layers.0.feed_forward.experts.0.gate_up_proj.weight",
        loaded_weight=non_contiguous_weight,
        params_dict=params_dict,
        expert_params_mapping=expert_params_mapping,
        loaded_params=loaded_params,
        fused=True,
    )

    assert result is True
    assert len(received_tensors) > 0
    for weight in received_tensors:
        assert weight.is_contiguous(), "Loaded weight passed to weight_loader must be contiguous"
