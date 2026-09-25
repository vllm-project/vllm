# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for bulk 3D expert weight loading vs iterative unbind loading."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


def make_dummy_moe_config(
    num_experts: int = 1,
    num_local_experts: int | None = None,
    experts_per_token: int = 1,
    hidden_dim: int = 1,
    intermediate_size: int = 1,
    in_dtype: torch.dtype = torch.bfloat16,
    max_num_tokens: int = 512,
    activation: MoEActivation = MoEActivation.SILU,
    device: str = "cpu",
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts
        if num_local_experts is not None
        else num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=in_dtype,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_num_tokens,
    )


def _create_routed_experts_layer(
    num_experts: int,
    hidden_dim: int,
    intermediate_size: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int = 1,
    ep_rank: int = 0,
    is_fused_transposed: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> RoutedExperts:
    num_local_experts = num_experts // ep_size
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        in_dtype=dtype,
    )
    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    parallel_config.tp_size = tp_size
    parallel_config.tp_rank = tp_rank
    parallel_config.ep_size = ep_size
    parallel_config.ep_rank = ep_rank
    moe_config.moe_parallel_config = parallel_config
    moe_config.device = device
    moe_config.is_fused_checkpoint_transposed = is_fused_transposed
    moe_config.enable_expert_parallel = ep_size > 1
    moe_config.expert_placement_strategy = "linear"

    experts = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.layer_name = "model.layers.0.mlp.experts"
    experts.moe_config = moe_config
    experts.global_num_experts = num_experts
    experts.hidden_size = hidden_dim
    experts.is_fused_checkpoint_transposed = is_fused_transposed
    experts.lora_base_layer_prefix = ""
    experts.ckpt_gate_proj_name = "gate_proj"
    experts.ckpt_down_proj_name = "down_proj"
    experts.ckpt_up_proj_name = "up_proj"
    experts.quant_config = None

    class DummyMapManager:
        num_fused_shared_experts = 0
        def map_global_to_local(self, expert_id: int) -> int:
            start_exp = ep_rank * num_local_experts
            end_exp = (ep_rank + 1) * num_local_experts
            if start_exp <= expert_id < end_exp:
                return expert_id - start_exp
            return -1

    experts.expert_map_manager = DummyMapManager()
    with set_current_vllm_config(VllmConfig()):
        experts.quant_method = UnquantizedFusedMoEMethod(moe_config)

    # Allocate parameters
    intermediate_per_rank = intermediate_size // tp_size
    experts.w13_weight = torch.nn.Parameter(
        torch.zeros(num_local_experts, 2 * intermediate_per_rank, hidden_dim, device=device, dtype=dtype)
    )
    experts.w2_weight = torch.nn.Parameter(
        torch.zeros(num_local_experts, hidden_dim, intermediate_per_rank, device=device, dtype=dtype)
    )
    experts.w13_weight.weight_loader = experts.weight_loader
    experts.w2_weight.weight_loader = experts.weight_loader
    return experts


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("is_transposed", [False, True])
def test_bulk_3d_vs_unbind_equivalence(tp_rank: int, is_transposed: bool):
    """Verify that bulk 3D loading produces exact bitwise identical weights to unbind."""
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2

    torch.manual_seed(42 + tp_rank)
    if is_transposed:
        gate_up_weight = torch.randn(num_experts, hidden_dim, 2 * intermediate_size, dtype=torch.bfloat16)
        down_weight = torch.randn(num_experts, intermediate_size, hidden_dim, dtype=torch.bfloat16)
    else:
        gate_up_weight = torch.randn(num_experts, 2 * intermediate_size, hidden_dim, dtype=torch.bfloat16)
        down_weight = torch.randn(num_experts, hidden_dim, intermediate_size, dtype=torch.bfloat16)

    weights_dict = [
        ("gate_up_proj", gate_up_weight),
        ("down_proj", down_weight),
    ]

    # Instance A: uses patched bulk 3D fast-path
    layer_bulk = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        is_fused_transposed=is_transposed,
    )
    list(layer_bulk.load_weights(weights_dict))

    # Instance B: emulate fallback to iterative unbind
    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        is_fused_transposed=is_transposed,
    )

    for name, weight in weights_dict:
        if "gate_up_proj" in name:
            shard_id_list = [("w1", 0), ("w3", 1)]
            fused_weight = layer_unbind._orient_fused_weight(weight, is_transposed)
            for shard_id, chunk_idx in shard_id_list:
                experts_shard = fused_weight.chunk(2, dim=1)[chunk_idx]
                for exp_id, exp_tensor in enumerate(experts_shard.unbind()):
                    layer_unbind.w13_weight.weight_loader(
                        param=layer_unbind.w13_weight,
                        loaded_weight=exp_tensor,
                        weight_name="model.layers.0.mlp.experts.w13_weight",
                        shard_id=shard_id,
                        expert_id=exp_id,
                    )
        elif "down_proj" in name:
            fused_weight = layer_unbind._orient_fused_weight(weight, is_transposed)
            for exp_id, exp_tensor in enumerate(fused_weight.unbind()):
                layer_unbind.w2_weight.weight_loader(
                    param=layer_unbind.w2_weight,
                    loaded_weight=exp_tensor,
                    weight_name="model.layers.0.mlp.experts.w2_weight",
                    shard_id="w2",
                    expert_id=exp_id,
                )

    assert torch.equal(layer_bulk.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_bulk.w2_weight, layer_unbind.w2_weight)


@pytest.mark.parametrize("ep_rank", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_bulk_3d_with_linear_ep(ep_rank: int, tp_rank: int):
    """Verify that bulk 3D loading with linear Expert Parallelism produces bitwise identical weights."""
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2
    ep_size = 2

    torch.manual_seed(100 + ep_rank * 10 + tp_rank)
    gate_up_weight = torch.randn(num_experts, 2 * intermediate_size, hidden_dim, dtype=torch.bfloat16)
    down_weight = torch.randn(num_experts, hidden_dim, intermediate_size, dtype=torch.bfloat16)

    weights_dict = [
        ("gate_up_proj", gate_up_weight),
        ("down_proj", down_weight),
    ]

    layer_bulk = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    list(layer_bulk.load_weights(weights_dict))

    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    for name, weight in weights_dict:
        if "gate_up_proj" in name:
            shard_id_list = [("w1", 0), ("w3", 1)]
            fused_weight = layer_unbind._orient_fused_weight(weight, False)
            for shard_id, chunk_idx in shard_id_list:
                experts_shard = fused_weight.chunk(2, dim=1)[chunk_idx]
                for exp_id, exp_tensor in enumerate(experts_shard.unbind()):
                    layer_unbind.w13_weight.weight_loader(
                        param=layer_unbind.w13_weight,
                        loaded_weight=exp_tensor,
                        weight_name="model.layers.0.mlp.experts.w13_weight",
                        shard_id=shard_id,
                        expert_id=exp_id,
                    )
        elif "down_proj" in name:
            fused_weight = layer_unbind._orient_fused_weight(weight, False)
            for exp_id, exp_tensor in enumerate(fused_weight.unbind()):
                layer_unbind.w2_weight.weight_loader(
                    param=layer_unbind.w2_weight,
                    loaded_weight=exp_tensor,
                    weight_name="model.layers.0.mlp.experts.w2_weight",
                    shard_id="w2",
                    expert_id=exp_id,
                )

    assert torch.equal(layer_bulk.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_bulk.w2_weight, layer_unbind.w2_weight)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("arrival_order", ["sequential", "reverse", "interleaved"])
def test_online_2d_buffering_equivalence(tp_rank: int, arrival_order: str):
    """Verify online 2D-to-3D layer staging produces bitwise identical weights to unbind."""
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2

    torch.manual_seed(200 + tp_rank)
    gate_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    up_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    down_weights = [torch.randn(hidden_dim, intermediate_size, dtype=torch.bfloat16) for _ in range(num_experts)]

    weights_2d = []
    if arrival_order == "sequential":
        for i in range(num_experts):
            weights_2d.append((f"{i}.gate_proj.weight", gate_weights[i]))
            weights_2d.append((f"{i}.up_proj.weight", up_weights[i]))
            weights_2d.append((f"{i}.down_proj.weight", down_weights[i]))
    elif arrival_order == "reverse":
        for i in reversed(range(num_experts)):
            weights_2d.append((f"{i}.down_proj.weight", down_weights[i]))
            weights_2d.append((f"{i}.up_proj.weight", up_weights[i]))
            weights_2d.append((f"{i}.gate_proj.weight", gate_weights[i]))
    elif arrival_order == "interleaved":
        # All gates first, then all downs, then all ups
        for i in range(num_experts):
            weights_2d.append((f"{i}.gate_proj.weight", gate_weights[i]))
        for i in reversed(range(num_experts)):
            weights_2d.append((f"{i}.down_proj.weight", down_weights[i]))
        for i in range(num_experts):
            weights_2d.append((f"{i}.up_proj.weight", up_weights[i]))

    layer_online = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    loaded_names = list(layer_online.load_weights(weights_2d))
    assert len(loaded_names) == len(weights_2d)

    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    for i in range(num_experts):
        layer_unbind.w13_weight.weight_loader(
            param=layer_unbind.w13_weight,
            loaded_weight=gate_weights[i],
            weight_name="model.layers.0.mlp.experts.w13_weight",
            shard_id="w1",
            expert_id=i,
        )
        layer_unbind.w13_weight.weight_loader(
            param=layer_unbind.w13_weight,
            loaded_weight=up_weights[i],
            weight_name="model.layers.0.mlp.experts.w13_weight",
            shard_id="w3",
            expert_id=i,
        )
        layer_unbind.w2_weight.weight_loader(
            param=layer_unbind.w2_weight,
            loaded_weight=down_weights[i],
            weight_name="model.layers.0.mlp.experts.w2_weight",
            shard_id="w2",
            expert_id=i,
        )

    assert torch.equal(layer_online.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_online.w2_weight, layer_unbind.w2_weight)


@pytest.mark.parametrize("ep_rank", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_online_2d_buffering_with_linear_ep(ep_rank: int, tp_rank: int):
    """Verify online 2D buffering with linear EP filters non-local experts and matches unbind."""
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2
    ep_size = 2

    torch.manual_seed(300 + ep_rank * 10 + tp_rank)
    gate_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    up_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    down_weights = [torch.randn(hidden_dim, intermediate_size, dtype=torch.bfloat16) for _ in range(num_experts)]

    weights_2d = []
    for i in range(num_experts):
        weights_2d.append((f"{i}.gate_proj.weight", gate_weights[i]))
        weights_2d.append((f"{i}.up_proj.weight", up_weights[i]))
        weights_2d.append((f"{i}.down_proj.weight", down_weights[i]))

    layer_online = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    list(layer_online.load_weights(weights_2d))

    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    for i in range(num_experts):
        layer_unbind.w13_weight.weight_loader(
            param=layer_unbind.w13_weight,
            loaded_weight=gate_weights[i],
            weight_name="model.layers.0.mlp.experts.w13_weight",
            shard_id="w1",
            expert_id=i,
        )
        layer_unbind.w13_weight.weight_loader(
            param=layer_unbind.w13_weight,
            loaded_weight=up_weights[i],
            weight_name="model.layers.0.mlp.experts.w13_weight",
            shard_id="w3",
            expert_id=i,
        )
        layer_unbind.w2_weight.weight_loader(
            param=layer_unbind.w2_weight,
            loaded_weight=down_weights[i],
            weight_name="model.layers.0.mlp.experts.w2_weight",
            shard_id="w2",
            expert_id=i,
        )

    assert torch.equal(layer_online.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_online.w2_weight, layer_unbind.w2_weight)


def test_online_2d_buffering_partial_flush():
    """Verify incomplete/partial expert streams drain cleanly via residual flush without error."""
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2
    tp_rank = 0

    torch.manual_seed(400)
    # Only provide 3 out of 8 experts (partial stream)
    partial_experts = [1, 3, 5]
    weights_2d = []
    for i in partial_experts:
        weights_2d.append((f"{i}.gate_proj.weight", torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16)))
        weights_2d.append((f"{i}.up_proj.weight", torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16)))
        weights_2d.append((f"{i}.down_proj.weight", torch.randn(hidden_dim, intermediate_size, dtype=torch.bfloat16)))

    layer_online = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    loaded_names = list(layer_online.load_weights(weights_2d))
    # Each partial expert yields through residual drain (3 experts * 3 projections = 9 yields)
    assert len(loaded_names) == 9

    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    for name, weight in weights_2d:
        exp_id = int(name.split(".")[0])
        if "gate_proj" in name:
            layer_unbind.w13_weight.weight_loader(
                param=layer_unbind.w13_weight,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.w13_weight",
                shard_id="w1",
                expert_id=exp_id,
            )
        elif "up_proj" in name:
            layer_unbind.w13_weight.weight_loader(
                param=layer_unbind.w13_weight,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.w13_weight",
                shard_id="w3",
                expert_id=exp_id,
            )
        elif "down_proj" in name:
            layer_unbind.w2_weight.weight_loader(
                param=layer_unbind.w2_weight,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.w2_weight",
                shard_id="w2",
                expert_id=exp_id,
            )

    layer_unbind.flush_host_staging()
    assert torch.equal(layer_online.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_online.w2_weight, layer_unbind.w2_weight)


def test_online_2d_per_expert_fused_w13():
    """Verify online buffering with per-expert fused w13 (gate_up_proj per expert)."""
    num_experts = 4
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2
    tp_rank = 0

    torch.manual_seed(500)
    weights_2d = []
    for i in range(num_experts):
        weights_2d.append((f"{i}.gate_up_proj.weight", torch.randn(2 * intermediate_size, hidden_dim, dtype=torch.bfloat16)))
        weights_2d.append((f"{i}.down_proj.weight", torch.randn(hidden_dim, intermediate_size, dtype=torch.bfloat16)))

    layer_online = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    loaded_names = list(layer_online.load_weights(weights_2d))
    assert len(loaded_names) == 12

    layer_unbind = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    for name, weight in weights_2d:
        exp_id = int(name.split(".")[0])
        if "gate_up_proj" in name:
            chunks = weight.chunk(2, dim=0)
            layer_unbind.w13_weight.weight_loader(
                param=layer_unbind.w13_weight,
                loaded_weight=chunks[0],
                weight_name="model.layers.0.mlp.experts.w13_weight",
                shard_id="w1",
                expert_id=exp_id,
            )
            layer_unbind.w13_weight.weight_loader(
                param=layer_unbind.w13_weight,
                loaded_weight=chunks[1],
                weight_name="model.layers.0.mlp.experts.w13_weight",
                shard_id="w3",
                expert_id=exp_id,
            )
        elif "down_proj" in name:
            layer_unbind.w2_weight.weight_loader(
                param=layer_unbind.w2_weight,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.w2_weight",
                shard_id="w2",
                expert_id=exp_id,
            )

    layer_unbind.flush_host_staging()
    assert torch.equal(layer_online.w13_weight, layer_unbind.w13_weight)
    assert torch.equal(layer_online.w2_weight, layer_unbind.w2_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_online_2d_buffering_on_cuda_gpu():
    """Verify online host pinned staging transfers to CUDA GPU with exact bitwise match to CPU."""
    if not torch.cuda.is_available():
        return
    num_experts = 8
    hidden_dim = 128
    intermediate_size = 256
    tp_size = 2
    tp_rank = 0

    torch.manual_seed(42)
    gate_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    up_weights = [torch.randn(intermediate_size, hidden_dim, dtype=torch.bfloat16) for _ in range(num_experts)]
    down_weights = [torch.randn(hidden_dim, intermediate_size, dtype=torch.bfloat16) for _ in range(num_experts)]

    weights_2d = []
    for i in range(num_experts):
        weights_2d.append((f"{i}.gate_proj.weight", gate_weights[i]))
        weights_2d.append((f"{i}.up_proj.weight", up_weights[i]))
        weights_2d.append((f"{i}.down_proj.weight", down_weights[i]))

    layer_cuda = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        device="cuda",
    )
    list(layer_cuda.load_weights(weights_2d))

    layer_cpu = _create_routed_experts_layer(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        device="cpu",
    )
    list(layer_cpu.load_weights(weights_2d))

    assert torch.equal(layer_cuda.w13_weight.cpu(), layer_cpu.w13_weight)
    assert torch.equal(layer_cuda.w2_weight.cpu(), layer_cpu.w2_weight)


if __name__ == "__main__":
    for tp in (0, 1):
        for transposed in (False, True):
            test_bulk_3d_vs_unbind_equivalence(tp, transposed)
    for ep in (0, 1):
        for tp in (0, 1):
            test_bulk_3d_with_linear_ep(ep, tp)
    for tp in (0, 1):
        for order in ("sequential", "reverse", "interleaved"):
            test_online_2d_buffering_equivalence(tp, order)
    for ep in (0, 1):
        for tp in (0, 1):
            test_online_2d_buffering_with_linear_ep(ep, tp)
    test_online_2d_buffering_partial_flush()
    test_online_2d_per_expert_fused_w13()
    test_online_2d_buffering_on_cuda_gpu()
