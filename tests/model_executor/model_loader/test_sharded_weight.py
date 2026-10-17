# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    get_layerwise_info,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.sharded_weight import (
    ShardedWeightRequest,
    ShardedWeightSpec,
    ShardedWeightTarget,
    resolve_sharded_weight_target,
)


def _target(
    *,
    semantic_id: str = "test.weight.v1",
    local_shape: tuple[int, ...] = (2, 3),
    consume: Any = None,
) -> ShardedWeightTarget:
    if consume is None:
        consume = lambda _: True
    return ShardedWeightTarget(
        spec=ShardedWeightSpec(
            semantic_id=semantic_id,
            dtype=torch.float16,
            shard_dim=0,
            local_shape=local_shape,
            shard_index=0,
            num_shards=2,
        ),
        retention_key=semantic_id,
        consume=consume,
    )


class _Provider(torch.nn.Module):
    def __init__(self, target: ShardedWeightTarget | None):
        super().__init__()
        self.target = target
        self.calls: list[str] = []

    def resolve_sharded_weight_target(
        self,
        relative_name: str,
        request: ShardedWeightRequest,
    ) -> ShardedWeightTarget | None:
        self.calls.append(relative_name)
        return self.target


def test_resolver_uses_nearest_provider_and_validates_before_loading():
    consumed: list[torch.Tensor] = []

    def consume(weight: torch.Tensor) -> bool:
        consumed.append(weight)
        return True

    root = _Provider(_target(semantic_id="root"))
    child_target = _target(consume=consume)
    root.child = _Provider(child_target)
    request = ShardedWeightRequest(
        name="child.weight",
        dtype=torch.float16,
        global_shape=(4, 3),
    )

    resolved = resolve_sharded_weight_target(root, request)

    assert resolved is child_target
    assert root.child.calls == ["weight"]
    assert root.calls == []
    with pytest.raises(ValueError, match="local shard shape"):
        resolved.load(torch.empty((1, 3), dtype=torch.float16))
    assert resolved.load(torch.ones((2, 3), dtype=torch.float16))
    assert len(consumed) == 1


def test_resolver_rejects_invalid_targets_and_supports_aliases():
    request = ShardedWeightRequest(
        name="block.weight",
        dtype=torch.float16,
        global_shape=(4, 3),
    )
    invalid = torch.nn.Module()
    invalid.block = _Provider(_target(local_shape=(3, 3)))
    with pytest.raises(ValueError, match="incompatible local shape"):
        resolve_sharded_weight_target(invalid, request)

    target = _target()
    no_retention = torch.nn.Module()
    no_retention.block = _Provider(
        ShardedWeightTarget(
            spec=target.spec,
            retention_key=None,
            consume=target.consume,
        )
    )
    with pytest.raises(ValueError, match="has no retention key"):
        resolve_sharded_weight_target(no_retention, request)

    aliased = _Provider(_target(semantic_id="alias"))
    aliases = torch.nn.Module()
    aliases.canonical = aliased
    aliases.block = aliased
    assert resolve_sharded_weight_target(aliases, request) is aliased.target
    assert aliased.calls == ["weight"]

    with pytest.raises(ValueError, match="retention_group_size"):
        ShardedWeightTarget(
            spec=target.spec,
            retention_key="invalid",
            consume=lambda _: True,
            retention_group_size=0,
        )


class _LinearExpertMap:
    placement_strategy = "linear"
    num_fused_shared_experts = 0

    def __init__(self, local_ids: tuple[int, ...], num_experts: int):
        self._local_ids = local_ids
        self._local_by_global = {
            global_id: local_id for local_id, global_id in enumerate(local_ids)
        }
        self.global_num_experts = num_experts

    def get_local_expert_ids(self) -> list[int]:
        return list(self._local_ids)

    def map_global_to_local(self, global_id: int) -> int:
        return self._local_by_global.get(global_id, -1)


def _make_routed_experts(
    *,
    dtype: torch.dtype = torch.bfloat16,
    gated: bool = True,
    projections: tuple[str, str, str | None] | None = None,
) -> RoutedExperts:
    num_experts = 4
    hidden_size = 4
    intermediate_size = 3
    ep_size = 2
    ep_rank = 1
    experts_per_rank = num_experts // ep_size
    local_ids = tuple(
        range(
            ep_rank * experts_per_rank,
            (ep_rank + 1) * experts_per_rank,
        )
    )
    parallel = SimpleNamespace(
        tp_size=1,
        ep_size=ep_size,
        ep_rank=ep_rank,
        use_ep=True,
        enable_eplb=False,
    )
    w13_num_shards = 2 if gated else 1
    config = SimpleNamespace(
        num_experts=num_experts,
        num_logical_experts=num_experts,
        num_local_experts=experts_per_rank,
        hidden_dim=hidden_size,
        hidden_dim_unpadded=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        intermediate_size_per_partition_unpadded=intermediate_size,
        moe_parallel_config=parallel,
        tp_rank=0,
        tp_shard_with_padding=False,
        has_bias=False,
        is_act_and_mul=gated,
        w13_num_shards=w13_num_shards,
    )
    if projections is None:
        projections = (
            ("gate_proj", "down_proj", "up_proj")
            if gated
            else ("up_proj", "down_proj", None)
        )

    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.params_dtype = dtype
    layer.moe_config = config
    layer.quant_config = None
    layer._fused_shared_expert_quantizer = None
    (
        layer.ckpt_gate_proj_name,
        layer.ckpt_down_proj_name,
        layer.ckpt_up_proj_name,
    ) = projections
    layer.is_fused_checkpoint_transposed = False
    layer.expert_map_manager = _LinearExpertMap(local_ids, num_experts)

    quant_method = object.__new__(UnquantizedFusedMoEMethod)
    object.__setattr__(quant_method, "use_global_sf", False)
    object.__setattr__(
        quant_method,
        "unquantized_backend",
        UnquantizedMoeBackend.TRITON,
    )
    object.__setattr__(
        quant_method,
        "process_weights_after_loading",
        lambda _: None,
    )
    object.__setattr__(layer, "quant_method", quant_method)

    w13 = torch.nn.Parameter(
        torch.zeros(
            experts_per_rank,
            w13_num_shards * intermediate_size,
            hidden_size,
            dtype=dtype,
        ),
        requires_grad=False,
    )
    w2 = torch.nn.Parameter(
        torch.zeros(
            experts_per_rank,
            hidden_size,
            intermediate_size,
            dtype=dtype,
        ),
        requires_grad=False,
    )
    w13.weight_loader = layer.weight_loader
    w2.weight_loader = layer.weight_loader
    layer.register_parameter("w13_weight", w13)
    layer.register_parameter("w2_weight", w2)
    return layer


def _expert_request(
    name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    shape: tuple[int, ...] | None = None,
) -> ShardedWeightRequest:
    if shape is None:
        shape = {
            "gate_up_proj.weight": (4, 6, 4),
            "w13.weight": (4, 6, 4),
            "up_proj.weight": (4, 3, 4),
            "down_proj.weight": (4, 4, 3),
        }[name]
    return ShardedWeightRequest(
        name=f"model.layers.0.mlp.experts.{name}",
        dtype=dtype,
        global_shape=shape,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("gated", "projections", "name", "local_shape"),
    [
        (
            True,
            ("gate_proj", "down_proj", "up_proj"),
            "gate_up_proj.weight",
            (2, 6, 4),
        ),
        (True, ("w1", "w2", "w3"), "w13.weight", (2, 6, 4)),
        (False, ("up_proj", "down_proj", None), "up_proj.weight", (2, 3, 4)),
    ],
)
def test_moe_runner_resolves_ep_local_logical_weights(
    dtype: torch.dtype,
    gated: bool,
    projections: tuple[str, str, str | None],
    name: str,
    local_shape: tuple[int, ...],
):
    routed_experts = _make_routed_experts(
        dtype=dtype,
        gated=gated,
        projections=projections,
    )
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = routed_experts
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = runner

    target = resolve_sharded_weight_target(
        model,
        _expert_request(name, dtype=dtype),
    )

    assert target is not None
    assert target.spec.semantic_id == "routed_experts.w13.v1"
    assert target.spec.local_shape == local_shape
    assert target.spec.shard_index == 1
    assert target.spec.num_shards == 2
    assert target.retention_group_size == 2


@pytest.mark.parametrize(
    ("gated", "w13_name", "w13_shape"),
    [
        (True, "gate_up_proj.weight", (2, 6, 4)),
        (False, "up_proj.weight", (2, 3, 4)),
    ],
)
def test_fused_expert_shards_load_into_local_owners(
    gated: bool,
    w13_name: str,
    w13_shape: tuple[int, ...],
):
    layer = _make_routed_experts(gated=gated)
    w13 = torch.arange(prod(w13_shape), dtype=torch.bfloat16).reshape(w13_shape)
    w2 = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)

    layer.load_fused_expert_shard(w13_name, w13)
    layer.load_fused_expert_shard("down_proj.weight", w2)

    assert torch.equal(layer.w13_weight, w13)
    assert torch.equal(layer.w2_weight, w2)


def test_sharded_expert_pair_uses_layerwise_reload_and_releases_together():
    layer = _make_routed_experts()
    gate_up = torch.arange(2 * 6 * 4, dtype=torch.bfloat16).reshape(2, 6, 4)
    down = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
    gate_target = layer.resolve_sharded_weight_target(
        "gate_up_proj.weight",
        _expert_request("gate_up_proj.weight"),
    )
    assert gate_target is not None
    with pytest.raises(RuntimeError, match="active layerwise reload"):
        gate_target.load(gate_up)

    record_metadata_for_reloading(layer)
    w13_ptr = layer.w13_weight.data_ptr()
    w2_ptr = layer.w2_weight.data_ptr()
    initialize_layerwise_reload(layer)
    gate_target = layer.resolve_sharded_weight_target(
        "gate_up_proj.weight",
        _expert_request("gate_up_proj.weight"),
    )
    down_target = layer.resolve_sharded_weight_target(
        "down_proj.weight",
        _expert_request("down_proj.weight"),
    )
    assert gate_target is not None
    assert down_target is not None

    assert not gate_target.load(gate_up)
    assert get_layerwise_info(layer).can_load()
    assert down_target.load(down)

    assert not get_layerwise_info(layer).can_load()
    assert layer.w13_weight.data_ptr() == w13_ptr
    assert layer.w2_weight.data_ptr() == w2_ptr
    assert torch.equal(layer.w13_weight, gate_up)
    assert torch.equal(layer.w2_weight, down)
    finalize_layerwise_reload(layer, SimpleNamespace(dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "case",
    [
        "quantized",
        "unsupported_dtype",
        "dtype_mismatch",
        "transposed",
        "tp",
        "no_ep",
        "eplb",
        "placement",
        "redundant",
        "shared",
        "bias",
        "padding",
        "uneven",
        "owners",
        "local_count",
    ],
)
def test_unsupported_logical_expert_target_is_declined(case: str):
    layer = _make_routed_experts()
    request = _expert_request("gate_up_proj.weight")
    if case == "quantized":
        layer.quant_config = object()
    elif case == "unsupported_dtype":
        request = _expert_request("gate_up_proj.weight", dtype=torch.float32)
    elif case == "dtype_mismatch":
        request = _expert_request("gate_up_proj.weight", dtype=torch.float16)
    elif case == "transposed":
        layer.is_fused_checkpoint_transposed = True
    elif case == "tp":
        layer.moe_config.moe_parallel_config.tp_size = 2
    elif case == "no_ep":
        layer.moe_config.moe_parallel_config.use_ep = False
    elif case == "eplb":
        layer.moe_config.moe_parallel_config.enable_eplb = True
    elif case == "placement":
        layer.expert_map_manager.placement_strategy = "round_robin"
    elif case == "redundant":
        layer.moe_config.num_experts = 5
    elif case == "shared":
        layer.expert_map_manager.num_fused_shared_experts = 1
    elif case == "bias":
        layer.moe_config.has_bias = True
    elif case == "padding":
        layer.moe_config.hidden_dim = layer.moe_config.hidden_dim + 1
    elif case == "uneven":
        layer.moe_config.num_experts = 5
        layer.moe_config.num_logical_experts = 5
        request = _expert_request("gate_up_proj.weight", shape=(5, 6, 4))
    elif case == "owners":
        layer.expert_map_manager._local_ids = 0, 1
    elif case == "local_count":
        layer.moe_config.num_local_experts = 3
    assert layer.resolve_sharded_weight_target("gate_up_proj.weight", request) is None


def test_malformed_logical_expert_shape_fails_closed():
    layer = _make_routed_experts()
    request = _expert_request("gate_up_proj.weight", shape=(4, 7, 4))

    with pytest.raises(ValueError, match="global shape"):
        layer.resolve_sharded_weight_target("gate_up_proj.weight", request)


def test_moonep_logical_expert_target_fails_before_transfer():
    layer = _make_routed_experts()
    object.__setattr__(
        layer.quant_method,
        "unquantized_backend",
        UnquantizedMoeBackend.MOONEP,
    )

    with pytest.raises(ValueError, match="MoonEP.*in-place weight reloads"):
        layer.resolve_sharded_weight_target(
            "gate_up_proj.weight",
            _expert_request("gate_up_proj.weight"),
        )


def test_expert_resolver_declines_unknown_or_unstacked_names():
    layer = _make_routed_experts(
        projections=("fc1", "fc2", "fc3"),
    )
    request = _expert_request("gate_up_proj.weight")

    assert layer.resolve_sharded_weight_target("unrelated.weight", request) is None
    assert layer.resolve_sharded_weight_target("gate_up_proj.weight", request) is None
    assert layer.resolve_sharded_weight_target("fc2.weight", request) is None


@pytest.mark.parametrize(
    "name",
    [
        "w13_weight",
        "w2_weight",
    ],
)
def test_kernel_formatted_expert_storage_fails_closed(name: str):
    layer = _make_routed_experts()

    with pytest.raises(ValueError, match="kernel-formatted expert storage"):
        layer.resolve_sharded_weight_target(
            name,
            _expert_request("gate_up_proj.weight"),
        )
