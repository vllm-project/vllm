# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_module
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.models.deepseek_v4.expert_map import (
    get_expert_map_layer_index,
    load_static_expert_map,
    remap_expert_params_mapping,
    remap_router_expert_ids,
    remap_router_weight,
)


def test_get_expert_map_layer_index_ignores_expert_id():
    name = "layers.0.ffn.experts.127.w1.weight_scale"

    assert get_expert_map_layer_index(name) == 0


def test_load_static_expert_map_shared_across_layers(tmp_path):
    path = tmp_path / "expert-map.json"
    path.write_text(json.dumps([2, 0, 3, 1]))

    expert_map = load_static_expert_map(str(path), num_layers=2, num_experts=4)

    torch.testing.assert_close(
        expert_map,
        torch.tensor(
            [
                [2, 0, 3, 1],
                [2, 0, 3, 1],
            ]
        ),
    )


def test_load_static_expert_map_per_layer(tmp_path):
    path = tmp_path / "expert-map.json"
    path.write_text(json.dumps([[2, 0, 3, 1], [1, 3, 0, 2]]))

    expert_map = load_static_expert_map(str(path), num_layers=2, num_experts=4)

    torch.testing.assert_close(
        expert_map,
        torch.tensor(
            [
                [2, 0, 3, 1],
                [1, 3, 0, 2],
            ]
        ),
    )


def test_load_static_expert_map_with_replicas(tmp_path):
    path = tmp_path / "expert-map.json"
    path.write_text(json.dumps([2, 0, 3, 1, 2, 1]))

    expert_map = load_static_expert_map(
        str(path),
        num_layers=2,
        num_experts=4,
        num_physical_experts=6,
    )

    torch.testing.assert_close(
        expert_map,
        torch.tensor(
            [
                [2, 0, 3, 1, 2, 1],
                [2, 0, 3, 1, 2, 1],
            ]
        ),
    )


def test_load_static_expert_map_preserves_router_groups(tmp_path):
    path = tmp_path / "expert-map.json"
    path.write_text(json.dumps([2, 3, 0, 1]))

    expert_map = load_static_expert_map(
        str(path), num_layers=1, num_experts=4, num_expert_groups=2
    )

    torch.testing.assert_close(expert_map, torch.tensor([[2, 3, 0, 1]]))

    mixed_path = tmp_path / "mixed-expert-map.json"
    mixed_path.write_text(json.dumps([0, 2, 1, 3]))
    with pytest.raises(ValueError, match="mixes logical expert groups"):
        load_static_expert_map(
            str(mixed_path),
            num_layers=1,
            num_experts=4,
            num_expert_groups=2,
        )


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ([], "non-empty"),
        ([0, 1, 2], "physical slots"),
        ([0, 1, 2, 4], "assign every logical"),
        ([0, 1, 1, 3], "assign every logical"),
        ([0, True, 2, 3], "non-integer"),
        ([[0, 1, 2, 3]] * 3, "either one shared"),
    ],
)
def test_load_static_expert_map_rejects_invalid_maps(tmp_path, contents, match):
    path = tmp_path / "expert-map.json"
    path.write_text(json.dumps(contents))

    with pytest.raises(ValueError, match=match):
        load_static_expert_map(str(path), num_layers=2, num_experts=4)


def test_static_expert_map_remaps_router_tensors():
    physical_to_logical = torch.tensor([2, 0, 3, 1])
    router_weight = torch.arange(12).view(4, 3)
    correction_bias = torch.arange(4)
    hash_expert_ids = torch.tensor([[0, 1], [2, 3]])

    torch.testing.assert_close(
        remap_router_weight(router_weight, physical_to_logical),
        router_weight[[2, 0, 3, 1]],
    )
    torch.testing.assert_close(
        remap_router_weight(correction_bias, physical_to_logical),
        correction_bias[[2, 0, 3, 1]],
    )
    torch.testing.assert_close(
        remap_router_expert_ids(hash_expert_ids, physical_to_logical),
        torch.tensor([[1, 3], [0, 2]]),
    )


def test_static_expert_map_preserves_routed_output():
    physical_to_logical = torch.tensor([2, 0, 3, 1])
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    router_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [-1.0, 3.0]])
    expert_scale = torch.tensor([2.0, 3.0, 5.0, 7.0])

    logical_ids = (hidden_states @ router_weight.T).argmax(dim=-1)
    baseline_output = hidden_states * expert_scale[logical_ids, None]

    static_router_weight = remap_router_weight(router_weight, physical_to_logical)
    physical_ids = (hidden_states @ static_router_weight.T).argmax(dim=-1)
    static_expert_scale = expert_scale[physical_to_logical]
    static_output = hidden_states * static_expert_scale[physical_ids, None]

    torch.testing.assert_close(static_output, baseline_output)


def test_static_expert_map_remaps_fused_expert_checkpoint_names():
    mapping = [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w13_", "experts.2.w1.", 2, "w1"),
    ]

    remapped = remap_expert_params_mapping(mapping, torch.tensor([2, 0, 1]))

    assert remapped == [
        ("experts.w13_", "experts.2.w1.", 0, "w1"),
        ("experts.w13_", "experts.0.w1.", 1, "w1"),
        ("experts.w13_", "experts.1.w1.", 2, "w1"),
    ]


def test_static_expert_map_requires_expert_parallelism():
    with pytest.raises(ValueError, match="enable_expert_parallel"):
        ParallelConfig(eplb_config=EPLBConfig(expert_map_path="map.json"))


def test_static_expert_map_allows_eplb_runtime_mapping():
    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(expert_map_path="map.json"),
    )

    assert config.enable_eplb


def test_static_replicated_expert_map_requires_eplb_runtime_mapping():
    with pytest.raises(ValueError, match="enable_eplb must be True"):
        ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            eplb_config=EPLBConfig(expert_map_path="map.json", num_redundant_experts=2),
        )


def test_static_replicated_map_initializes_fixed_eplb_state(monkeypatch):
    physical_to_logical = torch.tensor(
        [
            [2, 0, 3, 1, 2, 1],
            [1, 3, 0, 2, 0, 3],
        ]
    )

    class FakeModel:
        num_moe_layers = 2
        num_routed_experts = 4
        num_redundant_experts = 2
        num_physical_experts = 6
        num_logical_experts = 4
        num_expert_groups = 1
        num_shared_experts = 0
        num_local_physical_experts = 3
        static_expert_map = physical_to_logical
        moe_layers = []

        def set_eplb_state(
            self,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        ):
            self.logical_to_physical_map = logical_to_physical_map
            self.logical_replica_count = logical_replica_count
            self.expert_weights = [[torch.zeros(3, 1)] for _ in range(2)]

    monkeypatch.setattr(eplb_state_module, "get_eplb_group", lambda: SimpleNamespace())
    monkeypatch.setattr(
        eplb_state_module, "create_eplb_communicator", lambda **_: object()
    )
    parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            expert_map_path="map.json",
            use_async=True,
            window_size=2,
            step_interval=4,
            policy="default",
            communicator="torch_nccl",
        ),
        num_ubatches=0,
    )
    model_config = SimpleNamespace(model="fake", compute_hash=lambda: "fake")
    model = FakeModel()
    state = EplbState(parallel_config, torch.device("cpu"))

    state.add_model(model, model_config)

    torch.testing.assert_close(
        model.logical_replica_count,
        torch.tensor([[1, 2, 2, 1], [2, 1, 1, 2]]),
    )
    torch.testing.assert_close(
        model.logical_to_physical_map,
        torch.tensor(
            [
                [[1, -1], [3, 5], [0, 4], [2, -1]],
                [[2, 4], [0, -1], [3, -1], [1, 5]],
            ]
        ),
    )
    torch.testing.assert_close(
        state.model_states["fake"].physical_to_logical_map,
        physical_to_logical,
    )
    assert not state.is_async
    state.step()
