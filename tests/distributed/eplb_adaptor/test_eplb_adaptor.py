import os
import json
import tempfile
import torch
import pytest
from unittest.mock import MagicMock, patch
from vllm.distributed.eplb.eplb_adaptor.vllm_adaptor import VllmEplbAdaptor

class DummyConfig:
    def __init__(self, model_type="qwen3_moe", num_experts=4,
                 first_k_dense_replace=3, n_routed_experts=6,
                 num_hidden_layers=6):
        self.model_type = model_type
        self.num_experts = num_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.num_hidden_layers = num_hidden_layers

class DummyModel:
    def __init__(self, config, quant=False):
        self.config = config
        self.quant_config = {} if quant else None

        # mock expert weights for each layer
        self._params = {}
        for i in range(config.num_hidden_layers):
            self._params[f"model.layers.{i}.mlp.experts.w13_weight"] = torch.randn(2, 4)
            self._params[f"model.layers.{i}.mlp.experts.w2_weight"] = torch.randn(2, 4)

        self.layer_count = config.num_hidden_layers

    def get_all_expert_map(self, num_moe_layers: int):
        per_layer = torch.tensor([0, 1], dtype=torch.int32)
        return per_layer.repeat(num_moe_layers, 1)  # shape: [num_moe_layers, 2]


    def named_parameters(self):
        for k, v in self._params.items():
            yield k, v

    def get_expert_map(self, layer_idx):
        return torch.tensor([0, 1], dtype=torch.int32)

    def get_log2phy_map(self, layer_idx):
        return torch.tensor([0, 1], dtype=torch.int32)

    def get_all_moe_loads(self):
        return torch.tensor([1, 2, 3])

    def get_all_expert_map(self, num_moe_layers):
        return torch.tensor([[0, 1], [2, 3]])


@pytest.fixture
def dummy_model():
    return DummyModel(DummyConfig())


@pytest.fixture(autouse=True)
def adaptor(dummy_model):
    from vllm.distributed.eplb.eplb_adaptor.vllm_adaptor import VllmEplbAdaptor
    with patch("vllm.distributed.eplb.eplb_adaptor.vllm_adaptor.dist.get_rank", return_value=0), \
         patch("vllm.distributed.eplb.eplb_adaptor.vllm_adaptor.dist.get_world_size", return_value=2), \
         patch("vllm.distributed.eplb.eplb_adaptor.vllm_adaptor.dist.is_initialized", return_value=True), \
         patch("vllm.distributed.eplb.eplb_adaptor.vllm_adaptor.dist.init_process_group") as mock_init_pg, \
         patch("vllm.distributed.eplb.eplb_adaptor.vllm_adaptor.dist.all_gather_into_tensor") as mock_gather:

        # ---- mock init_process_group ----
        mock_pg = MagicMock(name="fake_pg")
        mock_init_pg.return_value = mock_pg

        # ---- fake all_gather_into_tensor ----
        def fake_all_gather(out, x):
            W, L, E = out.shape
            out.copy_(x.unsqueeze(0).expand(W, L, E))
        mock_gather.side_effect = fake_all_gather

        # ---- create adaptor ----
        adaptor = VllmEplbAdaptor(dummy_model)
        print(f"[DEBUG:init] adaptor.rank_id = {adaptor.rank_id}, world_size = {adaptor.world_size}")
        return adaptor


def test_init_and_buffer(adaptor):
    assert isinstance(adaptor.buffer_tensor_list, list)
    assert all(isinstance(x, list) for x in adaptor.buffer_tensor_list)


def test_init_expert_param_per_layer(adaptor):
    assert isinstance(adaptor.expert_param_per_layer, dict)
    for k, v in adaptor.expert_param_per_layer.items():
        assert isinstance(v, list)

def test_get_rank_expert_workload(adaptor):
    workload = adaptor.get_rank_expert_workload()
    assert torch.equal(workload, torch.tensor([1, 2, 3]))

def test_get_init_expert_map_from_file(adaptor):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
        data = {
            "moe_layer_count": 2,
            "layer_list": [
                {
                    "device_count": 2,
                    "device_list": [
                        {"device_expert": [0, 1]},
                        {"device_expert": [2, 3]}
                    ]
                },
                {
                    "device_count": 2,
                    "device_list": [
                        {"device_expert": [4, 5]},
                        {"device_expert": [6, 7]}
                    ]
                }
            ]
        }
        json.dump(data, tmp)
        tmp_path = tmp.name

    result = adaptor.get_init_expert_map_from_file(2, tmp_path)
    os.remove(tmp_path)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 2   


def test__expert_file_to_tensor(adaptor):
    import tempfile, os, json, torch

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump({
            "moe_layer_count": 1,
            "layer_list": [{
                "device_count": 2,
                "device_list": [{"device_expert": [0]}, {"device_expert": [1]}]
            }]
        }, tmp)
        tmp_path = tmp.name  

    t, layers, gpus = adaptor._expert_file_to_tensor(tmp_path)
    os.remove(tmp_path)

    assert isinstance(t, torch.Tensor)
    assert layers == 1
    assert gpus == 2
    assert t.shape == (1, 2, 1)  

def test_do_update_expert_map(adaptor):
    import torch

    t = torch.tensor([9, 9], dtype=torch.int32)
    layer_id = adaptor.num_dense_layers

    # init
    adaptor.expert_map_per_layer[layer_id] = torch.zeros_like(t)
    adaptor.expert_map_per_layer_cpu[layer_id] = torch.zeros_like(t)

    adaptor.do_update_expert_map(layer_id, t.clone())

    assert torch.equal(adaptor.expert_map_per_layer[layer_id], t)
    assert torch.equal(adaptor.expert_map_per_layer_cpu[layer_id], t)



def test_do_update_expert_weight(adaptor):
    layer_id = adaptor.num_dense_layers
    adaptor.do_update_expert_weight(layer_id, 0, 0)
    # just ensure no error


def test_do_update_log2phy_map(adaptor):
    layer_id = adaptor.num_dense_layers
    new_map = torch.tensor([5, 6], dtype=torch.int32)
    adaptor.do_update_log2phy_map(layer_id, new_map)
    assert torch.equal(adaptor.log2phy_map_per_layer[layer_id], new_map)


def test_local2global(adaptor):
    local = torch.tensor([[[0, 1], [2, 3]]])
    global_map = adaptor.local2global(local)
    assert isinstance(global_map, torch.Tensor)
    assert global_map.shape[2] == 4


def test_local2global_empty(adaptor):
    local = torch.tensor([[[-1, -1], [-1, -1]]])
    global_map = adaptor.local2global(local)
    assert global_map.shape[2] == 0


def test_determine_expert_map_all(adaptor):
    result = adaptor.determine_expert_map_all()
    assert isinstance(result, torch.Tensor)
    assert result.shape[1] == adaptor.world_size