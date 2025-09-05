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

        # 模拟每层的 expert 参数
        self._params = {}
        for i in range(config.num_hidden_layers):
            self._params[f"model.layers.{i}.mlp.experts.w13_weight"] = torch.randn(2, 4)
            self._params[f"model.layers.{i}.mlp.experts.w2_weight"] = torch.randn(2, 4)

        self.layer_count = config.num_hidden_layers

    def get_all_expert_map(self, num_moe_layers):
        # 每层有 2 个专家
        return torch.arange(num_moe_layers * 2).reshape(num_moe_layers, 2)


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

@pytest.fixture
def adaptor(dummy_model):
    with patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.all_gather_into_tensor") as mock_gather:

        def fake_all_gather(out, x):
            # out.shape = (world_size, *x.shape)
            out.copy_(x.unsqueeze(0).repeat(out.shape[0], *[1 for _ in x.shape]))
        mock_gather.side_effect = fake_all_gather

        return VllmEplbAdaptor(dummy_model)



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


def test_get_init_expert_map(monkeypatch, adaptor):
    import torch.distributed as dist

    # mock 分布式
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        dist, 
        "all_gather_into_tensor", 
        lambda gathered, expert_map: gathered.copy_(expert_map.unsqueeze(0))
    )

    maps = adaptor.get_init_expert_map(adaptor.num_moe_layers)

    assert isinstance(maps, torch.Tensor)
    assert maps.shape[0] == adaptor.num_moe_layers
    assert maps.shape[1] == 1   # world_size mock 为 1
    assert maps.shape[2] == 2   # DummyModel 里 num_experts=2



def test_get_init_expert_map_from_file(adaptor):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    data = {
        "moe_layer_count": 2,
        "layer_list": [{
            "device_count": 2,
            "device_list": [
                {"device_expert": [0, 1]},
                {"device_expert": [2, 3]}
            ]
        }]
    }
    json.dump(data, tmp)
    tmp.close()
    result = adaptor.get_init_expert_map_from_file(2, tmp.name)
    os.remove(tmp.name)
    assert isinstance(result, torch.Tensor)


def test__expert_file_to_tensor(adaptor):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump({
        "moe_layer_count": 1,
        "layer_list": [{
            "device_count": 2,
            "device_list": [{"device_expert": [0]}, {"device_expert": [1]}]
        }]
    }, tmp)
    tmp.close()
    t, layers, gpus = adaptor._expert_file_to_tensor(tmp.name)
    os.remove(tmp.name)
    assert isinstance(t, torch.Tensor)
    assert layers == 1
    assert gpus == 2


def test_do_update_expert_map(adaptor):
    t = torch.tensor([9, 9], dtype=torch.int32)
    adaptor.do_update_expert_map(adaptor.num_dense_layers, t.clone())
    assert torch.equal(adaptor.expert_map_per_layer_cpu[adaptor.num_dense_layers], t)


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
    