# vllm/tests/distributed/eplb/test_model_register.py
# -*- coding: utf-8 -*-
import pytest
import torch
from unittest.mock import MagicMock

# 直接按包路径导入被测模块
import importlib.util
import pathlib
import sys

# 计算被测文件的绝对路径：vllm/vllm/distributed/eplb/model_register.py
_THIS = pathlib.Path(__file__).resolve()
MR_PATH = _THIS.parents[3] / "vllm" / "distributed" / "eplb" / "model_register.py"

spec = importlib.util.spec_from_file_location("mr", MR_PATH)
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)



# ---------- 最小桩，匹配 self.model.layers[...] 访问 ----------
class DummyExperts:
    def __init__(self):
        self.get_map = MagicMock(return_value=torch.tensor([1, 2, 3], dtype=torch.int64))
        self.get_log2phy_map = MagicMock(return_value=torch.tensor([4, 5, 6], dtype=torch.int64))
        self.expert_load_view = torch.tensor([7, 8, 9], dtype=torch.int64)
        self.clear_moe_load = MagicMock()

class DummyMLP:
    def __init__(self):
        self.experts = DummyExperts()

class DummyLayer:
    def __init__(self):
        self.mlp = DummyMLP()

class InnerRealModel:
    """真正承载 layers 的内部模型"""
    def __init__(self, num_layers):
        self.layers = [DummyLayer() for _ in range(num_layers)]

class OuterWrapperModel:
    """
    被测对象（外层模型）：
    model_register 中的方法通过 self.model.layers[...] 访问，因此这里让 .model 指向内部模型。
    """
    def __init__(self, num_layers):
        self.model = InnerRealModel(num_layers)

class DummyHFConfig:
    def __init__(self, model_type, num_hidden_layers=2, first_k_dense_replace=0):
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        self.first_k_dense_replace = first_k_dense_replace

class DummyModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


# ---------------- 主流程：两种模型类型 ----------------
@pytest.mark.parametrize(
    "model_type,num_hidden_layers,first_k_dense_replace,expected_moe_layers,expected_dense_layers",
    [
        ("qwen3_moe", 4, 0, 4, 0),   # 全 MoE，起始 0
        ("deepseek_v2", 6, 2, 4, 2), # 前2层 dense，MoE 起始 2
    ],
)
def test_model_register_methods(model_type, num_hidden_layers, first_k_dense_replace, expected_moe_layers, expected_dense_layers):
    outer = OuterWrapperModel(num_hidden_layers)
    cfg = DummyHFConfig(model_type, num_hidden_layers, first_k_dense_replace)
    model_cfg = DummyModelConfig(cfg)

    # 注册（猴子补丁）
    mr.model_register(outer, model_cfg)

    # 方法已挂载
    for name in ["get_expert_map", "get_log2phy_map", "get_all_expert_map", "get_all_moe_loads", "clear_all_moe_loads"]:
        assert hasattr(outer, name), f"缺少方法: {name}"

    # MoE 层数正确
    assert outer.num_moe_layers == expected_moe_layers
    assert outer.num_dense_layers == expected_dense_layers

    # 单层查询
    assert torch.equal(outer.get_expert_map(0), torch.tensor([1, 2, 3], dtype=torch.int64))
    assert torch.equal(outer.get_log2phy_map(0), torch.tensor([4, 5, 6], dtype=torch.int64))

    # 聚合映射（传入 num_moe_layers）
    all_map = outer.get_all_expert_map(expected_moe_layers)
    assert isinstance(all_map, torch.Tensor)
    assert all_map.shape == (expected_moe_layers, 3)
    for row in all_map:
        assert torch.equal(row, torch.tensor([1, 2, 3], dtype=torch.int64))

    # 聚合负载
    all_loads = outer.get_all_moe_loads()
    assert isinstance(all_loads, torch.Tensor)
    assert all_loads.shape == (expected_moe_layers, 3)
    for row in all_loads:
        assert torch.equal(row, torch.tensor([7, 8, 9], dtype=torch.int64))

    # 清理逐层触发
    outer.clear_all_moe_loads()
    for i, layer in enumerate(outer.model.layers):
        called = layer.mlp.experts.clear_moe_load.called
        if expected_dense_layers <= i < expected_dense_layers + expected_moe_layers:
            assert called, f"物理层 {i} 应被清理"
        else:
            assert not called, f"物理层 {i} 不应被清理"


# ---------------- 异常与边界 ----------------
def test_model_register_not_implemented():
    outer = OuterWrapperModel(2)
    cfg = DummyHFConfig("unknown_type", 2)
    model_cfg = DummyModelConfig(cfg)
    with pytest.raises(NotImplementedError):
        mr.model_register(outer, model_cfg)


def test_get_all_expert_map_exceeds_raises():
    """传入超过实际 MoE 层数时，get_expert_map 会越界 -> IndexError."""
    outer = OuterWrapperModel(5)
    cfg = DummyHFConfig("qwen3_moe", num_hidden_layers=5, first_k_dense_replace=0)
    model_cfg = DummyModelConfig(cfg)
    mr.model_register(outer, model_cfg)

    assert outer.num_moe_layers == 5
    with pytest.raises(IndexError):
        outer.get_all_expert_map(10)

def test_zero_moe_layers_returns_empty_tensor():
    """num_moe_layers == 0 时，get_all_moe_loads 返回形状 (0, 0) 的空张量。"""
    outer = OuterWrapperModel(3)
    cfg = DummyHFConfig("deepseek_v2", num_hidden_layers=3, first_k_dense_replace=3)  # 全 dense
    model_cfg = DummyModelConfig(cfg)
    mr.model_register(outer, model_cfg)

    assert outer.num_dense_layers == 3
    assert outer.num_moe_layers == 0

    loads = outer.get_all_moe_loads()
    assert isinstance(loads, torch.Tensor)
    assert loads.shape == (0, 0)
