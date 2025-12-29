# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.config import VllmConfig, CacheConfig, ModelConfig
from vllm.distributed import init_distributed_environment
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.quantization import QuantizationConfig

# 导入待测试的模块
from vllm.model_executor.models.my_model import (
    LlamaForCausalLM,
    LlamaModel,
    MTPModule,
    MTPDecoderLayer,
    LlamaAttention,
    LlamaMLP,
)

# 初始化分布式环境（单卡测试）
init_distributed_environment(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 基础配置Fixture
@pytest.fixture(scope="module")
def base_config():
    """基础Llama配置（小型测试配置）"""
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        # MTP相关配置
        enable_mtp=True,
        mtp_num_layers=2,
        mtp_prediction_length=4,
        mtp_loss_weight=0.5,
    )
    return config

@pytest.fixture(scope="module")
def vllm_config(base_config):
    """vLLM配置Fixture"""
    model_config = ModelConfig(
        hf_config=base_config,
        model_name="test-llama-mtp",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    cache_config = CacheConfig(
        block_size=16,
        num_gpu_blocks=1024,
        num_cpu_blocks=128,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        quant_config=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return vllm_config

@pytest.fixture(scope="module")
def dummy_inputs():
    """测试用虚拟输入"""
    batch_size = 2
    seq_len = 32
    return {
        "input_ids": torch.randint(0, 32000, (batch_size, seq_len), device="cuda" if torch.cuda.is_available() else "cpu"),
        "positions": torch.arange(seq_len, device="cuda" if torch.cuda.is_available() else "cpu").repeat(batch_size, 1),
        "labels": torch.randint(-100, 32000, (batch_size, seq_len), device="cuda" if torch.cuda.is_available() else "cpu"),  # -100为ignore_index
    }

# ======================== 单元测试 ========================
class TestMTPComponents:
    """MTP核心组件单元测试"""

    def test_mtp_decoder_layer_init(self, base_config):
        """测试MTPDecoderLayer初始化"""
        layer = MTPDecoderLayer(
            config=base_config,
            hidden_size=base_config.hidden_size,
            num_heads=base_config.num_attention_heads // 2,
            num_kv_heads=base_config.num_key_value_heads // 2,
            quant_config=None,
            bias=False,
            prefix="mtp_layer.test",
        )
        # 验证属性
        assert layer.hidden_size == 512
        assert layer.self_attn.num_heads == 4  # 8//2
        assert layer.self_attn.num_kv_heads == 2  # 4//2
        assert layer.mlp.intermediate_size == 512  # 1024//2
        assert isinstance(layer.input_layernorm, nn.Module)

    def test_mtp_module_init(self, base_config):
        """测试MTPModule初始化"""
        mtp_module = MTPModule(
            config=base_config,
            quant_config=None,
            prefix="mtp_module.test",
        )
        # 验证配置
        assert mtp_module.mtp_num_layers == 2
        assert mtp_module.mtp_prediction_length == 4
        assert mtp_module.mtp_loss_weight == 0.5
        # 验证层数量
        assert len(mtp_module.mtp_layers) == 2
        # 验证投影层
        assert isinstance(mtp_module.mtp_proj, nn.Module)
        assert mtp_module.mtp_proj.input_size == 512
        assert mtp_module.mtp_proj.output_size == 512

    def test_mtp_module_forward(self, base_config, dummy_inputs):
        """测试MTPModule前向传播"""
        # 初始化MTP模块和模拟的lm_head
        mtp_module = MTPModule(
            config=base_config,
            quant_config=None,
            prefix="mtp_module.test",
        )
        mtp_module.to(dummy_inputs["input_ids"].device)
        mtp_module.train()

        # 模拟主模型hidden states
        batch_size, seq_len = dummy_inputs["input_ids"].shape
        main_hidden = torch.randn(batch_size, seq_len, base_config.hidden_size, device=dummy_inputs["input_ids"].device)
        
        # 模拟lm_head（共享权重）
        class MockLMHead(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
                self.linear.weight = self.embed_tokens.weight  # 权重共享
            
            def forward(self, x):
                return self.linear(x)
        
        lm_head = MockLMHead(base_config.vocab_size, base_config.hidden_size)
        lm_head.to(dummy_inputs["input_ids"].device)

        # 前向传播
        mtp_logits, mtp_loss = mtp_module(
            main_hidden_states=main_hidden,
            positions=dummy_inputs["positions"],
            lm_head=lm_head,
            labels=dummy_inputs["labels"],
        )

        # 验证输出形状
        assert mtp_logits.shape == (batch_size, seq_len - 4, 4, base_config.vocab_size)
        assert mtp_loss is not None
        assert isinstance(mtp_loss, torch.Tensor)
        assert mtp_loss.dim() == 0  # 标量损失

        # 验证推理模式（无labels）
        mtp_module.eval()
        mtp_logits_infer, mtp_loss_infer = mtp_module(
            main_hidden_states=main_hidden,
            positions=dummy_inputs["positions"],
            lm_head=lm_head,
            labels=None,
        )
        assert mtp_loss_infer is None
        assert mtp_logits_infer.shape == (batch_size, seq_len - 4, 4, base_config.vocab_size)

class TestLlamaComponents:
    """原有Llama组件测试（兼容MTP）"""

    def test_llama_attention_init(self, base_config):
        """测试LlamaAttention初始化"""
        attention = LlamaAttention(
            config=base_config,
            hidden_size=base_config.hidden_size,
            num_heads=base_config.num_attention_heads,
            num_kv_heads=base_config.num_key_value_heads,
            max_position_embeddings=base_config.max_position_embeddings,
            quant_config=None,
            bias=False,
            bias_o_proj=False,
            cache_config=None,
            prefix="test_attn",
        )
        assert attention.num_heads == 8
        assert attention.num_kv_heads == 4
        assert attention.head_dim == 64  # 512 / 8

    def test_llama_mlp_forward(self, base_config):
        """测试LlamaMLP前向传播"""
        mlp = LlamaMLP(
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            hidden_act="silu",
            quant_config=None,
            bias=False,
        )
        mlp.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试输入输出形状
        x = torch.randn(2, 32, 512, device=mlp.gate_up_proj.weight.device)
        output = mlp(x)
        assert output.shape == (2, 32, 512)

# ======================== 集成测试 ========================
class TestLlamaModelWithMTP:
    """集成MTP的LlamaModel测试"""

    def test_llama_model_init(self, vllm_config):
        """测试LlamaModel初始化（启用MTP）"""
        model = LlamaModel(
            vllm_config=vllm_config,
            prefix="test_model",
        )
        assert model.enable_mtp is True
        assert model.mtp_module is not None
        assert isinstance(model.mtp_module, MTPModule)
        assert isinstance(model.embed_tokens, nn.Module)

    def test_llama_for_causal_lm_init(self, vllm_config):
        """测试LlamaForCausalLM初始化"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        assert model.model.enable_mtp is True
        assert isinstance(model.lm_head, nn.Module)
        assert isinstance(model.logits_processor, nn.Module)

    def test_forward_train_mode(self, vllm_config, dummy_inputs):
        """测试训练模式下的前向传播（含MTP损失）"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        model.to(dummy_inputs["input_ids"].device)
        model.train()

        # 前向传播
        model_output, mtp_loss = model(
            input_ids=dummy_inputs["input_ids"],
            positions=dummy_inputs["positions"],
            intermediate_tensors=None,
            inputs_embeds=None,
            labels=dummy_inputs["labels"],
        )

        # 验证输出
        assert isinstance(model_output, torch.Tensor)
        assert model_output.shape == (2, 32, 512)  # batch, seq_len, hidden_size
        assert mtp_loss is not None
        assert isinstance(mtp_loss, torch.Tensor)
        assert mtp_loss.requires_grad is True  # 损失可求导

    def test_forward_infer_mode(self, vllm_config, dummy_inputs):
        """测试推理模式下的前向传播（无MTP损失）"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        model.to(dummy_inputs["input_ids"].device)
        model.eval()

        # 前向传播
        model_output = model(
            input_ids=dummy_inputs["input_ids"],
            positions=dummy_inputs["positions"],
            intermediate_tensors=None,
            inputs_embeds=None,
            labels=None,
        )

        # 验证输出
        assert isinstance(model_output, torch.Tensor)
        assert model_output.shape == (2, 32, 512)

    def test_compute_logits(self, vllm_config, dummy_inputs):
        """测试logits计算（兼容MTP）"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        model.to(dummy_inputs["input_ids"].device)

        # 模拟hidden states
        hidden_states = torch.randn(2, 32, 512, device=dummy_inputs["input_ids"].device)
        
        # 训练模式
        model.train()
        main_logits = model.compute_logits(hidden_states)
        assert main_logits.shape == (2, 32, 32000)

        # 推理模式
        model.eval()
        infer_logits = model.compute_logits(hidden_states)
        assert infer_logits.shape == (2, 32, 32000)

    def test_compute_mtp_logits(self, vllm_config, dummy_inputs):
        """测试MTP logits计算（投机解码）"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        model.to(dummy_inputs["input_ids"].device)
        model.eval()

        # 模拟hidden states
        hidden_states = torch.randn(2, 32, 512, device=dummy_inputs["input_ids"].device)
        
        # 计算MTP logits
        mtp_logits = model.compute_mtp_logits(hidden_states, dummy_inputs["positions"])
        assert mtp_logits.shape == (2, 32 - 4, 4, 32000)

    def test_weight_loading(self, vllm_config):
        """测试权重加载逻辑（MTP兼容）"""
        model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="test_causal_lm",
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        # 生成虚拟权重
        dummy_weights = []
        # 主模型权重
        dummy_weights.append(("model.embed_tokens.weight", torch.randn(32000, 512)))
        dummy_weights.append(("model.layers.0.self_attn.qkv_proj.weight", torch.randn(512, 512 + 256 + 256)))
        dummy_weights.append(("model.layers.0.mlp.gate_up_proj.weight", torch.randn(512, 1024 * 2)))
        # MTP模块权重
        dummy_weights.append(("model.mtp_module.mtp_proj.weight", torch.randn(512, 512)))
        dummy_weights.append(("model.mtp_module.mtp_layers.0.self_attn.qkv_proj.weight", torch.randn(512, 256 + 128 + 128)))

        # 加载权重
        loaded_params = model.load_weights(dummy_weights)
        assert len(loaded_params) > 0
        assert "model.mtp_module.mtp_proj.weight" in loaded_params

# ======================== 边界测试 ========================
class TestEdgeCases:
    """边界场景测试"""

    def test_mtp_disabled(self, vllm_config, base_config):
        """测试禁用MTP的情况"""
        # 修改配置禁用MTP
        base_config.enable_mtp = False
        model_config = ModelConfig(
            hf_config=base_config,
            model_name="test-llama-no-mtp",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        vllm_config.model_config = model_config

        model = LlamaModel(vllm_config=vllm_config)
        assert model.enable_mtp is False
        assert model.mtp_module is None

    def test_empty_labels(self, vllm_config, dummy_inputs):
        """测试空标签（全为ignore_index）的MTP损失"""
        model = LlamaForCausalLM(vllm_config=vllm_config)
        model.to(dummy_inputs["input_ids"].device)
        model.train()

        # 全为ignore_index的labels
        empty_labels = torch.full_like(dummy_inputs["labels"], -100)
        model_output, mtp_loss = model(
            input_ids=dummy_inputs["input_ids"],
            positions=dummy_inputs["positions"],
            labels=empty_labels,
        )

        # 损失应为0（无有效标签）
        assert torch.isclose(mtp_loss, torch.tensor(0.0, device=mtp_loss.device))

    def test_short_sequence(self, vllm_config, base_config):
        """测试短序列（长度<MTP预测长度）"""
        model = LlamaForCausalLM(vllm_config=vllm_config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.train()

        # 序列长度=3（<MTP预测长度4）
        short_input_ids = torch.randint(0, 32000, (2, 3), device=model.device)
        short_positions = torch.arange(3, device=model.device).repeat(2, 1)
        short_labels = torch.randint(-100, 32000, (2, 3), device=model.device)

        # 验证无报错（MTP自动处理短序列）
        try:
            model_output, mtp_loss = model(
                input_ids=short_input_ids,
                positions=short_positions,
                labels=short_labels,
            )
            assert True
        except Exception as e:
            pytest.fail(f"短序列处理报错: {e}")

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])