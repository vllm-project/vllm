import pytest
import torch
from vllm.model_executor.models.modeling_minimax_vl_01 import (
    MiniMaxVL01ForConditionalGeneration,
    MiniMaxVL01Config,
    MiniMaxVL01ProcessingInfo,
    MiniMaxVL01DummyInputsBuilder
)
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from transformers import BatchFeature
from vllm.transformers_utils.configs.configuration_minimax_text_01 import MiniMaxText01Config
from vllm.transformers_utils.configs.configuration_minimax_vl_01 import MiniMaxVL01Config as HfMiniMaxVL01Config
from vllm.multimodal.processing import ProcessingContext

def test_minimax_vl_01_basic_flow():
    """测试MiniMaxVL01模型的基本流程"""
    # 1. 创建配置
    text_config = MiniMaxText01Config(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
    )
    
    config = HfMiniMaxVL01Config(
        text_config=text_config,
        vision_config=None,  # 使用默认的CLIPVisionConfig
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_grid_pinpoints=[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
        tie_word_embeddings=False,
        image_seq_length=576,
    )
    
    vllm_config = VllmConfig(
        model_config=config,
        quant_config=None,
        multimodal_config=None
    )
    
    # 2. 创建模型实例
    model = MiniMaxVL01ForConditionalGeneration(vllm_config=vllm_config)
    
    # 3. 创建处理信息
    ctx = ProcessingContext()
    processing_info = MiniMaxVL01ProcessingInfo(ctx)
    
    # 4. 创建虚拟输入构建器
    dummy_builder = MiniMaxVL01DummyInputsBuilder(processing_info)
    
    # 5. 测试虚拟输入生成
    dummy_inputs = dummy_builder.get_dummy_processor_inputs(
        seq_len=10,
        mm_counts={"image": 1}
    )
    
    # 验证生成的输入
    assert dummy_inputs.prompt_text is not None
    assert "image" in dummy_inputs.mm_data
    assert len(dummy_inputs.mm_data["image"]) == 1
    
    # 6. 测试模型前向传播
    batch_size = 2
    seq_len = 10
    hidden_size = config.text_config.hidden_size
    
    # 创建模拟输入
    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 创建模拟图像输入
    pixel_values = torch.randn(batch_size, 3, 224, 224)  # 示例图像尺寸
    image_sizes = torch.tensor([[224, 224] for _ in range(batch_size)])
    
    # 执行前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            positions=positions,
            pixel_values=pixel_values,
            image_sizes=image_sizes
        )
    
    # 验证输出
    assert outputs is not None
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, seq_len, hidden_size)

def test_minimax_vl_01_processor():
    """测试MiniMaxVL01处理器的基本功能"""
    # 1. 创建处理信息
    ctx = ProcessingContext()
    processing_info = MiniMaxVL01ProcessingInfo(ctx)
    
    # 2. 测试获取配置
    hf_config = processing_info.get_hf_config()
    assert hf_config is not None
    
    # 3. 测试获取处理器
    processor = processing_info.get_hf_processor()
    assert processor is not None
    
    # 4. 测试获取图像token数量
    num_tokens = processing_info.get_num_image_tokens(
        image_width=224,
        image_height=224
    )
    assert num_tokens > 0
    
    # 5. 测试获取最大特征尺寸
    max_size = processing_info.get_image_size_with_most_features()
    assert max_size is not None
    assert hasattr(max_size, 'width')
    assert hasattr(max_size, 'height') 