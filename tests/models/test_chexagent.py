# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ModelConfig
from vllm.model_executor.model_loader import get_model_architecture


@pytest.mark.core_model
def test_chexagent_model_loading():
    """Test that CheXagent model can be loaded correctly."""
    model_config = ModelConfig(
        "StanfordAIMI/CheXagent-8b",
        task="auto",
        trust_remote_code=True,
        seed=0,
        dtype="auto",
    )
    
    # Test that the model architecture can be resolved
    model_cls, arch = get_model_architecture(model_config)
    assert arch == "CheXagentForConditionalGeneration"
    assert model_cls.__name__ == "CheXagentForConditionalGeneration"


@pytest.mark.core_model
def test_chexagent_model_initialization():
    """Test that CheXagent model can be initialized correctly."""
    from vllm.config import VllmConfig
    from vllm.model_executor.models.chexagent import CheXagentForConditionalGeneration
    
    # Create a minimal config for testing
    model_config = ModelConfig(
        "StanfordAIMI/CheXagent-8b",
        task="auto",
        trust_remote_code=True,
        seed=0,
        dtype="auto",
    )
    
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=None,
        quant_config=None,
    )
    
    # Test model initialization
    model = CheXagentForConditionalGeneration(vllm_config=vllm_config)
    
    # Check that the model has the expected components
    assert hasattr(model, 'vision_model')
    assert hasattr(model, 'qformer')
    assert hasattr(model, 'language_model')
    assert hasattr(model, 'query_tokens')
    assert hasattr(model, 'language_projection')


@pytest.mark.core_model
def test_chexagent_multimodal_processor():
    """Test that CheXagent multimodal processor is registered correctly."""
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.chexagent import CheXagentForConditionalGeneration
    
    # Test that the processor is registered
    model_cls = CheXagentForConditionalGeneration
    assert MULTIMODAL_REGISTRY._processor_factories.contains(model_cls, strict=True)
    
    # Test that we can create a processor
    model_config = ModelConfig(
        "StanfordAIMI/CheXagent-8b",
        task="auto",
        trust_remote_code=True,
        seed=0,
        dtype="auto",
    )
    
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    assert processor is not None
    assert processor.__class__.__name__ == "CheXagentMultiModalProcessor"


@pytest.mark.core_model
def test_chexagent_qformer_components():
    """Test that CheXagent QFormer components work correctly."""
    from vllm.model_executor.models.chexagent import (
        CheXagentQFormerModel,
        CheXagentQFormerMultiHeadAttention,
        CheXagentQFormerAttention,
    )
    from transformers import PretrainedConfig
    
    # Create a minimal config for testing
    config = PretrainedConfig()
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.intermediate_size = 3072
    config.num_hidden_layers = 2
    config.attention_probs_dropout_prob = 0.1
    config.hidden_dropout_prob = 0.1
    config.layer_norm_eps = 1e-12
    config.hidden_act = "gelu"
    config.encoder_hidden_size = 1024
    
    # Test QFormer attention
    attention = CheXagentQFormerMultiHeadAttention(
        config,
        quant_config=None,
        cache_config=None,
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len, config.encoder_hidden_size)
    
    # Test self-attention
    output = attention(hidden_states)
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    # Test cross-attention
    output = attention(hidden_states, encoder_hidden_states)
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    # Test QFormer attention wrapper
    qformer_attention = CheXagentQFormerAttention(
        config,
        quant_config=None,
        cache_config=None,
    )
    
    output = qformer_attention(hidden_states)
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    output = qformer_attention(hidden_states, encoder_hidden_states)
    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.core_model
def test_chexagent_image_processing():
    """Test that CheXagent can process image inputs correctly."""
    from vllm.model_executor.models.chexagent import CheXagentForConditionalGeneration
    from vllm.config import VllmConfig, ModelConfig
    
    model_config = ModelConfig(
        "StanfordAIMI/CheXagent-8b",
        task="auto",
        trust_remote_code=True,
        seed=0,
        dtype="auto",
    )
    
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=None,
        quant_config=None,
    )
    
    model = CheXagentForConditionalGeneration(vllm_config=vllm_config)
    
    # Test image input validation
    batch_size = 2
    image_size = 224
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    # Test pixel values validation
    validated_pixel_values = model._validate_pixel_values(pixel_values)
    assert validated_pixel_values.shape == (batch_size, 3, image_size, image_size)
    
    # Test image input parsing
    image_input = model._parse_and_validate_image_input(pixel_values=pixel_values)
    assert image_input is not None
    assert image_input["type"] == "pixel_values"
    assert image_input["data"].shape == (batch_size, 3, image_size, image_size)
    
    # Test image embedding input parsing
    embedding_size = 768
    image_embeds = torch.randn(batch_size, 32, embedding_size)
    image_input = model._parse_and_validate_image_input(image_embeds=image_embeds)
    assert image_input is not None
    assert image_input["type"] == "image_embeds"
    assert image_input["data"].shape == (batch_size, 32, embedding_size) 