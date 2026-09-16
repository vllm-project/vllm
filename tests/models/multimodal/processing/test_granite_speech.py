# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import Blip2QFormerConfig

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import ReplicatedLinearWithLoRA
from vllm.lora.utils import from_layer, get_supported_lora_modules
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.blip2 import Blip2QFormerModel
from vllm.model_executor.models.granite_speech import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechEncoderProjector,
    GraniteSpeechForConditionalGeneration,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


class _StubProjector:
    window_size = 15
    downsample_rate = 5
    num_queries = window_size // downsample_rate


class _StubEncoderConfig:
    context_size = 200
    num_layers = 1


class _StubProjectorConfig:
    num_hidden_layers = 1
    cross_attention_frequency = 1


class _StubModel:
    projector = _StubProjector()
    config = SimpleNamespace(
        encoder_config=_StubEncoderConfig(),
        projector_config=_StubProjectorConfig(),
    )


get_num_mm_encoder_tokens = (
    GraniteSpeechForConditionalGeneration.get_num_mm_encoder_tokens
)
get_mm_lora_token_counts = (
    GraniteSpeechForConditionalGeneration.get_mm_lora_token_counts
)


@pytest.fixture
def mock_tensor_parallel(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )


@pytest.mark.parametrize(
    ("num_audio_tokens", "expected_encoder_tokens"),
    [(0, 0), (3, 200), (6, 200), (5001, 25200)],
)
def test_num_mm_encoder_tokens(num_audio_tokens, expected_encoder_tokens):
    stub = _StubModel()

    assert get_num_mm_encoder_tokens(stub, num_audio_tokens) == expected_encoder_tokens


@pytest.mark.parametrize(
    (
        "feature_frames",
        "num_mm_embeds",
        "expected_tower_tokens",
        "expected_connector_tokens",
    ),
    [
        (1, 3, 200, 15),
        (14, 3, 200, 15),
        (15, 3, 200, 15),
        (16, 6, 200, 30),
        (201, 42, 400, 210),
    ],
)
def test_mm_lora_token_counts_use_actual_feature_frames(
    feature_frames,
    num_mm_embeds,
    expected_tower_tokens,
    expected_connector_tokens,
):
    stub = _StubModel()
    mm_kwargs = {
        "input_features": SimpleNamespace(
            data=torch.empty(1, feature_frames, 160),
        )
    }

    tower_tokens, connector_tokens = get_mm_lora_token_counts(
        stub,
        modality="audio",
        mm_kwargs=mm_kwargs,
        num_mm_embeds=num_mm_embeds,
    )
    assert tower_tokens == {
        "encoder": feature_frames,
        "encoder.layers.0.attn.to_q": expected_tower_tokens,
        "encoder.layers.0.attn.to_kv": expected_tower_tokens,
    }
    assert connector_tokens == {
        "projector": num_mm_embeds,
        "projector.qformer.encoder.layer.0.attention.attention.query": 3,
        "projector.qformer.encoder.layer.0.attention.attention.key": 3,
        "projector.qformer.encoder.layer.0.attention.attention.value": 3,
        "projector.qformer.encoder.layer.0.attention.output.dense": 3,
        "projector.qformer.encoder.layer.0.crossattention.attention.query": 3,
        "projector.qformer.encoder.layer.0.crossattention.attention.key": (
            expected_connector_tokens
        ),
        "projector.qformer.encoder.layer.0.crossattention.attention.value": (
            expected_connector_tokens
        ),
    }


def test_mm_lora_token_counts_without_runtime_features():
    stub = _StubModel()

    tower_tokens, connector_tokens = get_mm_lora_token_counts(
        stub,
        modality="audio",
        mm_kwargs=None,
        num_mm_embeds=5001,
    )
    assert tower_tokens == {
        "encoder": 25005,
        "encoder.layers.0.attn.to_q": 25200,
        "encoder.layers.0.attn.to_kv": 25200,
    }
    assert connector_tokens == {
        "projector": 5001,
        "projector.qformer.encoder.layer.0.attention.attention.query": 3,
        "projector.qformer.encoder.layer.0.attention.attention.key": 3,
        "projector.qformer.encoder.layer.0.attention.attention.value": 3,
        "projector.qformer.encoder.layer.0.attention.output.dense": 3,
        "projector.qformer.encoder.layer.0.crossattention.attention.query": 3,
        "projector.qformer.encoder.layer.0.crossattention.attention.key": 25005,
        "projector.qformer.encoder.layer.0.crossattention.attention.value": 25005,
    }


def test_granite_speech_requires_per_item_mm_lora_mapping():
    assert GraniteSpeechForConditionalGeneration.requires_mm_lora_per_item_mapping


def test_granite_speech_lora_targets_use_vllm_linear_layers(
    default_vllm_config, mock_tensor_parallel
):
    qformer_config = Blip2QFormerConfig(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=1,
        encoder_hidden_size=8,
        cross_attention_frequency=1,
    )
    projector_config = SimpleNamespace(
        projector_config=qformer_config,
        downsample_rate=5,
        window_size=15,
        text_config=SimpleNamespace(hidden_size=16),
    )
    projector = GraniteSpeechEncoderProjector(
        projector_config,
        cache_config=None,  # type: ignore[arg-type]
    )

    encoder_config = SimpleNamespace(
        context_size=4,
        max_pos_emb=4,
        input_dim=6,
        hidden_dim=8,
        output_dim=10,
        num_layers=1,
        feedforward_mult=2,
        dim_head=4,
        num_heads=2,
        conv_expansion_factor=2,
        conv_kernel_size=3,
    )
    encoder = GraniteSpeechCTCEncoder(encoder_config, prefix="encoder")

    qformer_layer = projector.qformer.encoder.layer[0]
    lora_target_layers = (
        projector.linear,
        qformer_layer.attention.attention.query,
        qformer_layer.attention.attention.key,
        qformer_layer.attention.attention.value,
        qformer_layer.attention.output.dense,
        qformer_layer.crossattention.attention.query,
        qformer_layer.crossattention.attention.key,
        qformer_layer.crossattention.attention.value,
        qformer_layer.crossattention.output.dense,
        qformer_layer.intermediate_query.dense,
        qformer_layer.output_query.dense,
        encoder.input_linear,
        encoder.layers[0].attn.to_q,
        encoder.layers[0].attn.to_kv,
        encoder.layers[0].attn.to_out,
    )

    assert all(isinstance(layer, ReplicatedLinear) for layer in lora_target_layers)
    assert {
        "linear",
        "query",
        "key",
        "value",
        "dense",
        "input_linear",
        "to_q",
        "to_kv",
        "to_out",
    } <= set(get_supported_lora_modules(torch.nn.ModuleList([projector, encoder])))
    wrapped_projector_linear = from_layer(
        projector.linear,
        max_loras=1,
        lora_config=LoRAConfig(
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_dtype=torch.float32,
        ),
        packed_modules_list=[],
    )
    assert isinstance(wrapped_projector_linear, ReplicatedLinearWithLoRA)

    params = dict(projector.named_parameters())
    weights = {name: torch.randn_like(param) for name, param in params.items()}
    loaded = AutoWeightsLoader(projector).load_weights(weights.items())
    assert loaded == set(weights)
    for name, weight in weights.items():
        assert torch.equal(params[name], weight)

    projector.eval()
    assert projector(torch.randn(1, 16, 8)).shape == (1, 6, 16)


def test_qformer_vllm_linear_matches_torch_linear(
    default_vllm_config, mock_tensor_parallel
):
    config = Blip2QFormerConfig(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=1,
        encoder_hidden_size=8,
        cross_attention_frequency=1,
        attention_probs_dropout_prob=0,
        hidden_dropout_prob=0,
    )
    torch_model = Blip2QFormerModel(
        config,
        quant_config=None,
        cache_config=None,
    )
    vllm_model = Blip2QFormerModel(
        config,
        quant_config=None,
        cache_config=None,
        use_vllm_linear=True,
    )
    vllm_model.load_state_dict(torch_model.state_dict())

    query_embeds = torch.randn(1, 3, config.hidden_size)
    encoder_hidden_states = torch.randn(1, 5, config.encoder_hidden_size)

    expected = torch_model(query_embeds, encoder_hidden_states)
    actual = vllm_model(query_embeds, encoder_hidden_states)

    torch.testing.assert_close(actual, expected)
