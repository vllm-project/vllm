# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
)

import vllm.model_executor.layers.linear as linear
import vllm.model_executor.models.qwen2_5_omni_thinker as qwen2_5_omni
import vllm.model_executor.parameter as parameter


def tiny_audio_config() -> Qwen2_5OmniAudioEncoderConfig:
    return Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        max_source_positions=16,
        n_window=4,
        output_dim=12,
    )


class RecordingMMEncoderAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.prefix = prefix
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "sequence_lengths": sequence_lengths,
            }
        )
        return torch.zeros_like(query)


def patch_single_rank_tensor_parallel(monkeypatch):
    monkeypatch.setattr(
        qwen2_5_omni,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda x: x)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", lambda x: x)


def test_qwen2_5_omni_audio_tower_is_vllm_native():
    assert not hasattr(qwen2_5_omni, "flash_attn")
    assert qwen2_5_omni.Qwen2_5OmniAudioEncoder.__module__ == qwen2_5_omni.__name__


def test_audio_attention_forwards_varlen_metadata_to_mm_encoder_attention(
    monkeypatch,
    default_vllm_config,
):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )
    patch_single_rank_tensor_parallel(monkeypatch)

    config = tiny_audio_config()
    attention = qwen2_5_omni.Qwen2_5OmniAudioAttention(config)
    hidden_states = torch.randn(5, config.d_model)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    max_seqlen = torch.tensor(3, dtype=torch.int32)

    output = attention(hidden_states, cu_seqlens, max_seqlen)

    assert output.shape == hidden_states.shape
    assert len(attention.attn.calls) == 1
    call = attention.attn.calls[0]
    assert call["query"].shape == (1, 5, 2, 4)
    assert call["key"].shape == (1, 5, 2, 4)
    assert call["value"].shape == (1, 5, 2, 4)
    assert torch.equal(call["cu_seqlens"], cu_seqlens)
    assert call["max_seqlen"] is max_seqlen


def test_audio_encoder_uses_packed_qkv_weight_structure(
    monkeypatch,
    default_vllm_config,
):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )
    patch_single_rank_tensor_parallel(monkeypatch)

    encoder = qwen2_5_omni.Qwen2_5OmniAudioEncoder(tiny_audio_config())
    keys = set(encoder.state_dict())

    assert "conv1.weight" in keys
    assert "layers.0.self_attn.qkv.weight" in keys
    assert "layers.0.self_attn.qkv.bias" in keys
    assert "layers.0.self_attn.out_proj.bias" in keys
    assert "audio_bos_eos_token.weight" in keys
    assert "proj.weight" in keys
    assert "layers.0.self_attn.q_proj.weight" not in keys
    assert "layers.0.self_attn.k_proj.weight" not in keys
    assert "layers.0.self_attn.v_proj.bias" not in keys


def test_audio_encoder_load_weights_remaps_hf_qkv_to_packed_qkv(
    monkeypatch,
    default_vllm_config,
):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )
    patch_single_rank_tensor_parallel(monkeypatch)

    config = tiny_audio_config()
    encoder = qwen2_5_omni.Qwen2_5OmniAudioEncoder(config)
    attention = encoder.layers[0].self_attn
    hidden_size = config.d_model

    q_weight = torch.arange(hidden_size * hidden_size, dtype=torch.float32).view(
        hidden_size, hidden_size
    )
    k_weight = q_weight + 100
    v_weight = q_weight + 200
    q_bias = torch.arange(hidden_size, dtype=torch.float32)
    v_bias = q_bias + 20

    with torch.no_grad():
        attention.qkv.bias.fill_(123)

    loaded = encoder.load_weights(
        [
            ("layers.0.self_attn.q_proj.weight", q_weight),
            ("layers.0.self_attn.k_proj.weight", k_weight),
            ("layers.0.self_attn.v_proj.weight", v_weight),
            ("layers.0.self_attn.q_proj.bias", q_bias),
            ("layers.0.self_attn.v_proj.bias", v_bias),
        ]
    )

    assert "layers.0.self_attn.qkv.weight" in loaded
    assert "layers.0.self_attn.qkv.bias" in loaded
    torch.testing.assert_close(attention.qkv.weight[:hidden_size], q_weight)
    torch.testing.assert_close(
        attention.qkv.weight[hidden_size : hidden_size * 2],
        k_weight,
    )
    torch.testing.assert_close(attention.qkv.weight[hidden_size * 2 :], v_weight)
    torch.testing.assert_close(attention.qkv.bias[:hidden_size], q_bias)
    torch.testing.assert_close(
        attention.qkv.bias[hidden_size : hidden_size * 2],
        torch.zeros_like(q_bias),
    )
    torch.testing.assert_close(attention.qkv.bias[hidden_size * 2 :], v_bias)


def test_audio_encoder_forward_uses_mm_encoder_attention(
    monkeypatch,
    default_vllm_config,
):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )
    patch_single_rank_tensor_parallel(monkeypatch)
    monkeypatch.setattr(
        qwen2_5_omni,
        "get_vit_attn_backend",
        lambda **_: qwen2_5_omni.AttentionBackendEnum.FLASH_ATTN,
    )

    config = tiny_audio_config()
    encoder = qwen2_5_omni.Qwen2_5OmniAudioEncoder(config)
    input_features = torch.randn(config.num_mel_bins, 8)
    feature_lens = torch.tensor([8], dtype=torch.long)
    aftercnn_lens, output_lens = encoder._get_feat_extract_output_lengths(feature_lens)

    outputs = encoder(
        input_features,
        feature_lens=feature_lens,
        aftercnn_lens=aftercnn_lens,
    )

    assert outputs.shape == (output_lens.item(), config.output_dim)
    attention = encoder.layers[0].self_attn.attn
    assert len(attention.calls) == 1
    call = attention.calls[0]
    assert call["query"].shape == (1, aftercnn_lens.item(), 2, 4)
    assert torch.equal(
        call["cu_seqlens"],
        torch.tensor([0, aftercnn_lens.item()], dtype=torch.int32),
    )
    assert call["max_seqlen"].item() == aftercnn_lens.item()
