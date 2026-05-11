# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
)

import vllm.model_executor.models.qwen2_5_omni_thinker as qwen2_5_omni


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


def test_qwen2_5_omni_audio_tower_is_vllm_native():
    assert not hasattr(qwen2_5_omni, "flash_attn")
    assert qwen2_5_omni.Qwen2_5OmniAudioEncoder.__module__ == qwen2_5_omni.__name__


def test_audio_attention_forwards_varlen_metadata_to_mm_encoder_attention(
    monkeypatch,
):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )

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


def test_audio_encoder_keeps_huggingface_checkpoint_key_names(monkeypatch):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )

    encoder = qwen2_5_omni.Qwen2_5OmniAudioEncoder(tiny_audio_config())
    keys = set(encoder.state_dict())

    assert "conv1.weight" in keys
    assert "layers.0.self_attn.q_proj.weight" in keys
    assert "layers.0.self_attn.k_proj.weight" in keys
    assert "layers.0.self_attn.v_proj.bias" in keys
    assert "layers.0.self_attn.out_proj.bias" in keys
    assert "audio_bos_eos_token.weight" in keys
    assert "proj.weight" in keys
    assert not any("qkv" in key for key in keys)


def test_audio_encoder_forward_uses_mm_encoder_attention(monkeypatch):
    monkeypatch.setattr(
        qwen2_5_omni,
        "MMEncoderAttention",
        RecordingMMEncoderAttention,
    )
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

    assert outputs.last_hidden_state.shape == (output_lens.item(), config.output_dim)
    attention = encoder.layers[0].self_attn.attn
    assert len(attention.calls) == 1
    call = attention.calls[0]
    assert call["query"].shape == (1, aftercnn_lens.item(), 2, 4)
    assert torch.equal(
        call["cu_seqlens"],
        torch.tensor([0, aftercnn_lens.item()], dtype=torch.int32),
    )
    assert call["max_seqlen"].item() == aftercnn_lens.item()
