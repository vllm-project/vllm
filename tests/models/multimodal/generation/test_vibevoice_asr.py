# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the VibeVoice ASR model.

These tests cover:
- Unit tests for the audio encoder (CPU, no model weights required)
- Integration smoke test (requires GPU + model download, skipped by default)

Run unit tests only:
    pytest tests/models/multimodal/generation/test_vibevoice_asr.py -v -k unit

Run all tests (requires GPU and HF model access):
    pytest tests/models/multimodal/generation/test_vibevoice_asr.py -v
"""

import pytest
import torch

from vllm.model_executor.models.vibevoice_asr import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAudioEncoder,
    VibeVoiceForASRTraining,
    VibeVoiceForCausalLM,
    VibeVoiceSemanticTokenizerConfig,
    _AcousticTokenizerModel,
    _SConv1d,
    _SemanticTokenizerModel,
    _TokenizerEncoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24_000
_COMPRESS_RATIO = 3200  # prod([8,5,5,4,2,2])


def _make_acoustic_cfg(**overrides) -> VibeVoiceAcousticTokenizerConfig:
    """Tiny acoustic tokenizer config for fast CPU unit tests.

    depths must have len(ratios)+1 stages: one stem stage followed by one
    downsampling stage per ratio entry.
    """
    defaults = dict(
        encoder_n_filters=4,
        encoder_ratios=[2, 2],
        encoder_depths="1-1-1",
        vae_dim=8,
        causal=True,
        pad_mode="constant",
        layernorm="RMSNorm",
        mixer_layer="depthwise_conv",
        conv_bias=True,
        disable_last_norm=True,
        layer_scale_init_value=1e-6,
    )
    defaults.update(overrides)
    return VibeVoiceAcousticTokenizerConfig(**defaults)


def _make_semantic_cfg(**overrides) -> VibeVoiceSemanticTokenizerConfig:
    defaults = dict(
        encoder_n_filters=4,
        encoder_ratios=[2, 2],
        encoder_depths="1-1-1",
        vae_dim=8,
        causal=True,
        pad_mode="constant",
        layernorm="RMSNorm",
        mixer_layer="depthwise_conv",
        conv_bias=True,
        disable_last_norm=True,
        layer_scale_init_value=1e-6,
    )
    defaults.update(overrides)
    return VibeVoiceSemanticTokenizerConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit tests (CPU, no weights needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sconv1d_causal_output_shape():
    """SConv1d should produce expected output length for causal padding."""
    conv = _SConv1d(1, 4, kernel_size=3, stride=2, causal=True, pad_mode="constant")
    x = torch.randn(2, 1, 100)
    y = conv(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 4
    # output length ≈ ceil(100 / 2)
    assert y.shape[2] == 50


@pytest.mark.unit
def test_tokenizer_encoder_output_shape():
    """TokenizerEncoder should downsample by the product of ratios."""
    cfg = _make_acoustic_cfg()
    # Build encoder config manually (mirrors _AcousticTokenizerModel logic)
    import copy

    enc_cfg = copy.copy(cfg)
    enc_cfg.dimension = cfg.vae_dim
    enc_cfg.n_filters = cfg.encoder_n_filters
    enc_cfg.ratios = cfg.encoder_ratios
    enc_cfg.depths = [int(d) for d in cfg.encoder_depths.split("-")]
    enc_cfg.norm = cfg.conv_norm
    enc_cfg.bias = cfg.conv_bias

    encoder = _TokenizerEncoder(enc_cfg)
    encoder.eval()
    T = 400  # 400 samples
    x = torch.randn(1, 1, T)
    with torch.no_grad():
        out = encoder(x)
    # ratios=[2,2] → downsampling factor 4; depth stages match len(ratios)+1
    expected_t = T // (2 * 2)
    assert out.shape[0] == 1
    assert out.shape[1] == cfg.vae_dim
    assert abs(out.shape[2] - expected_t) <= 2, (
        f"Expected T≈{expected_t}, got {out.shape[2]}"
    )


@pytest.mark.unit
def test_acoustic_tokenizer_encode():
    """AcousticTokenizerModel.encode returns (B, T', vae_dim) mean."""
    cfg = _make_acoustic_cfg()
    model = _AcousticTokenizerModel(cfg)
    model.eval()
    audio = torch.randn(1, 1, 800)
    with torch.no_grad():
        out = model.encode(audio)
    assert out.mean.ndim == 3
    assert out.mean.shape[-1] == cfg.vae_dim


@pytest.mark.unit
def test_semantic_tokenizer_encode():
    cfg = _make_semantic_cfg()
    model = _SemanticTokenizerModel(cfg)
    model.eval()
    audio = torch.randn(1, 1, 800)
    with torch.no_grad():
        out = model.encode(audio)
    assert out.mean.ndim == 3
    assert out.mean.shape[-1] == cfg.vae_dim


@pytest.mark.unit
def test_audio_encoder_output_shape():
    """VibeVoiceAudioEncoder should return (B, N, hidden_size)."""

    class _FakeCfg:
        acoustic_tokenizer_config = dict(
            encoder_n_filters=4,
            encoder_ratios=[2, 2],
            encoder_depths="1-1-1",
            vae_dim=8,
            causal=True,
            pad_mode="constant",
            layernorm="RMSNorm",
            mixer_layer="depthwise_conv",
            conv_bias=True,
            disable_last_norm=True,
            layer_scale_init_value=1e-6,
            fix_std=0.5,
            std_dist_type="gaussian",
        )
        semantic_tokenizer_config = dict(
            encoder_n_filters=4,
            encoder_ratios=[2, 2],
            encoder_depths="1-1-1",
            vae_dim=8,
            causal=True,
            pad_mode="constant",
            layernorm="RMSNorm",
            mixer_layer="depthwise_conv",
            conv_bias=True,
            disable_last_norm=True,
            layer_scale_init_value=1e-6,
        )
        acoustic_vae_dim = 8
        semantic_vae_dim = 8
        decoder_config = type("DC", (), {"hidden_size": 16})()

    enc = VibeVoiceAudioEncoder(_FakeCfg())
    enc.eval()
    audio = torch.randn(1, 3200)  # 1 second at 3200 samples (toy rate)
    with torch.no_grad():
        embeds = enc(audio)
    assert embeds.ndim == 3
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 16  # hidden_size


@pytest.mark.unit
def test_alias_is_same_class():
    """VibeVoiceForASRTraining must resolve to VibeVoiceForCausalLM."""
    assert VibeVoiceForASRTraining is VibeVoiceForCausalLM


@pytest.mark.unit
def test_registry_contains_vibevoice():
    """Both architecture names must be registered in the vLLM model registry."""
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS

    assert "VibeVoiceForASRTraining" in _MULTIMODAL_MODELS
    assert "VibeVoiceForCausalLM" in _MULTIMODAL_MODELS


@pytest.mark.unit
def test_compress_ratio():
    """Default encoder_ratios should give a compress_ratio of 3200."""
    import math

    cfg = VibeVoiceAcousticTokenizerConfig()
    ratio = math.prod(cfg.encoder_ratios)
    assert ratio == _COMPRESS_RATIO, f"Expected {_COMPRESS_RATIO}, got {ratio}"


# ---------------------------------------------------------------------------
# Integration test (GPU + model weights required)
# ---------------------------------------------------------------------------

MODEL_ID = "microsoft/VibeVoice-ASR"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_vibevoice_asr_transcription(vllm_runner, dtype: str):
    """Smoke-test: model loads and produces non-empty transcription output."""
    from vllm.assets.audio import AudioAsset

    audio_asset = AudioAsset("winning_call")
    audio_array, sr = audio_asset.audio_and_sample_rate

    # Resample to 24 kHz if needed
    if sr != _SAMPLE_RATE:
        import librosa

        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=_SAMPLE_RATE)

    with vllm_runner(
        MODEL_ID,
        dtype=dtype,
        max_model_len=4096,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_transcription(
            prompts=[{"audio": audio_array}],
            max_tokens=200,
        )

    assert outputs, "Expected at least one output"
    text = outputs[0][0].outputs[0].text
    assert isinstance(text, str)
    assert len(text) > 0, "Transcription should not be empty"
