# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nemotron-3.5-Transcribe config checks, audio limits and the audio tower."""

import json
from typing import Any

import pytest
import torch

from vllm.transformers_utils.configs.nemotron3_5_transcribe import (
    SPEECHLM_EMBED_EXTRA_ROWS,
    Nemotron3_5TranscribeConfig,
    _checked_online,
    _preprocessor_config,
)

pytestmark = pytest.mark.cpu_test

ONLINE = {
    "window_frames": 500,
    "left_context_frames": 50,
    "right_context_frames": 50,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}

AUDIO_CONFIG = {
    "preprocessor": {"sample_rate": 16000, "n_fft": 512, "hop_length": 160},
    "asr_encoder": {"subsampling_factor": 8},
}


def test_default_preprocessor_resolves():
    preprocessor = _preprocessor_config({"normalize": None})
    assert (preprocessor["n_mels"], preprocessor["win_length"]) == (64, 320)
    # n_fft defaults to the next power of two at or above the window length.
    assert preprocessor["n_fft"] == 512
    longer_window = _preprocessor_config({"normalize": None, "window_size": 0.05})
    assert longer_window["n_fft"] == 1024


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({}, "normalize"),  # NeMo normalizes per feature when the key is missing.
        ({"normalize": "per_feature"}, "normalize"),
        ({"normalize": None, "log": False}, "log-mel"),
        ({"normalize": None, "log_zero_guard_type": "clamp"}, "log_zero_guard_type"),
    ],
)
def test_unsupported_mel_settings_are_refused(setting, message):
    with pytest.raises(ValueError, match=message):
        _preprocessor_config(setting)


def test_window_settings():
    assert _checked_online(dict(ONLINE)) == ONLINE
    for key, value in (
        ("window_frames", 0),
        ("left_context_frames", -1),
        ("right_context_frames", -1),
        ("fifo_len", -1),
        ("spkcache_update_period", 0),
        ("spkcache_len", 0),
    ):
        with pytest.raises(ValueError, match=key.split("_")[0]):
            _checked_online(ONLINE | {key: value})


def test_transcription_pieces_fit_the_context():
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        _transcription_clip_s,
    )

    assert _transcription_clip_s(AUDIO_CONFIG, 65536) == 3600  # one hour at most
    # 12.5 audio tokens and 4 transcript tokens per second in 16,384 - 256 tokens.
    assert _transcription_clip_s(AUDIO_CONFIG, 16384) == 977
    assert _transcription_clip_s(AUDIO_CONFIG, 300) == 2


def test_audio_length_sets_the_largest_clip():
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        _max_audio_samples,
        _num_audio_embeddings,
    )

    def max_tokens(length, seq_len=65536):
        samples = _max_audio_samples(AUDIO_CONFIG, seq_len, length)
        return _num_audio_embeddings(AUDIO_CONFIG, samples)

    assert max_tokens(None) == 45_000  # one hour by default
    # 69 minutes, as `--limit-mm-per-prompt '{"audio": {"length": 66240000}}'`
    # sets it, so a 65-minute clip (48,750 embeddings) fits the encoder cache.
    assert max_tokens(66_240_000) == 51_750
    # Never more than the context holds beside a short prompt.
    assert max_tokens(115_200_000) == 65_536 - 256


ENCODER = {
    "d_model": 64,
    "n_heads": 2,
    "n_layers": 1,
    "self_attention_model": "rope",
    "subsampling_factor": 8,
}

# The smallest config.json in the released checkpoint's NeMo SpeechLM layout.
CHECKPOINT_CONFIG: dict[str, Any] = {
    "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
    "model_type": "nemo_speechlm",
    "llm_config": {"model_type": "nemotron_h", "hidden_size": 64, "vocab_size": 1000},
    "perception": {
        "preprocessor": {"features": 128, "normalize": None, "window_size": 0.025},
        "modality_adapter": {"_target_": "IdentityConnector", "d_model": 64},
        "output_dim": 64,
    },
    "pe_encoder_config": {
        "asr_encoder_type": "transformer",
        "asr_encoder_cfg": ENCODER,
        "diarization_model_cfg": {
            "encoder": ENCODER,
            "high_resolution": True,
            "sortformer_modules": {"use_learnable_sil_emb": True},
        },
        "speaker_feature_mode": "thresholded",
        "speaker_activity_threshold": 0.5,
    },
}


def test_released_checkpoint_layout_loads(tmp_path):
    from vllm.config import ModelConfig

    (tmp_path / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    config = ModelConfig(str(tmp_path)).hf_config
    assert isinstance(config, Nemotron3_5TranscribeConfig)
    preprocessor = config.audio_config["preprocessor"]
    assert (preprocessor["win_length"], preprocessor["hop_length"]) == (400, 160)
    assert config.audio_config["online"] == ONLINE
    # The audio placeholder sits in embedding rows past the trained vocabulary.
    assert config.output_vocab_size == 1000
    assert config.text_config.vocab_size == 1000 + SPEECHLM_EMBED_EXTRA_ROWS


@pytest.mark.parametrize(
    ("changes", "hint"),
    [
        # This model's structure under NeMo's generic name, as older exports have.
        (
            {},
            'set "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"]',
        ),
        # Another NeMo SpeechLM: no Parallel Expert Encoder.
        ({"pe_encoder_config": None}, "NeMo Speech vLLM plugin"),
    ],
)
def test_other_nemo_speechlm_architectures_are_refused(tmp_path, changes, hint):
    from vllm.transformers_utils.config import get_config

    other = (
        CHECKPOINT_CONFIG
        | changes
        | {"architectures": ["NeMoSpeechLMForConditionalGeneration"]}
    )
    (tmp_path / "config.json").write_text(json.dumps(other))
    with pytest.raises(ValueError) as excinfo:
        get_config(str(tmp_path), trust_remote_code=False)
    assert hint in str(excinfo.value)


def _tiny_encoder(d_model: int, rotary_fraction: float, qk_norm: bool) -> dict:
    return {
        "feat_in": 16,
        "d_model": d_model,
        "n_heads": 4,
        "n_layers": 2,
        "ff_expansion": 4.0,
        "qkv_bias": False,
        "qk_norm": qk_norm,
        "pre_block_norm": True,
        "subsampling_factor": 8,
        "rope_base": 10000.0,
        "rotary_fraction": rotary_fraction,
        "xscaling": False,
    }


# 0.8 s windows, so a few seconds of audio spans several windows and fills and
# compresses the speaker cache. Continuous speaker features carry the speaker
# probabilities into the embeddings unthresholded.
TINY_AUDIO_CONFIG = {
    "preprocessor": {
        "sample_rate": 16000,
        "win_length": 400,
        "hop_length": 160,
        "n_fft": 512,
        "n_mels": 16,
        "preemph": 0.97,
        "log_zero_guard_value": 2**-24,
        "mag_power": 2.0,
        "pad_value": 0.0,
    },
    "asr_encoder": _tiny_encoder(32, 0.5, True),
    "speaker_encoder": _tiny_encoder(24, 1.0, False),
    "sortformer": {
        "num_speakers": 4,
        "fc_d_model": 24,
        "tf_d_model": 12,
        "spkcache_sil_frames_per_spk": 1,
        "pred_score_threshold": 0.25,
        "max_index": 99999,
        "scores_boost_latest": 0.05,
        "strong_boost_rate": 0.75,
        "weak_boost_rate": 1.5,
        "min_pos_scores_rate": 0.5,
    },
    "online": {
        "window_frames": 10,
        "left_context_frames": 2,
        "right_context_frames": 2,
        "fifo_len": 3,
        "spkcache_update_period": 6,
        "spkcache_len": 12,
    },
    "fusion": {
        "speaker_feature_mode": "continuous",
        "speaker_activity_threshold": None,
        "spk_kernel_scale": 1.0,
    },
    "projection": {"input_dim": 32, "output_dim": 40},
}


@pytest.fixture
def tiny_tower(dist_init):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.models.nemotron3_5_transcribe_audio import (
        Nemotron3_5TranscribeAudioTower,
    )

    torch.manual_seed(0)
    tower = Nemotron3_5TranscribeAudioTower(TINY_AUDIO_CONFIG).eval()
    with torch.no_grad():
        # vLLM's linear layers allocate their weights uninitialized.
        for module in tower.modules():
            if isinstance(module, ReplicatedLinear):
                module.weight.normal_(0.0, module.input_size**-0.5)
                if module.bias is not None:
                    module.bias.normal_(0.0, 0.1)
        featurizer = tower.preprocessor.featurizer
        featurizer.fb.uniform_()
        window = torch.hann_window(featurizer.win_length, periodic=False)
        featurizer.window.copy_(window)
        sortformer = tower.encoder.diarization_model.sortformer_modules
        sortformer.learnable_sil_emb.normal_()
        for module in tower.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.normal_(1.0, 0.1)
                module.bias.normal_(0.0, 0.1)
        # Weight loading ends with this step; on CPU it selects the GEMM kernel.
        for module in tower.modules():
            if isinstance(module, ReplicatedLinear):
                module.quant_method.process_weights_after_loading(module)
    return tower


@pytest.mark.parametrize("num_samples", [160, 161, 1_281, 12_345, 12_960, 48_001])
def test_audio_tower_returns_one_embedding_per_audio_token(tiny_tower, num_samples):
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        _num_audio_embeddings,
    )

    clip = torch.randn(num_samples) * 0.1
    with torch.inference_mode():
        (embeddings,) = tiny_tower.encode([clip], torch.tensor([num_samples]))
    num_tokens = _num_audio_embeddings(TINY_AUDIO_CONFIG, num_samples)
    assert embeddings.shape == (num_tokens, 40)


def test_decoder_weights_read_both_checkpoint_layouts():
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        Nemotron3_5TranscribeForConditionalGeneration as Model,
    )
    from vllm.model_executor.models.nemotron3_5_transcribe import _decoder_weights

    nemo = {
        "llm.model.embed_tokens.weight": torch.ones(8, 4),
        "llm.model.layers.0.mixer._fp32_params.A_log": torch.ones(2),
        "llm.model.layers.1.mixer.experts.down_projs": torch.arange(24.0).view(2, 3, 4),
        "llm.model.norm.weight": torch.ones(4),
    }
    # The FP8 and NVFP4 checkpoints: Hugging Face names, one tensor per expert.
    hf = {
        "backbone.embeddings.weight": torch.ones(8, 4),
        "backbone.layers.1.mixer.experts.0.up_proj.weight_scale": torch.ones(()),
        "lm_head.weight": torch.ones(8, 4),
    }
    mapped = Model.hf_to_vllm_mapper.apply((nemo | hf).items())
    decoder = ((name.removeprefix("language_model."), t) for name, t in mapped)
    out = dict(_decoder_weights(decoder, vocab_size=10))
    assert sorted(out) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.embeddings.weight",
        "model.layers.0.mixer.A_log",
        "model.layers.1.mixer.experts.0.down_proj.weight",
        "model.layers.1.mixer.experts.0.up_proj.weight_scale",
        "model.layers.1.mixer.experts.1.down_proj.weight",
        "model.norm_f.weight",
    ]
    for name in (
        "model.embed_tokens.weight",
        "model.embeddings.weight",
        "lm_head.weight",
    ):
        assert out[name].shape == (10, 4) and out[name][8:].eq(0).all()
    assert torch.equal(
        out["model.layers.1.mixer.experts.1.down_proj.weight"],
        nemo["llm.model.layers.1.mixer.experts.down_projs"][1].t(),
    )
    # Names outside both layouts stay unmapped, so load_weights refuses them.
    unknown = [("llm.vision.weight", torch.ones(1))]
    assert [name for name, _ in Model.hf_to_vllm_mapper.apply(unknown)] == [
        "llm.vision.weight"
    ]


def test_only_audio_has_a_placeholder():
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        AUDIO_PLACEHOLDER,
    )
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        Nemotron3_5TranscribeForConditionalGeneration as Model,
    )

    for modality in ("audio", "audio_embeds"):
        assert Model.get_placeholder_str(modality, 1) == AUDIO_PLACEHOLDER
    with pytest.raises(ValueError, match="Only audio modality is supported"):
        Model.get_placeholder_str("image", 1)


def test_quantization_layer_names_map_onto_the_model():
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        Nemotron3_5TranscribeForConditionalGeneration as Model,
    )

    mapper = Model.hf_to_vllm_mapper.get_rename_mapper()
    assert mapper.apply_list(
        [
            "backbone.layers.1.mixer.gate",
            "lm_head",
            "perception.encoder.asr_encoder.pre_encode",
            # Names the NVFP4 recipe already gives in vLLM's form stay as they are.
            "language_model.model.layers.0.mixer.in_proj",
            "mtp.layers.1.mixer.gate",
        ]
    ) == [
        "language_model.model.layers.1.mixer.gate",
        "language_model.lm_head",
        "audio_tower.encoder.asr_encoder.pre_encode",
        "language_model.model.layers.0.mixer.in_proj",
        "mtp.layers.1.mixer.gate",
    ]


@pytest.mark.usefixtures("dist_init")
def test_fp8_recipe_quantizes_the_asr_encoder_linear_layers():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.nemotron3_5_transcribe import (
        Nemotron3_5TranscribeForConditionalGeneration as Model,
    )
    from vllm.model_executor.models.nemotron3_5_transcribe_audio import (
        Nemotron3_5TranscribeAudioTower,
    )

    # As in the FP8 checkpoint's config.json.
    quant_config = ModelOptFp8Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 8,
                        "type": "float",
                    },
                }
            },
            "ignore": [
                "perception.encoder.diarization_model*",
                "perception.encoder.asr_encoder.pre_encode*",
                "perception.proj",
            ],
        }
    )
    configure_quant_config(quant_config, Model)
    tower = Nemotron3_5TranscribeAudioTower(
        TINY_AUDIO_CONFIG, quant_config=quant_config, prefix="audio_tower"
    )
    fp8_layers = sorted(
        name
        for name, module in tower.named_modules()
        if isinstance(getattr(module, "weight", None), torch.Tensor)
        and module.weight.dtype == torch.float8_e4m3fn
    )
    assert fp8_layers == sorted(
        f"encoder.asr_encoder.layers.{i}.{layer}"
        for i in range(2)
        for layer in ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
    )

    # The checkpoint stores each scale as a scalar.
    weights = []
    for name, target in [*tower.named_parameters(), *tower.named_buffers()]:
        if name == "encoder.diar_kernel":
            continue
        tensor = torch.full(target.shape, 0.5).to(target.dtype)
        if name.endswith("_scale"):
            tensor = torch.tensor(0.25)
        weights.append((name, tensor))
    tower.load_weights(weights)
    w_qkv = tower.get_submodule("encoder.asr_encoder.layers.0.attn.w_qkv")
    assert w_qkv.weight_scale.item() == w_qkv.input_scale.item() == 0.25


def test_mtp_draft_config_keeps_the_quantization_recipe():
    from vllm.config.speculative import SpeculativeConfig

    recipe = {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": ["lm_head"]}
    llm_config = CHECKPOINT_CONFIG["llm_config"] | {"num_nextn_predict_layers": 1}
    config = Nemotron3_5TranscribeConfig.from_dict(
        CHECKPOINT_CONFIG | {"llm_config": llm_config, "quantization_config": recipe}
    )
    draft = SpeculativeConfig.hf_config_override(config)
    assert draft.architectures == ["Nemotron3_5TranscribeMTPModel"]
    # The recipe lives on the SpeechLM config; the MTP head is quantized too.
    assert draft.quantization_config == recipe


def test_audio_tower_encodes_each_clip_as_if_alone(tiny_tower):
    torch.manual_seed(1)
    # The 1.3 s and 1.5 s clips span two windows and share a padded batch; the
    # 3 s clip spans four windows and is encoded apart, between them in order.
    clips = [torch.randn(num_samples) * 0.1 for num_samples in (20_800, 48_000, 24_000)]
    lengths = torch.tensor([clip.numel() for clip in clips])
    with torch.inference_mode():
        together = tiny_tower.encode(clips, lengths)
        alone = [
            tiny_tower.encode([clip], lengths[index : index + 1])[0]
            for index, clip in enumerate(clips)
        ]
    for got, expected in zip(together, alone):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
