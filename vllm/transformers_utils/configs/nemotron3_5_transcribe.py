# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration for NVIDIA Nemotron-3.5-Transcribe.

The checkpoint's ``config.json`` is a NeMo SpeechLM config: the decoder under
``llm_config``, the speech front end under ``perception`` and the Parallel
Expert Encoder (an ASR encoder plus a Sortformer speaker encoder) under
``pe_encoder_config``. Everything the audio tower needs is resolved into one
flat ``audio_config`` dict.
"""

import math
from typing import Any

from transformers import AutoConfig, PreTrainedConfig

# The trained output vocabulary is followed by this many embedding rows. The
# audio placeholder token lives in that range.
SPEECHLM_EMBED_EXTRA_ROWS = 10

_ARCHITECTURE = "Nemotron3_5TranscribeForConditionalGeneration"
_DECODER_ARCHITECTURE = "NemotronHForCausalLM"
_HYBRID_ARCHITECTURES = ("NemotronHForCausalLM", "NemotronHybridForCausalLM")
_SUPPORTED_SPEAKER_FEATURE_MODES = ("thresholded", "continuous")


def _encoder_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a NeMo ``TransformerEncoder`` config, applying its defaults."""
    subsampling = cfg.get("subsampling", "feature_stacking")
    if subsampling != "feature_stacking":
        raise ValueError(
            f"{name}: only feature_stacking subsampling is supported, "
            f"got {subsampling!r}."
        )
    if cfg.get("self_attention_model", "rel_pos") != "rope":
        raise ValueError(f"{name}: only rope self-attention is supported.")
    if cfg.get("attn_mode", "full") != "full":
        raise ValueError(f"{name}: only full attention is supported.")
    if int(cfg.get("causal_tail_len", 0)) != 0:
        raise ValueError(f"{name}: causal_tail_len must be 0 at inference.")
    feat_out = int(cfg.get("feat_out", -1))
    d_model = int(cfg["d_model"])
    if feat_out > 0 and feat_out != d_model:
        raise ValueError(
            f"{name}: an output projection (feat_out={feat_out}) is not supported."
        )
    return {
        "feat_in": int(cfg.get("feat_in", 128)),
        "d_model": d_model,
        "n_heads": int(cfg["n_heads"]),
        "n_layers": int(cfg["n_layers"]),
        "ff_expansion": float(cfg.get("ff_expansion", 4.0)),
        "qkv_bias": bool(cfg.get("qkv_bias", False)),
        "qk_norm": bool(cfg.get("qk_norm", False)),
        "pre_block_norm": bool(cfg.get("pre_block_norm", True)),
        "subsampling_factor": int(cfg.get("subsampling_factor", 4)),
        "rope_base": float(cfg.get("rope_base", 10000.0)),
        "rotary_fraction": float(cfg.get("rotary_fraction", 1.0)),
        "xscaling": bool(cfg.get("xscaling", False)),
    }


def _preprocessor_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a NeMo ``AudioToMelSpectrogramPreprocessor`` config."""
    sample_rate = int(cfg.get("sample_rate", 16000))
    # NeMo normalizes per feature when the key is missing.
    if cfg.get("normalize", "per_feature") not in (None, "", "NA"):
        raise ValueError(
            "The perception preprocessor must not normalize; the encoders normalize "
            "their inputs."
        )
    if bool(cfg.get("exact_pad", False)):
        raise ValueError("exact_pad preprocessing is not supported.")
    if int(cfg.get("frame_splicing", 1)) != 1:
        raise ValueError("frame_splicing is not supported.")
    if cfg.get("window", "hann") != "hann":
        raise ValueError("Only the hann window is supported.")
    if not bool(cfg.get("log", True)):
        raise ValueError("Only log-mel features are supported (log=true).")
    if cfg.get("log_zero_guard_type", "add") != "add":
        raise ValueError("Only log_zero_guard_type='add' is supported.")
    win_length = int(float(cfg.get("window_size", 0.02)) * sample_rate)
    return {
        "sample_rate": sample_rate,
        "win_length": win_length,
        "hop_length": int(float(cfg.get("window_stride", 0.01)) * sample_rate),
        # NeMo's default: the next power of two at or above the window length.
        "n_fft": int(cfg.get("n_fft") or 2 ** math.ceil(math.log2(win_length))),
        "n_mels": int(cfg.get("features", 64)),
        "preemph": float(cfg.get("preemph", 0.97)),
        "log_zero_guard_value": float(cfg.get("log_zero_guard_value", 2**-24)),
        "mag_power": float(cfg.get("mag_power", 2.0)),
        "pad_value": float(cfg.get("pad_value", 0.0)),
    }


def _checked_online(online: dict[str, int]) -> dict[str, int]:
    """Reject window settings the native encoders do not implement."""
    if online["window_frames"] <= 0:
        raise ValueError(
            "The encoders run in windows; online_inference_length must be positive, "
            f"got {online['window_frames']}."
        )
    for key in ("left_context_frames", "right_context_frames", "fifo_len"):
        if online[key] < 0:
            raise ValueError(f"{key} must not be negative, got {online[key]}.")
    for key in ("spkcache_update_period", "spkcache_len"):
        if online[key] <= 0:
            raise ValueError(f"{key} must be positive, got {online[key]}.")
    return online


def resolve_audio_config(
    perception: dict[str, Any], pe_encoder_config: dict[str, Any]
) -> dict[str, Any]:
    """Flatten the NeMo perception and Parallel Expert Encoder configs."""
    if not perception:
        raise ValueError("Nemotron-3.5-Transcribe config must declare perception.")
    if not pe_encoder_config:
        raise ValueError(
            "Nemotron-3.5-Transcribe config must declare pe_encoder_config."
        )
    asr_type = pe_encoder_config.get("asr_encoder_type", "fastconformer")
    if asr_type != "transformer":
        raise ValueError(
            f"Only a transformer ASR encoder is supported, got {asr_type!r}."
        )

    diar = pe_encoder_config["diarization_model_cfg"]
    sortformer = diar.get("sortformer_modules", {})
    asr_encoder = _encoder_config(
        pe_encoder_config["asr_encoder_cfg"], "asr_encoder_cfg"
    )
    speaker_encoder = _encoder_config(diar["encoder"], "diarization_model_cfg.encoder")
    if speaker_encoder["subsampling_factor"] != asr_encoder["subsampling_factor"]:
        raise ValueError(
            "The speaker and ASR encoders must share one subsampling factor."
        )
    if diar.get("transformer_encoder") is not None:
        raise ValueError("A Sortformer transformer_encoder stage is not supported.")
    if bool(diar.get("async_streaming", False)):
        raise ValueError("Asynchronous Sortformer streaming is not supported.")
    if not bool(diar.get("high_resolution", False)):
        raise ValueError("The speaker encoder must be a high-resolution Sortformer.")
    if not bool(sortformer.get("use_learnable_sil_emb", False)):
        raise ValueError("The speaker encoder must use a learnable silence embedding.")
    if float(sortformer.get("activity_weight", diar.get("activity_weight", 0.0))) > 0.0:
        raise ValueError("A Sortformer activity head is not supported.")
    diar_normalize = pe_encoder_config.get("diar_normalize_type")
    if diar_normalize is None:
        diar_normalize = diar.get("preprocessor", {}).get("normalize")
    if diar_normalize not in (None, "", "NA"):
        raise ValueError(
            f"Speaker-encoder input normalization {diar_normalize!r} is not supported."
        )
    asr_normalize = pe_encoder_config.get("asr_normalize_type") or "per_feature"
    if asr_normalize != "per_feature":
        raise ValueError(
            f"ASR-encoder input normalization {asr_normalize!r} is not supported."
        )

    mode = str(pe_encoder_config.get("speaker_feature_mode", "thresholded")).lower()
    if mode not in _SUPPORTED_SPEAKER_FEATURE_MODES:
        raise ValueError(f"Unsupported speaker_feature_mode {mode!r}.")
    threshold = pe_encoder_config.get("speaker_activity_threshold")
    if mode == "thresholded" and threshold is None:
        raise ValueError(
            "speaker_feature_mode='thresholded' requires speaker_activity_threshold."
        )

    modality_adapter = perception.get("modality_adapter", {})
    if not str(modality_adapter.get("_target_", "")).endswith("IdentityConnector"):
        raise ValueError("Only the identity modality adapter is supported.")
    projection_in = int(modality_adapter.get("d_model", asr_encoder["d_model"]))
    if projection_in != asr_encoder["d_model"]:
        raise ValueError("The projection input must match the ASR encoder width.")

    return {
        "preprocessor": _preprocessor_config(perception["preprocessor"]),
        "asr_encoder": asr_encoder,
        "speaker_encoder": speaker_encoder,
        "sortformer": {
            "num_speakers": int(
                sortformer.get("num_spks", diar.get("max_num_of_spks", 4))
            ),
            "fc_d_model": int(sortformer.get("fc_d_model", speaker_encoder["d_model"])),
            "tf_d_model": int(sortformer.get("tf_d_model", 192)),
            "spkcache_sil_frames_per_spk": int(
                sortformer.get("spkcache_sil_frames_per_spk", 3)
            ),
            "pred_score_threshold": float(sortformer.get("pred_score_threshold", 0.25)),
            "max_index": int(sortformer.get("max_index", 99999)),
            "scores_boost_latest": float(sortformer.get("scores_boost_latest", 0.05)),
            "strong_boost_rate": float(sortformer.get("strong_boost_rate", 0.75)),
            "weak_boost_rate": float(sortformer.get("weak_boost_rate", 1.5)),
            "min_pos_scores_rate": float(sortformer.get("min_pos_scores_rate", 0.5)),
        },
        "online": _checked_online(
            {
                "window_frames": int(
                    pe_encoder_config.get("online_inference_length", 500)
                ),
                "left_context_frames": int(
                    pe_encoder_config.get("chunk_left_context", 50)
                ),
                "right_context_frames": int(
                    pe_encoder_config.get("chunk_right_context", 50)
                ),
                "fifo_len": int(pe_encoder_config.get("diar_fifo_len", 40)),
                "spkcache_update_period": int(
                    pe_encoder_config.get("diar_spkcache_update_period", 300)
                ),
                "spkcache_len": int(pe_encoder_config.get("diar_spkcache_len", 188)),
            }
        ),
        "fusion": {
            "speaker_feature_mode": mode,
            "speaker_activity_threshold": None
            if threshold is None
            else float(threshold),
            "spk_kernel_scale": float(pe_encoder_config.get("spk_kernel_scale", 1.0)),
        },
        "projection": {
            "input_dim": projection_in,
            "output_dim": int(perception["output_dim"]),
        },
    }


def _build_text_config(llm_config: dict[str, Any] | None) -> PreTrainedConfig:
    if llm_config is None:
        return AutoConfig.for_model("nemotron_h")
    if not isinstance(llm_config, dict):
        raise ValueError(f"llm_config must be a dict, got {type(llm_config).__name__}.")
    llm_config = dict(llm_config)
    model_type = llm_config.pop("model_type", None)
    if not model_type:
        raise ValueError("llm_config must declare model_type.")
    return AutoConfig.for_model(model_type, **llm_config)


def _decoder_architectures(text_config: PreTrainedConfig) -> list[str]:
    return getattr(text_config, "architectures", None) or [_DECODER_ARCHITECTURE]


def _other_architecture_hint(
    pe_encoder_config: dict[str, Any] | None, text_config: PreTrainedConfig
) -> str:
    # NeMo names checkpoints with this structure after this model; older exports
    # carry NeMo's generic SpeechLM name.
    if pe_encoder_config and set(_decoder_architectures(text_config)) & set(
        _HYBRID_ARCHITECTURES
    ):
        return (
            "This checkpoint has the Parallel Expert Encoder on a NemotronH decoder: "
            f'set "architectures": ["{_ARCHITECTURE}"] in its config.json.'
        )
    return "Serve other NeMo SpeechLM models with the NeMo Speech vLLM plugin."


class Nemotron3_5TranscribeConfig(PreTrainedConfig):
    """Configuration for Nemotron-3.5-Transcribe.

    Reads the checkpoint's NeMo SpeechLM ``config.json`` (see the module
    docstring). The decoder config becomes ``text_config``; the audio tower
    settings become ``audio_config`` (see :func:`resolve_audio_config`).

    ``nemo_speechlm`` is NeMo's model type for all its SpeechLM checkpoints; the
    architecture name selects Nemotron-3.5-Transcribe.
    """

    model_type = "nemo_speechlm"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        perception: dict[str, Any] | None = None,
        pe_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | PreTrainedConfig | None = None,
        audio_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Transformers may read the text config while the base class
        # initializes; give it the real one before calling super().__init__.
        if llm_config is not None or text_config is None:
            self.text_config = _build_text_config(llm_config)
        elif isinstance(text_config, PreTrainedConfig):
            self.text_config = text_config
        else:
            self.text_config = _build_text_config(text_config)
        super().__init__(**kwargs)
        if self.architectures and self.architectures != [_ARCHITECTURE]:
            raise ValueError(
                f"Only {_ARCHITECTURE} is supported among NeMo SpeechLM checkpoints, "
                f"got architectures={self.architectures!r}. "
                + _other_architecture_hint(pe_encoder_config, self.text_config)
            )

        self.llm_config = llm_config
        self.perception = perception or {}
        self.pe_encoder_config = pe_encoder_config or {}

        decoder_config = self.text_config
        architectures = _decoder_architectures(decoder_config)
        if len(architectures) != 1 or architectures[0] not in _HYBRID_ARCHITECTURES:
            raise ValueError(
                f"Expected a NemotronH decoder, got architectures={architectures!r}."
            )
        decoder_config.architectures = [_DECODER_ARCHITECTURE]

        if (
            llm_config is not None
            and getattr(decoder_config, "speechlm_output_vocab_size", None) is None
        ):
            decoder_config.speechlm_output_vocab_size = int(decoder_config.vocab_size)
            decoder_config.vocab_size = (
                int(decoder_config.vocab_size) + SPEECHLM_EMBED_EXTRA_ROWS
            )
        self.output_vocab_size = int(
            getattr(
                decoder_config, "speechlm_output_vocab_size", decoder_config.vocab_size
            )
        )

        if self.perception:
            self.audio_config = resolve_audio_config(
                self.perception, self.pe_encoder_config
            )
        else:
            self.audio_config = audio_config or {}

    def get_text_config(self, decoder: bool = False) -> PreTrainedConfig:
        return self.text_config
