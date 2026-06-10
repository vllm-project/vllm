# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import BatchFeature, PretrainedConfig

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import (
    MultiModalDataDict,
    PromptType,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.omniasr import OmniASRConfig
from vllm.transformers_utils.processors.omniasr import (
    OmniASRFeatureExtractor,
    OmniASRProcessor,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .whisper import ISO639_1_SUPPORTED_LANGS

logger = init_logger(__name__)
MAX_AUDIO_CLIP_S = 40
SAMPLING_RATE_HZ = 16000
NORMALIZATION_EPS = 1e-7
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 0

# ISO->Omniasr code mapping for common languages.
# Users can also pass OmniASR native codes (e.g., "eng_Latn") directly.
ISO_TO_OMNIASR_CODE = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "cmn_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "uk": "ukr_Cyrl",
    "id": "ind_Latn",
    "fi": "fin_Latn",
    "sv": "swe_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "ro": "ron_Latn",
}


@lru_cache(maxsize=8)
def _load_lang_table(model_id: str) -> dict[str, int]:
    if os.path.isdir(model_id):
        path = os.path.join(model_id, "lang_lookup.json")
    else:
        path = hf_hub_download(repo_id=model_id, filename="lang_lookup.json")
    with open(path) as f:
        return json.load(f)


def _resolve_lang_id(lang: str | None, model_id: str, n_special: int) -> int:
    """Resolve ISO-639-1 or OmniASR-native code to lang_id."""
    table = _load_lang_table(model_id)
    if lang is None:
        return table["eng_Latn"] + n_special
    omniasr_code = ISO_TO_OMNIASR_CODE.get(lang, lang)
    if omniasr_code not in table:
        raise ValueError(
            f"Unsupported language: {lang!r}. "
            f"Use ISO-639-1 (e.g., 'en') or OmniASR native code (e.g., 'eng_Latn')."
        )
    return table[omniasr_code] + n_special


def _permute_q_k_for_neox(w, num_heads):
    """Permute Q/K weights from GPT-J to NeoX RoPE convention.

    Per-head: take rows at even positions first, then rows at odd positions.
    This makes vLLM's default NeoX RoPE produce output mathematically
    equivalent to the original GPT-J RoPE.
    """
    out_dim = w.shape[0]
    head_dim = out_dim // num_heads
    rest_shape = w.shape[1:]

    return (
        w.view(num_heads, head_dim // 2, 2, *rest_shape)
        .transpose(1, 2)
        .reshape(out_dim, *rest_shape)
    )


class OmniASRModel(nn.Module):
    """
    Full OmniASR model: encoder + projection + LLaMA decoder.

    This class encapsulates the audio encoder (Wav2Vec2), the projection layer
    to match decoder dimensions, and the text/language embeddings.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.encoder_frontend = Wav2Vec2Frontend(config)
        self.encoder = Wav2Vec2TransformerEncoder(config)
        self.encoder_proj = ColumnParallelLinear(
            config.encoder_embed_dim, config.projection_dim, bias=True
        )
        self.lang_embeddings = VocabParallelEmbedding(
            config.num_languages, config.text_config.hidden_size
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OmniASR model.

        Args:
            audio: [batch_size, num_samples] - input audio waveform.

        Returns:
            [batch_size, seq_len, projection_dim] - encoder output projected to
            decoder hidden size.
        """
        x = self.encoder_frontend(audio)
        x = self.encoder(x)
        x, _ = self.encoder_proj(x)

        return x  # [batch, seq, config.projection_dim] ready for LLaMA decoder

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for the OmniASR model.

        Args:
            weights: An iterable of (name, loaded_weight) tuples.

        Returns:
            A set of parameter names that were loaded.
        """
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # replace weight name with param name
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Wav2Vec2FeatureExtractor(nn.Module):
    """
    Wav2Vec2 feature extractor consisting of multiple 1D convolutional layers.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.sampling_rate = config.sampling_rate

        in_ch = 1  # mono audio
        for out_ch, kernel_size, stride in config.feature_extractor_layer_descs:
            conv = nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                bias=config.feature_extractor_bias,
            )
            layer_norm = nn.LayerNorm(out_ch)
            self.layers.append(nn.ModuleDict({"conv": conv, "layer_norm": layer_norm}))
            in_ch = out_ch
        for layer in self.layers:
            conv = layer["conv"]
            if conv.padding[0] != 0 or conv.dilation[0] != 1:
                raise ValueError(
                    f"Wav2Vec2FeatureExtractor.compute_seq_len assumes "
                    f"padding=0/dilation=1; got padding={conv.padding[0]}, "
                    f"dilation={conv.dilation[0]}"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional layers to input waveform.

        Args:
            x: [batch_size, 1, num_samples] - input audio waveform.

        Returns:
            [batch_size, feature_dim, seq_len] - extracted features.
        """
        for layer in self.layers:
            x = layer["conv"](x)
            x = x.transpose(1, 2)
            x = layer["layer_norm"](x)
            x = x.transpose(1, 2)
            x = nn.functional.gelu(x)
        return x

    @staticmethod
    def compute_seq_len(config: OmniASRConfig, num_samples: int) -> int:
        """
        Compute output sequence length after CNN feature extraction.

        Pure function — operates on config without instantiating the module.
        Mirrors PyTorch Conv1d's output_size formula.

        IMPORTANT: padding=0, dilation=1 must match the defaults used when
        Conv1d is constructed in __init__. If __init__ is modified to use
        non-default padding or dilation, this method MUST be updated too.
        """
        seq_len = num_samples
        padding = 0  # default in Conv1d construction
        dilation = 1  # default in Conv1d construction
        for _out_ch, kernel_size, stride in config.feature_extractor_layer_descs:
            seq_len = (
                seq_len + 2 * padding - dilation * (kernel_size - 1) - 1
            ) // stride + 1
        return seq_len


class Wav2Vec2Attention(nn.Module):
    """
    Self-attention with separate q/k/v/output projections (matching checkpoint).
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        embed_dim = config.encoder_embed_dim
        num_heads = config.encoder_num_heads
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_heads // tp_size
        self.head_dim = embed_dim // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
        )
        self.output_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
        )
        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Wav2Vec2 attention.

        Args:
            x: [batch_size, seq_len, hidden_size] - input hidden states.

        Returns:
            [batch_size, seq_len, hidden_size] - output of attention layer.
        """
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.output_proj(attn_output)
        return output


class Wav2Vec2FFN(nn.Module):
    """
    Feed-forward network for Wav2Vec2 encoder layer.
    """

    def __init__(
        self,
        config: OmniASRConfig,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        embed_dim = config.encoder_embed_dim
        ffn_dim = config.encoder_ffn_dim
        self.inner_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.output_proj = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Wav2Vec2 FFN.

        Args:
            x: [batch_size, seq_len, hidden_size] - input hidden states.

        Returns:
            [batch_size, seq_len, hidden_size] - output of FFN.
        """
        x, _ = self.inner_proj(x)
        x = nn.functional.gelu(x)
        x, _ = self.output_proj(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer for Wav2Vec2.

    Each layer consists of a self-attention mechanism and a feed-forward
    network, with LayerNorm applied before each and residual connections
    after each.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        embed_dim = config.encoder_embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = Wav2Vec2Attention(config)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = Wav2Vec2FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Wav2Vec2Frontend(nn.Module):
    """
    Frontend for Wav2Vec2 including feature extraction and positional encoding.

    This module handles the initial processing of raw audio waveforms,
    including convolutional feature extraction, layer normalization,
    and additive positional embeddings.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        feature_dim = config.feature_dim
        embed_dim = config.encoder_embed_dim
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.post_extract_layer_norm = nn.LayerNorm(feature_dim)
        self.model_dim_proj = nn.Linear(feature_dim, embed_dim, bias=True)
        # pos_encoder: store as plain conv, handle weight_norm in weight loading
        self.pos_encoder = nn.ModuleDict(
            {
                "conv": nn.utils.weight_norm(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=config.pos_encoder_kernel_size,
                        padding=config.pos_encoder_kernel_size // 2,
                        groups=config.pos_encoder_groups,
                        bias=True,
                    ),
                    name="weight",
                    dim=2,
                )
            }
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.to(next(self.parameters()).dtype)  # add dtype conversion
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)
        pos = self.pos_encoder["conv"](x.transpose(1, 2))
        pos = pos[:, :, : x.shape[1]]
        pos = nn.functional.gelu(pos)
        x = x + pos.transpose(1, 2)
        return x


class Wav2Vec2TransformerEncoder(nn.Module):
    """
    Transformer encoder for Wav2Vec2.

    Comprises multiple stackable Transformer encoder layers followed by
    a final LayerNorm. Processes extracted audio features into high-level
    representations.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.encoder_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class OmniASRProcessingInfo(BaseProcessingInfo):
    """
    Processing information for the OmniASR model.

    Provides metadata and helper methods for handling audio inputs,
    including sequence length computation.
    """

    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_num_audio_tokens(self, num_samples: int) -> int:
        return Wav2Vec2FeatureExtractor.compute_seq_len(
            self.get_hf_config(), num_samples
        )

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.get_hf_config().sampling_rate)

    def get_hf_processor(self, **kwargs: object) -> OmniASRProcessor:
        if not hasattr(self, "_cached_hf_processor"):
            hf_config = self.get_hf_config()
            sampling_rate = hf_config.sampling_rate
            feature_extractor = OmniASRFeatureExtractor(
                sampling_rate=sampling_rate,
                padding_value=0.0,
                feature_size=1,
            )
            self._cached_hf_processor = OmniASRProcessor(
                feature_extractor, self.ctx.tokenizer
            )
        return self._cached_hf_processor

    def get_feature_extractor(self, **kwargs: object) -> OmniASRFeatureExtractor:
        feature_extractor = self.get_hf_processor(**kwargs).feature_extractor
        assert isinstance(feature_extractor, OmniASRFeatureExtractor)
        return feature_extractor


class OmniASRMultiModalProcessor(BaseMultiModalProcessor[OmniASRProcessingInfo]):
    """
    Multi-modal processor for the OmniASR model.

    Extends BaseMultiModalProcessor to handle audio modality inputs
    and coordinate prompt updates for speech-to-text tasks.
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.get("audios", [])
        if not audios:
            prompt_ids = self.info.get_tokenizer().encode(
                prompt, add_special_tokens=False
            )
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]))
        hf_config = self.info.get_hf_config()
        mm_data = dict(mm_data)
        mm_data["audio"] = mm_data.pop("audios")
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=hf_config.sampling_rate,
        )
        language = mm_kwargs.pop("language", None)
        result = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        lang_id = _resolve_lang_id(
            language,
            self.info.ctx.model_config.model,
            hf_config.n_special_tokens,
        )
        result["language_id"] = torch.tensor([lang_id], dtype=torch.long)
        return result

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            language_id=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        pad_token_id = self.info.get_hf_config().pad_token_id

        def get_audio_replacement_omniasr(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            num_tokens = self.info.get_num_audio_tokens(num_samples=audio_len)
            # N audio frames + 1 lid marker + 1 lang_id
            return [pad_token_id] * (num_tokens + 2)

        return [
            PromptReplacement(
                modality="audio",
                target=[pad_token_id],
                replacement=get_audio_replacement_omniasr,
            )
        ]


class OmniASRDummyInputsBuilder(BaseDummyInputsBuilder[OmniASRProcessingInfo]):
    """
    Dummy input builder for OmniASR.

    Generates synthetic audio and text data for testing and
    initialization purposes.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<pad>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        sampling_rate = self.info.get_hf_config().sampling_rate
        audio_len = sampling_rate * MAX_AUDIO_CLIP_S
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


@MULTIMODAL_REGISTRY.register_processor(
    OmniASRMultiModalProcessor,
    info=OmniASRProcessingInfo,
    dummy_inputs=OmniASRDummyInputsBuilder,
)
class OmniAsrForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    """
    OmniASR model for conditional generation (speech-to-text).

    This model integrates a Wav2Vec2-based audio tower with a LLaMA-based
    language model for generating transcriptions from audio input.
    """

    supports_transcription_only = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    def __init__(self, *, vllm_config=None, prefix: str = ""):
        super().__init__()
        config: OmniASRConfig = vllm_config.model_config.hf_config
        self.config = config
        if config.pad_token_id != PAD_TOKEN_ID:
            raise ValueError(
                f"OmniASR requires pad_token_id={PAD_TOKEN_ID}, "
                f"got {config.pad_token_id}"
            )
        if config.bos_token_id != BOS_TOKEN_ID:
            raise ValueError(
                f"OmniASR requires bos_token_id={BOS_TOKEN_ID}, "
                f"got {config.bos_token_id}"
            )
        self.model_id = vllm_config.model_config.model
        quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, "audio"):
            self.model = OmniASRModel(config)
        self.final_proj = ParallelLMHead(
            config.target_vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )
        self.logits_processor = LogitsProcessor(
            config.target_vocab_size, config.target_vocab_size
        )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["LlamaForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        _load_lang_table(self.model_id)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.pop("input_features", None)
        language_id = kwargs.pop("language_id", None)
        if input_features is None:
            return []
        if isinstance(input_features, list):
            audio_embs = []
            for feat in input_features:
                out = self.model.forward(feat.unsqueeze(0))
                audio_embs.append(out.squeeze(0))
        else:
            encoder_out = self.model.forward(input_features)
            audio_embs = list(encoder_out.unbind(dim=0))

        if not audio_embs:
            return []
        device = audio_embs[0].device
        dtype = audio_embs[0].dtype
        lid_emb = self.language_model.model.embed_tokens(
            torch.tensor([self.config.lid_marker_token_id], device=device)
        ).to(dtype)
        if language_id is not None:
            lang_id = language_id.flatten()[0].item()
        else:
            lang_id = _resolve_lang_id(
                None, self.model_id, self.config.n_special_tokens
            )
        lang_emb = self.model.lang_embeddings(
            torch.tensor([lang_id], device=device)
        ).to(dtype)
        lid_lang_embs = torch.cat([lid_emb, lang_emb], dim=0)
        results = []
        for audio_emb in audio_embs:
            combined = torch.cat([audio_emb, lid_lang_embs], dim=0)
            results.append(combined)
        return tuple(results)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.final_proj, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def transform(inputs):
            name, loaded_weight = inputs

            if name.startswith("llama_decoder.layer_norm"):
                name = name.replace(
                    "llama_decoder.layer_norm", "language_model.model.norm"
                )
            elif name.startswith("llama_decoder."):
                name = name.replace("llama_decoder.", "language_model.model.")
                name = name.replace(".self_attn_layer_norm", ".input_layernorm")
                name = name.replace(".ffn_layer_norm", ".post_attention_layernorm")
                name = name.replace(".self_attn.output_proj", ".self_attn.o_proj")
                name = name.replace(".ffn.inner_proj", ".mlp.up_proj")
                name = name.replace(".ffn.output_proj", ".mlp.down_proj")
                name = name.replace(".ffn.gate_proj", ".mlp.gate_proj")

                if ".self_attn.q_proj" in name:
                    num_heads = self.config.text_config.num_attention_heads
                    loaded_weight = _permute_q_k_for_neox(loaded_weight, num_heads)
                elif ".self_attn.k_proj" in name:
                    num_heads = self.config.text_config.num_key_value_heads
                    loaded_weight = _permute_q_k_for_neox(loaded_weight, num_heads)

            elif name.startswith("text_frontend"):
                name = name.replace(
                    "text_frontend", "language_model.model.embed_tokens"
                )
            elif name.startswith("final_proj"):
                # final_proj is top-level on this class, no prefix needed
                pass
            else:
                name = "model." + name

            return name, loaded_weight

        loader = AutoWeightsLoader(self)
        result = loader.load_weights(map(transform, weights))
        return result

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        sampling_rate = model_config.hf_config.sampling_rate
        if sampling_rate != SAMPLING_RATE_HZ:
            raise ValueError(
                f"OmniASR requires 16kHz sampling rate (architecture constraint), "
                f"got {sampling_rate}"
            )
        return SpeechToTextConfig(
            max_audio_clip_s=MAX_AUDIO_CLIP_S,
            sample_rate=sampling_rate,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        # OmniASR uses a separate lang_embeddings layer (not text tokens).
        audio = stt_params.audio
        stt_config = stt_params.stt_config
        language = stt_params.language
        mm_kwargs = {"language": language} if language else {}
        return TokensPrompt(
            prompt_token_ids=[PAD_TOKEN_ID, BOS_TOKEN_ID],
            multi_modal_data={"audio": (audio, stt_config.sample_rate)},
            mm_processor_kwargs=mm_kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")
