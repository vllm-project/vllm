import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, cast


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, BatchFeature
from vllm.inputs.data import PromptType
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.config.multimodal import BaseDummyOptions
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsTranscription, MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)

from vllm.multimodal.processing import BaseDummyInputsBuilder
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS,
)
from vllm.model_executor.layers.attention import (
    Attention,
    CrossAttention,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.v1.attention.backend import (
    AttentionType,
)

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.utils.jsontree import json_map_leaves
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.model_executor.models.fireredasr import Conv2dSubsampling, RelPositionalEncoding, RelPosEmbConformerBlock
from .utils import make_layers

class FireRedASRAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames (M)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]


class FireRedAsrAedConformerEncoder(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = ""
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.input_preprocessor = Conv2dSubsampling(config.num_mel_bins, config.d_model)
        self.positional_encoding = RelPositionalEncoding(config.d_model)
        self.dropout = nn.Dropout(config.residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(config.encoder_layers):
            block = RelPosEmbConformerBlock(config.d_model, config.encoder_attention_heads,
                        config.kernel_size)
            self.layer_stack.append(block)

    def forward(self, input_features):
        input_lengths = torch.tensor([input_features.size(1)], device=input_features.device)
        padded_input_features = F.pad(input_features, (0, 0, 0, self.input_preprocessor.context - 1), 'constant', 0.0)
        src_mask = self.padding_position_is_0(padded_input_features, input_lengths)
        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input_features, src_mask)
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                   pad_mask=src_mask)
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        B, T, _ = padded_input.shape
        pos = torch.arange(T, device=input_lengths.device).unsqueeze(0)
        mask = pos < input_lengths.unsqueeze(1)
        return mask.unsqueeze(1).float()



class TransformerDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = getattr(config, 'pad_token_id', config.pad_id)
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # Token embedding: checkpoint uses tgt_word_emb
        self.tgt_word_emb = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )
        # Positional encoding: checkpoint uses positional_encoding with buffer 'pe'
        # Use pe_maxlen (5000) from config, not max_target_positions (448)
        pe_maxlen = getattr(config, 'pe_maxlen', 5000)
        self.positional_encoding = PositionalEncoding(
            config.d_model, pe_maxlen
        )
        # Decoder layers: checkpoint uses layer_stack
        self.start_layer, self.end_layer, self.layer_stack = make_layers(
            config.decoder_layers,
            lambda prefix: DecoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}"
            ),
            prefix=f"{prefix}.layer_stack",
        )
        # Final layer norm: checkpoint uses layer_norm_out
        self.layer_norm_out = nn.LayerNorm(config.d_model)
        # Output projection: checkpoint uses tgt_word_prj
        self.tgt_word_prj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        inputs_embeds = self.embed_input_ids(input_ids)
        pos_emb = self.positional_encoding(positions)
        hidden_states = inputs_embeds * self.embed_scale + pos_emb
        for i, decoder_layer in enumerate(self.layer_stack):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.layer_norm_out(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tgt_word_emb(input_ids)


class DecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Self attention: checkpoint uses self_attn with w_qs/w_ks/w_vs/fc
        self.self_attn = DecoderSelfAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # Checkpoint uses self_attn_norm (not self_attn_layer_norm)
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        
        # Cross attention: checkpoint uses cross_attn (not encoder_attn)
        self.cross_attn = DecoderCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        # Checkpoint uses cross_attn_norm (not encoder_attn_layer_norm)
        self.cross_attn_norm = nn.LayerNorm(config.d_model)
        
        # MLP: checkpoint uses mlp with w_1/w_2
        self.mlp = MLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Checkpoint uses mlp_norm (not final_layer_norm)
        self.mlp_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class DecoderSelfAttention(nn.Module):
    """Decoder self-attention with separate w_qs/w_ks/w_vs/fc to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        per_layer_sliding_window: int | None = None,
        block_pool_size: int = 1,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # Checkpoint uses separate w_qs, w_ks, w_vs (not qkv_proj)
        # Note: w_ks has no bias in checkpoint
        self.w_qs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_qs",
        )
        self.w_ks = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=False,  # No bias for w_ks in checkpoint
            quant_config=quant_config,
            prefix=f"{prefix}.w_ks",
        )
        self.w_vs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_vs",
        )
        # Checkpoint uses fc (not out_proj)
        self.fc = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
            per_layer_sliding_window=per_layer_sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        q, _ = self.w_qs(hidden_states)
        k, _ = self.w_ks(hidden_states)
        v, _ = self.w_vs(hidden_states)

        attn_output = self.attn(q, k, v)

        output, _ = self.fc(attn_output)

        return output


class DecoderCrossAttention(nn.Module):
    """Decoder cross-attention with separate w_qs/w_ks/w_vs/fc to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = AttentionType.ENCODER_DECODER

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # Checkpoint uses w_qs for query projection
        self.w_qs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_qs",
        )
        # Checkpoint uses w_ks (no bias) and w_vs for key/value
        self.w_ks = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=False,  # No bias for w_ks in checkpoint
            quant_config=quant_config,
            prefix=f"{prefix}.w_ks",
        )
        self.w_vs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_vs",
        )
        # Checkpoint uses fc (not out_proj)
        self.fc = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        # Use vLLM's CrossAttention for KV cache support
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        q, _ = self.w_qs(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            k, _ = self.w_ks(encoder_hidden_states)
            v, _ = self.w_vs(encoder_hidden_states)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output, _ = self.fc(attn_output)

        return output


class MLP(nn.Module):
    """MLP with w_1/w_2 naming to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.activation_fn = get_act_fn(act_fn)
        # Checkpoint uses w_1 (not fc1)
        self.w_1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.w_1",
        )
        # Checkpoint uses w_2 (not fc2)
        self.w_2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.w_2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.w_1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.w_2(hidden_states)
        return hidden_states


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with 'pe' buffer to match checkpoint."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings for given positions.
        
        Args:
            positions: Position indices [seq_len] or [batch, seq_len]
        Returns:
            Positional embeddings with same shape + d_model
        """
        # Handle both 1D and 2D position inputs
        if positions.dim() == 1:
            # [seq_len] -> [seq_len, d_model]
            return self.pe[0, positions]
        else:
            # [batch, seq_len] -> [batch, seq_len, d_model]
            return self.pe[0, positions]


class FireRedASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = FireRedAsrAedConformerEncoder(
            vllm_config=vllm_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = TransformerDecoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        enc_states = torch.cat(encoder_outputs, dim=0) if len(encoder_outputs) else None
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )
        return decoder_outputs


class FireRedASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config(PretrainedConfig)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_target_channels(self) -> int:
        return 1

    def get_data_parser(self) -> MultiModalDataParser:
        """Override to provide target_sr for audio resampling."""
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions

    def get_audio_token_id(self) -> int:
        """获取 <|AUDIO|> 占位符的 token id"""
        hf_processor = self.get_hf_processor()
        return hf_processor.audio_token_id


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class FireRedASRMultiModalProcessor(EncDecMultiModalProcessor[FireRedASRProcessingInfo]):
    """MultiModal processor for FireRedASR encoder-decoder model.
    
    Inherits from EncDecMultiModalProcessor to properly handle
    encoder-decoder architecture (like Whisper).
    """
    
    def build_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.info.get_target_channels(),
        )

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        # For encoder-decoder models, encoder only accepts audio features.
        # Return a dummy encoder prompt which will be padded to num_audio_tokens.
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            speech_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        # Use [0] as target to match the dummy encoder prompt created in create_encoder_prompt
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder,
)
class FireRedAsrAedLForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    """
    FireRedASR model adapted for vLLM
    
    This follows the same pattern as WhisperForConditionalGeneration in vLLM.
    Key requirements for vLLM compatibility:
    1. __init__(vllm_config, prefix) signature
    2. forward(input_ids, positions, encoder_outputs) signature
    3. Implement: embed_multimodal, embed_input_ids, compute_logits
    4. Set supports_transcription_only = True
    """

    # Mark as transcription-only model (like Whisper)
    supports_transcription_only = True
    supports_segment_timestamp = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,  # not needed here
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if language is None:
            raise ValueError(
                "Language must be specified when creating the Whisper prompt"
            )
        prompt = {
            "encoder_prompt": {
                # Whisper does not support encoder prompt.
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": ""
        }
        return cast(PromptType, prompt)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        
        # Create encoder and decoder
        self.model = FireRedASRModel(vllm_config=vllm_config, prefix=prefix)

        # Logits processor
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for vLLM.
        
        Args:
            input_ids: Decoder input token IDs [batch, seq_len]
            positions: Position IDs for decoder [batch, seq_len]
            encoder_outputs: Pre-computed encoder outputs (list of tensors)

        Returns:
            Decoder hidden states
        """
        if encoder_outputs is None:
            encoder_outputs = []

        decoder_outputs = self.model(input_ids, positions, encoder_outputs)
        return decoder_outputs

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        # Encoder returns (enc_output, input_lengths, src_mask)
        # We only need enc_output for multimodal embeddings

        enc_output, _, _ = self.model.encoder(audio_input["input_features"])
        return enc_output.unbind(dim=0)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FireRedASRAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)

        input_features = input_features.transpose(1, 2)
        return FireRedASRAudioInputs(input_features=input_features)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return multimodal_embeddings

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits from hidden states.
        Required for VllmModelForTextGeneration interface.
        This method is critical - its presence is checked to determine
        if the model is a text generation model.
        """
        # tgt_word_prj is a standard nn.Linear, so we apply it directly
        # instead of using logits_processor which expects quant_method
        logits = self.model.decoder.tgt_word_prj(hidden_states)
        return logits
    
    def load_weights(self, weights):
        """Load model weights from checkpoint.

        The checkpoint uses FireRedASR naming convention which matches
        our model structure exactly.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params = set()

        # Mapping from Sequential index to named module for FFN layers
        # Sequential: 0=pre_layer_norm, 1=linear_expand, 4=linear_project
        ffn_net_mapping = {
            '.net.0.': '.pre_layer_norm.',
            '.net.1.': '.linear_expand.',
            '.net.4.': '.linear_project.',
        }

        for name, loaded_weight in weights:
            # Remap FFN Sequential indices to named modules
            # e.g., ffn1.net.0.weight -> ffn1.pre_layer_norm.weight
            for old_pattern, new_pattern in ffn_net_mapping.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
                    break

            # Handle parameters
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)
                loaded_params.add(name)
            # Handle buffers (like positional_encoding.pe)
            elif name in buffers_dict:
                buffers_dict[name].data.copy_(loaded_weight)
                loaded_params.add(name)

        return loaded_params
