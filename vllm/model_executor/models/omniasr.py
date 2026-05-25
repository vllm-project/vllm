# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping, Sequence

import torch
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
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.omniasr import OmniASRConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .whisper import ISO639_1_SUPPORTED_LANGS

MAX_AUDIO_CLIP_S = 40


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
        #self.text_frontend = VocabParallelEmbedding(
        #    config.target_vocab_size + config.n_special_tokens,
        #    config.text_config.hidden_size,
        #)
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

    def get_seq_len(self, num_samples: int) -> int:
        """
        Compute output sequence length for a given number of input samples.

        Args:
            num_samples: Number of input audio samples.

        Returns:
            Resulting sequence length after convolution layers.
        """
        seq_len = num_samples
        for layer in self.layers:
            conv = layer["conv"]
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            dilation = conv.dilation[0]
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
        x = audio.to(
            self.feature_extractor.layers[0]["conv"].weight.dtype
        )  # add dtype conversion
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
    including sequence length computation and feature extractor access.
    """

    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def get_feature_extractor(self) -> Wav2Vec2FeatureExtractor:
        config = self.get_hf_config()
        return Wav2Vec2FeatureExtractor(config)

    def get_num_audio_tokens(self, num_samples: int) -> int:
        return self.get_feature_extractor().get_seq_len(num_samples)


class OmniASRMultiModalProcessor(EncDecMultiModalProcessor):
    """
    Multi-modal processor for the OmniASR model.

    Extends EncDecMultiModalProcessor to handle audio modality inputs
    and coordinate prompt updates for speech-to-text tasks.
    """

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def _call_hf_processor(
        self,
        prompt,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        # TODO: Implement custom audio processing for fairseq2-based OmniASR
        # Cannot use HF processor — need to:
        # 1. Accept raw audio waveform from mm_data
        # 2. Return input_features tensor for Wav2Vec2Frontend
        # 3. Return length tensor for sequence tracking
        #
        # Expected return format:
        # {
        #     "input_features": processed audio tensor,
        #     "length": tensor of audio lengths,
        #     "input_ids": tokenized prompt,
        # }
        audios = mm_data.get("audios", [])
        if isinstance(audios, list) and len(audios) > 0:
            features = [torch.tensor(a, dtype=torch.float32) for a in audios]
            lengths = torch.tensor([f.shape[-1] for f in features])
        else:
            features = torch.zeros(1, 1, 16000)  # placeholder
            lengths = torch.tensor([16000])

        return {
            "input_features": features,
            "length": lengths,
            "input_ids": [0],
        }

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ):
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            length=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_audio_replacement_omniasr(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            num_tokens = self.info.get_num_audio_tokens(num_samples=audio_len)
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
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
        return "<s>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.pop("input_features", None)
        if input_features is None:
            return []
        if isinstance(input_features, list):
            results = []
            for feat in input_features:
                out = self.model.forward(feat.unsqueeze(0))
                results.append(out.squeeze(0))
            return tuple(results)
        encoder_out = self.model.forward(input_features)
        return encoder_out.unbind(dim=0)

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
            elif name.startswith("text_frontend"):
               name = name.replace(
                   "text_frontend", "language_model.model.embed_tokens"
               )
            elif name.startswith("final_proj"):
                pass
            else:
                name = "model." + name

            return name, loaded_weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(map(transform, weights))

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        sampling_rate = model_config.hf_config.sampling_rate
        assert sampling_rate == 16000
        return SpeechToTextConfig(
            max_audio_clip_s=MAX_AUDIO_CLIP_S,
            sample_rate=sampling_rate,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        # TODO: Add language conditioning support.
        # OmniASR uses a separate lang_embeddings layer (not text tokens).
        # Need to: map ISO 639-1 → OmniASR lang_id, pass through
        # lang_embeddings, and insert into decoder input sequence.
        # Currently uses audio-only mode (valid for 20% of training data).
        audio = stt_params.audio
        stt_config = stt_params.stt_config
        return TokensPrompt(
            prompt_token_ids=[0],
            multi_modal_data={"audio": (audio, stt_config.sample_rate)},
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")
