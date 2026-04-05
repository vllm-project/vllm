# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
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
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
)
from vllm.transformers_utils.configs.omniasr import OmniASRConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)


class OmniASRModel(nn.Module):
    """Full OmniASR: encoder + projection + LLaMA decoder.

    TODO: Integrate with vLLM's LlamaForCausalLM for decoder.
    """

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.encoder_frontend = Wav2Vec2Frontend(config)
        self.encoder = Wav2Vec2TransformerEncoder(config)
        self.encoder_proj = ColumnParallelLinear(
            config.encoder_embed_dim, config.projection_dim, bias=True
        )
        # TODO: Replace with vLLM's LlamaForCausalLM
        # self.language_model = LlamaForCausalLM(vllm_config)
        self.text_frontend = VocabParallelEmbedding(
            config.target_vocab_size + config.n_special_tokens,
            config.text_config.hidden_size,
        )
        self.lang_embeddings = VocabParallelEmbedding(
            config.num_languages, config.text_config.hidden_size
        )

    def forward(self, audio):
        x = self.encoder_frontend(audio)
        x = self.encoder(x)
        x, _ = self.encoder_proj(x)
        return x  # [batch, seq, config.projection_dim] ready for LLaMA decoder

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            # TODO:llama decoder implementation
            if name.startswith("llama_decoder"):
                continue
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
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.sampling_rate = config.sampling_rate
        self.subsampling_factor = config.subsampling_factor

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

    def forward(self, x):
        for layer in self.layers:
            x = layer["conv"](x)
            x = x.transpose(1, 2)
            x = layer["layer_norm"](x)
            x = x.transpose(1, 2)
            x = nn.functional.gelu(x)
        return x


class Wav2Vec2Attention(nn.Module):
    """Self-attention with separate q/k/v/output projections (matching checkpoint)"""

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

    def forward(self, x):
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.output_proj(attn_output)
        return output


class Wav2Vec2FFN(nn.Module):
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

    def forward(self, x):
        x, _ = self.inner_proj(x)
        x = nn.functional.gelu(x)
        x, _ = self.output_proj(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        embed_dim = config.encoder_embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = Wav2Vec2Attention(config)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = Wav2Vec2FFN(config)

    def forward(self, x):
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

    def forward(self, audio):
        x = self.feature_extractor(audio)
        x = x.transpose(1, 2)
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)
        pos = self.pos_encoder["conv"](x.transpose(1, 2))
        pos = pos[:, :, : x.shape[1]]
        x = x + pos.transpose(1, 2)
        return x


class Wav2Vec2TransformerEncoder(nn.Module):
    """encoder.* keys"""

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.encoder_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class OmniASRProcessingInfo(BaseProcessingInfo):
    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def get_feature_extractor(self):
        config = self.ctx.get_hf_config()
        return Wav2Vec2FeatureExtractor(config)

    def get_num_audio_tokens(self, num_samples):
        fe = self.get_feature_extractor()
        return num_samples // fe.subsampling_factor


class OmniASRMultiModalProcessor(EncDecMultiModalProcessor):
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

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            length=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        # TODO: implement proper prompt replacement
        return []


class OmniASRDummyInputsBuilder(BaseDummyInputsBuilder[OmniASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
        mm_processor_kwargs=None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = sampling_rate * 30  # 30 seconds max
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
    """OmniASR: Wav2Vec2 encoder + projection + LLaMA decoder.

    TODO:
    - Integrate LLaMA decoder via vLLM's LlamaForCausalLM
    """

    def __init__(self, *, vllm_config=None, prefix: str = ""):
        super().__init__()
        config: OmniASRConfig = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
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

    def get_encoder_outputs(
        self, audio: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.model.forward(audio)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.pop("input_features", None)
        length = kwargs.pop("length", None)
        return self.get_encoder_outputs(input_features, length)

    def get_language_model(self) -> nn.Module:
        # TODO: return self.model.language_model once LLaMA is integrated
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: integrate LLaMA decoder
        pass

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.final_proj, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)
