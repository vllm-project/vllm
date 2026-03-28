# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn
import math
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.quantization import QuantizationConfig
from typing import Iterable
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

class OmniASRModel(nn.Module):
    """Full OmniASR: encoder + projection + LLaMA decoder.
    
    TODO: Integrate with vLLM's LlamaForCausalLM for decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder_frontend = Wav2Vec2Frontend()
        self.encoder = Wav2Vec2TransformerEncoder()
        # TODO: replace hard-coded dimensions with config
        self.encoder_proj = ColumnParallelLinear(1024, 4096, bias=True)
        # TODO: Replace with vLLM's LlamaForCausalLM
        # self.language_model = LlamaForCausalLM(vllm_config)
        self.text_frontend = VocabParallelEmbedding(9813, 4096)
        self.lang_embeddings = VocabParallelEmbedding(1694, 4096)
        self.final_proj = ColumnParallelLinear(4096, 9812, bias=False)

    def forward(self, audio):
        x = self.encoder_frontend(audio)
        x = self.encoder(x)
        x, _ = self.encoder_proj(x)
        return x  # [batch, seq, 4096] ready for LLaMA decoder
    
    def load_weights(self, weights:Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
                (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
                (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
                (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
                ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            #TODO:llama decoder implementation
            if (name.startswith("llama_decoder")):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                #replace weight name with param name
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
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        #TODO:Remove hard-coded numbers with acutal config
        self.sampling_rate = 16000
        self.subsampling_factor = 320
        in_channels = [1, 512, 512, 512, 512, 512, 512]
        kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        for i in range(7):
            conv = nn.Conv1d(in_channels[i], 512, kernel_sizes[i], stride=strides[i], bias=True)
            layer_norm = nn.LayerNorm(512)
            self.layers.append(nn.ModuleDict({"conv": conv, "layer_norm": layer_norm}))

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
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_heads // tp_size
        self.head_dim = embed_dim // self.total_num_heads
        self.scaling = self.head_dim ** -0.5
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
    def __init__(self, embed_dim: int=1024, ffn_dim: int=4096, quant_config: QuantizationConfig | None = None,):
        super().__init__()
        self.inner_proj = ColumnParallelLinear(
            input_size=embed_dim, output_size=ffn_dim, bias=True, quant_config=quant_config,
        )
        self.output_proj = RowParallelLinear(
            input_size=ffn_dim, output_size=embed_dim, bias=True, quant_config=quant_config,
        )

    def forward(self, x):
        x, _ = self.inner_proj(x)
        x = nn.functional.gelu(x)
        x, _ = self.output_proj(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, embed_dim=1024, ffn_dim=4096, num_heads=16):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = Wav2Vec2Attention(embed_dim, num_heads)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = Wav2Vec2FFN(embed_dim, ffn_dim)

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
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.post_extract_layer_norm = nn.LayerNorm(512)
        self.model_dim_proj = nn.Linear(512, 1024, bias=True)
        # pos_encoder: store as plain conv, handle weight_norm in weight loading
        self.pos_encoder = nn.ModuleDict({
            "conv": nn.utils.weight_norm(
                nn.Conv1d(1024, 1024, kernel_size=128, padding=64, groups=16, bias=True),
                name="weight",
                dim=2
            )
        })

    def forward(self, audio):
        x = self.feature_extractor(audio)
        x = x.transpose(1, 2)
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)
        pos = self.pos_encoder["conv"](x.transpose(1, 2))
        pos = pos[:, :, :x.shape[1]]
        x = x + pos.transpose(1, 2)
        return x

class Wav2Vec2TransformerEncoder(nn.Module):
    """encoder.* keys"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Wav2Vec2EncoderLayer(embed_dim=1024, ffn_dim=4096, num_heads=16)
            for _ in range(24)
        ])
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x

# TODO: Add multimodal registry once OmniASR config/processor are defined
# @MULTIMODAL_REGISTRY.register_processor(...)

class OmniAsrForConditionalGeneration(nn.Module):
    """OmniASR: Wav2Vec2 encoder + projection + LLaMA decoder.
    
    TODO:
    - Add HuggingFace config class
    - Implement SupportsTranscription, SupportsMultiModal interfaces
    - Add multimodal processor and dummy inputs
    - Integrate LLaMA decoder via vLLM's LlamaForCausalLM
    """
    
    def __init__(self, *, vllm_config=None, prefix: str = ""):
        super().__init__()
        # TODO: read dimensions from vllm_config.model_config.hf_config

class OmniASRProcessingInfo(BaseProcessingInfo):
    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)
    def get_supported_mm_limits(self) -> Mapping[str, int|None]:
        return {"audio":1}
    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)
    def get_feature_extractor(self):
        return Wav2Vec2FeatureExtractor()
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
    def _call_hf_processor(self, prompt, mm_data:Mapping[str, object], mm_kwargs:Mapping[str, object], tok_kwargs:Mapping[str, object]):
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
        import torch
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
