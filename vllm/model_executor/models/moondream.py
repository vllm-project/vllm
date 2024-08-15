from typing import Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
from PIL import Image
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.phi import PhiForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData

from .interfaces import SupportsVision
from .utils import merge_vision_embeddings

MOONDREAM_IMAGE_TOKEN = "<image>"

class Attention(nn.Module):

    def __init__(self, dim, num_heads=16):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class VitBlock(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Attention(embed_dim)
        self.mlp = MLP(embed_dim, 4304)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        embed_len = 729
        embed_dim = 1152

        self.patch_embed = LinearPatchEmbedding()
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, embed_dim))
        self.blocks = nn.Sequential(*[VitBlock(embed_dim) for _ in range(27)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class EncoderWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({"visual": VisionTransformer()})

    def forward(self, x):
        return self.model["visual"](x)


class LinearPatchEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(588, 1152)

    def forward(self, x):
        b, c, hp1, wp2 = x.shape
        p1, p2 = 14, 14
        h, w = hp1 // p1, wp2 // p2
        x = x.reshape(b, c, h, p1, w, p2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, h * w, c * p1 * p2)

        return self.linear(x)


class MLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionProjection(nn.Module):

    def __init__(self):
        super().__init__()

        image_embedding_dim = 1152
        model_dim = 2048
        hidden_dim = model_dim * 4

        self.mlp = MLP(image_embedding_dim, hidden_dim, model_dim)

    def forward(self, x):
        return self.mlp(x)


class VisionEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = EncoderWrapper()
        self.projection = VisionProjection()

    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = self.projection(x)
        return x


class MoondreamImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channels, height, width)"""


MoondreamImageInputs = MoondreamImagePixelInputs

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens()
@INPUT_REGISTRY.register_dummy_data()
@INPUT_REGISTRY.register_input_processor()
class Moondream(nn.Module, SupportsVision):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = VisionEncoder(
            use_flash_attn=config._attn_implementation == "flash_attention_2"
        )
        self.text_model = PhiForCausalLM(config=config.text_config, 
                                         cache_config=cache_config, 
                                         quant_config=quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object
    ) -> SamplerOutput:
        
        parsed_image_input = self._parse_and_validate_image_input(**kwargs)

        if parsed_image_input is not None:
            vision_embeddings = self._process_image_input(parsed_image_input)

            inputs_embeds = self.text_model.model.embed_tokens(input_ids)

            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.config.image_token_index)

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.text_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.text_model.compute_logits(hidden_states, sampling_metadata)

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        return self.text_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            param = None

            if name in params_dict:
                param = params_dict[name]
            elif name.startswith("text_model."):
                replacements = {
                    "text_model.transformer.h": "text_model.model.layers",
                    "lm_head.ln": "model.final_layernorm",
                    "ln": "input_layernorm",
                    "mixer.Wqkv": "self_attn.qkv_proj",
                    "mixer.out_proj": "self_attn.dense",
                    "lm_head.linear": "lm_head",
                    "transformer.embd.wte": "model.embed_tokens",
                }

                mp = name
                for k, v in replacements.items():
                    if k in mp:
                        mp = mp.replace(k, v)
                if mp in params_dict:
                    param = params_dict[mp]

            if param is None:
                raise ValueError(f"Unmapped weight: {name}")
            else:
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
