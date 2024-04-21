from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import VisionLanguageConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.phi import PhiForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


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

        # TODO: Replace with VLLM attention implementation after adding support
        # for acasual attention.
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


class Moondream(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        vision_language_config: VisionLanguageConfig,
        linear_method: Optional["LinearMethodBase"] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.vision_language_config = vision_language_config

        assert self.vision_language_config, (
            "Provide `image_input_type` and other vision "
            "related configurations through LLM entrypoint "
            "or engine arguments.")

        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_encoder = VisionEncoder()
        else:
            self.vision_encoder = None

        self.linear_method = linear_method

        self.text_model = PhiForCausalLM(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        image_input: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        if image_input is not None:
            if list(image_input.shape[1:]) != list(
                    self.vision_language_config.image_input_shape[1:]):
                raise ValueError(
                    f"The expected image tensor shape is batch dimension "
                    f"plus "
                    f"{self.vision_language_config.image_input_shape[1:]}."
                    f" You supplied {image_input.shape}. "
                    f"If you are using vLLM's entrypoint, make sure your "
                    f"supplied image input is consistent with "
                    f"image_input_shape in engine args.")

            if self.vision_encoder is not None:
                image_features = self.vision_encoder(image_input)
            else:
                image_features = image_input

            inputs_embeds = self.text_model.model.embed_tokens(input_ids)
            mask = input_ids == self.vision_language_config.image_token_id
            inputs_embeds[mask] = image_features.view(
                -1,
                image_features.shape[-1],
            )
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
