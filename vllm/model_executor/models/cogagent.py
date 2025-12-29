# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    TypeAlias,
    overload,
)

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.cogagent_processing import (
    CogAgentDummyInputsBuilder,
    CogAgentImageEmbeddingInputs,
    CogAgentImagePixelInputs,
    CogAgentMultiModalProcessor,
    CogAgentProcessingInfo,
    get_max_image_tokens,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY

ImageData: TypeAlias = Any
if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from vllm.transformers_utils.configs.cogagent import CogAgentConfig

    ImageData: TypeAlias = np.ndarray | torch.Tensor | Image.Image

from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from .cogagent_vision_encoder import (
    CrossVisionModel,
    EVA2CLIPModel,
    sharded_weight_loader,
)
from .interfaces import SupportsLoRA, SupportsMultiModal
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)


def build_positions(scheduler_config, num_image_tokens, device):
    """prebuilt position ids to overwrite passed in ids."""
    # we make two assumptions here.
    # 1. There is no text before the image.
    #       - This is enforced via our processor.
    # 2. All prompts had an image during prefill.

    max_position_length = scheduler_config.max_num_batched_tokens
    position_ids = [0, 1]
    position_ids += [2] * num_image_tokens
    position_ids += list(range(3, (max_position_length + 3) - len(position_ids)))
    position_ids = torch.tensor(position_ids, dtype=torch.int64, device=device)

    return position_ids


class MLP(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.act_and_mul = get_act_and_mul_fn(config.hidden_act)  # SiLU default

    def forward(self, x):  # HD, HD
        x, no_bias = self.gate_up_proj(x)
        x = self.act_and_mul(x)
        x, no_bias = self.down_proj(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        weights_mapper = {
            "gate_proj.weight": ("gate_proj", "gate_up_proj", 0),
            "up_proj.weight": ("up_proj", "gate_up_proj", 1),
        }

        loaded_params = sharded_weight_loader(
            params_dict=params_dict, weights=weights, weights_mapper=weights_mapper
        )

        return loaded_params


class VisionExpertMLP(nn.Module):
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ):
        super().__init__()

        self.language_mlp = MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.language_mlp"
        )
        self.vision_mlp = MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.vision_mlp"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        language_token_ids: torch.Tensor,
        vision_token_ids: torch.Tensor,
    ):
        if vision_token_ids is None:
            output = self.language_mlp(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            output[vision_token_ids] = self.vision_mlp(hidden_states[vision_token_ids])
            output[language_token_ids] = self.language_mlp(
                hidden_states[language_token_ids]
            )

        return output


class VisionExpertAttention(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = divide(self.num_heads, self.tp_size)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=self.max_position_embeddings,
            is_neox_style=True,
            dtype=config.dtype
        )

        self.vision_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_expert_query_key_value",
        )

        self.vision_expert_dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_expert_dense",
        )

        self.language_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.language_expert_query_key_value",
        )

        self.language_expert_dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.language_expert_dense",
        )

        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_local_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.LongTensor,
        hidden_states: torch.Tensor,
        language_token_ids: torch.BoolTensor | None,  # 1D
        vision_token_ids: torch.BoolTensor | None,  # 1D
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # we don't expect only image tokens.
        # the bos token, the boi, and the eoi will always be text.
        if language_token_ids is None or vision_token_ids is None:
            mixed_raw_layer, bias = self.language_expert_query_key_value(hidden_states)
        elif self.tp_size <= 1:
            # expects num_tokens, hidden_size
            mixed_raw_layer = hidden_states.new_zeros(
                hidden_states.shape[-2], hidden_states.shape[-1] * 3
            )
            mixed_raw_layer[vision_token_ids], bias = (
                self.vision_expert_query_key_value(hidden_states[vision_token_ids])
            )
            mixed_raw_layer[language_token_ids], bias = (
                self.language_expert_query_key_value(hidden_states[language_token_ids])
            )
        else: # NOTE: Investigate performance loss for a unified graph
            vision_states, bias = self.vision_expert_query_key_value(
                hidden_states * vision_token_ids
            )
            language_states, bias = self.language_expert_query_key_value(
                hidden_states * language_token_ids
            )
            mixed_raw_layer = vision_states + language_states

        query_states, key_states, value_states = torch.split(
            mixed_raw_layer, self.hidden_size, dim=-1
        )

        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # context_layer -> [num_tokens, head * head_dim]
        context_layer = self.attn(query_states, key_states, value_states)

        if language_token_ids is None or vision_token_ids is None:
            attn_output, bias = self.language_expert_dense(hidden_states)
        elif self.tp_size <= 1:
            attn_output = torch.zeros_like(hidden_states)
            attn_output[vision_token_ids], bias = self.vision_expert_dense(
                context_layer[vision_token_ids]
            )
            attn_output[language_token_ids], bias = self.language_expert_dense(
                context_layer[language_token_ids]
            )
        else:
            vision_output, bias = self.vision_expert_dense(
                context_layer * vision_token_ids
            )
            language_output, bias = self.language_expert_dense(
                context_layer * language_token_ids
            )
            attn_output = vision_output + language_output

        return attn_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        loaded_params = set()
        for name, loaded_weight in weights:
            if "inv_freq" in name:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


class CogAgentCrossAttention(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size  # 4096
        self.cross_hidden_size = config.cross_hidden_size
        self.cross_compute_hidden_size = config.cross_compute_hidden_size

        self.num_heads = config.num_attention_heads
        self.cross_head_dim = (
            self.cross_compute_hidden_size // self.num_heads
        )  # default is 32
        self.max_position_embeddings = config.max_position_embeddings
        self.local_num_heads = self.num_heads // self.tp_size

        # query and key_value can have different head sizes,
        # so init them separately.
        self.query = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.cross_compute_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.query",
        )

        self.key_value = QKVParallelLinear(
            hidden_size=self.cross_hidden_size,
            head_size=self.cross_head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_value",
        )

        self.dense = RowParallelLinear(
            input_size=self.cross_compute_hidden_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.cross_attn = CrossAttention(
            self.local_num_heads,
            self.cross_head_dim,
            scale=self.cross_head_dim**-0.5,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )

    def forward(
        self, hidden_states: torch.Tensor, encoder_embeds: torch.FloatTensor | None
    ) -> torch.Tensor:
        query_states, no_bias = self.query(hidden_states)

        if encoder_embeds is None:
            key_states = None
            value_states = None
        else:
            encoder_states, no_bias = self.key_value(encoder_embeds)
            key_states, value_states = encoder_states.chunk(2, dim=-1)

        context_layer = self.cross_attn(
            query_states,
            key_states,
            value_states,
        )

        attn_output, no_bias = self.dense(context_layer)
        return attn_output


class CogAgentDecoderLayer(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size

        # NOTE: VisionExpertAttention and CrossAttention can
        # have different hidden sizes. Current implentation
        # relies on padding KV cache block sizes to match.

        self.self_attn = VisionExpertAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.cross_attn = CogAgentCrossAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        self.mlp = VisionExpertMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_cross_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.LongTensor,
        hidden_states: torch.Tensor,
        vision_token_ids: torch.Tensor,
        language_token_ids: torch.Tensor,
        encoder_embeds: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )

        hidden_states = residual + hidden_states
        cross_input = self.post_cross_attention_layernorm(hidden_states)
        
        attention_output = self.cross_attn(
            hidden_states=cross_input, encoder_embeds=encoder_embeds
        )

        hidden_states = hidden_states + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(
            mlp_input,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )

        hidden_states = mlp_output + hidden_states

        return hidden_states


class CogAgentModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hf_config = vllm_config.model_config.hf_config  # type: CogAgentConfig
        self.vocab_size = self.hf_config.vocab_size

        self.pad_token_id: int = self.hf_config.pad_token_id

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hf_config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=self.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList([
            CogAgentDecoderLayer(
                self.hf_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(self.hf_config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            self.hf_config.hidden_size,
            eps=self.hf_config.rms_norm_eps,
        )

        self.num_image_tokens = get_max_image_tokens(self.hf_config.vision_config)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hf_config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: torch.LongTensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.FloatTensor | None,
        encoder_embeds: torch.FloatTensor | None,
        vision_token_ids: torch.BoolTensor | torch.LongTensor | None,
        language_token_ids: torch.BoolTensor | torch.LongTensor | None,
    ):
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,  # L, D
                positions=positions,
                vision_token_ids=vision_token_ids,
                language_token_ids=language_token_ids,
                encoder_embeds=encoder_embeds,  # None or B, L, D
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    CogAgentMultiModalProcessor,
    info=CogAgentProcessingInfo,
    dummy_inputs=CogAgentDummyInputsBuilder,
)
class CogAgentForCausalLM(nn.Module, SupportsMultiModal, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_suffix={
            "linear_proj.gate_proj.weight": "linear_proj.glu_gate_proj.weight"
        },
        orig_to_new_prefix={
            "model.vision": "vision",
            "model.cross_vision": "cross_vision",
        },
    )
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "key_value": ["key_value"],
        "merged_proj": ["glu_gate_proj", "dense_h_to_4h"],
        "w1_2": ["w1", "w2"],
        "query_key_value": ["query_key_value"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    
    supported_lora_modules = [
        # EVA Large
        #"qkv_proj",
        # Decoder
        "language_expert_query_key_value",
        "vision_expert_query_key_value",
        "query",
        "key_value",
        # EVA small
        #"query_key_value",
    ]

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    requires_raw_input_tokens = True
    _no_split_modules = ["CogAgentDecoderLayer", "TransformerLayer", "Block"]

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config
        quant_config = vllm_config.quant_config
        self.hf_config = vllm_config.model_config.hf_config  # type: CogAgentConfig
        self.lora_config = vllm_config.lora_config

        self.model = CogAgentModel(vllm_config, prefix=maybe_prefix(prefix, "model"))

        self.vision = EVA2CLIPModel(
            self.hf_config.vision_config,
            multimodal_config=self.config.model_config.multimodal_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model.vision"),
        )

        self.cross_vision = CrossVisionModel(
            self.hf_config.cross_vision_config,
            multimodal_config=self.config.model_config.multimodal_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model.cross_vision"),
        )
        
        self.lm_head = ParallelLMHead(
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            org_num_embeddings=self.hf_config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.hf_config.vocab_size, self.hf_config.vocab_size
        )

        self.cross_hidden_size = self.hf_config.cross_hidden_size
        self.num_vision_tokens = get_max_image_tokens(self.hf_config.vision_config)
        self.num_cross_vision_tokens = self.cross_vision.num_tokens

        self.position_ids = build_positions(
            vllm_config.scheduler_config,
            self.num_vision_tokens - 2,  # -2 is for BOI/EOI
            device=vllm_config.device_config.device_type,
        )

    @cached_property
    def image_token_id(self) -> int:
        tokenizer = cached_tokenizer_from_config(self.config.model_config)
        image_token = self.hf_config.image_token
        return tokenizer.get_added_vocab()[image_token]

    def build_token_masks(self, input_ids: torch.LongTensor):
        vision_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        vision_token_mask[:-1] = (input_ids[:-1] == self.image_token_id) & (
            input_ids[1:] == self.image_token_id
        )
        language_token_mask = ~vision_token_mask
        # TODO: Investigate if this change is worth it. Affects the Attention and
        # MLP Layers
        #language_token_mask = language_token_mask ^ (input_ids == self.pad_token_id)

        return vision_token_mask, language_token_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        image_embeds = None
        cross_embeds = None
        vision_token_ids = None
        language_token_ids = None

        # NOTE: Images are fixed, so we index a precomputed version
        # The downside is each request must have an image. 
        # (An issue with the original as well)
        
        positions = self.position_ids.index_select(0, positions)
        encoder_outputs = self.embed_multimodal(**kwargs)

        if encoder_outputs is not None:
            vision_token_ids, language_token_ids = self.build_token_masks(input_ids)
            image_embeds = encoder_outputs[:, : self.num_vision_tokens, :]
            cross_embeds = encoder_outputs[
                :, self.num_vision_tokens :, : self.cross_hidden_size
            ]

        inputs_embeds = self.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=image_embeds,
            is_multimodal=input_ids == self.image_token_id,
            handle_oov_mm_token=True,
        )

        hidden_states = self.model(
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            encoder_embeds=cross_embeds,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )
        return hidden_states
    
    def get_num_mm_encoder_tokens(self, num_image_tokens):
        return self.num_vision_tokens + self.num_cross_vision_tokens
    
    def get_num_mm_connector_tokens(self, num_vision_tokens):
        return self.num_vision_tokens

    def _parse_and_validate_image_input(self, images, cross_images):
        dtype = self.hf_config.torch_dtype

        if images is None:
            return None

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim == 5:
            images = images.squeeze(1)

        if cross_images.ndim == 3:
            cross_images = cross_images.unsqueeze(0)
        if cross_images.ndim == 5:
            cross_images = cross_images.squeeze(1)

        images = images.to(dtype=dtype)
        cross_images = cross_images.to(dtype=dtype)

        inputs = CogAgentImagePixelInputs(
            type="pixel_values",
            pixel_values=images,
            cross_pixel_values=cross_images,
            resolve_bindings={
                "side": self.hf_config.image_size,
                "cross_side": self.hf_config.cross_image_size,
            },
        )
        return inputs

    @overload
    def _parse_and_validate_input(
        self: Self,
        images: ImageData,
        cross_images: ImageData,
    ) -> CogAgentImagePixelInputs: ...

    @overload
    def _parse_and_validate_input(
        self: Self,
        encoder_outputs: list[torch.Tensor],
    ) -> CogAgentImageEmbeddingInputs: ...

    def _parse_and_validate_input(
        self, **kwargs
    ) -> CogAgentImagePixelInputs | CogAgentImageEmbeddingInputs | None:
        inputs = None
        images = kwargs.pop("pixel_values", None)
        cross_images = kwargs.pop("cross_pixel_values", None)
        image_embeds = kwargs.pop("encoder_outputs", None)

        if image_embeds is not None and images is not None:
            raise ValueError("Cannot pass both images and embeddings")

        if image_embeds is None and cross_images is not None:
            inputs = self._parse_and_validate_image_input(
                images=images, cross_images=cross_images
            )

        elif image_embeds is not None:
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.stack(image_embeds)

            inputs = CogAgentImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                resolve_bindings={
                    "L": self.num_vision_tokens + self.num_cross_vision_tokens,
                    "HD": self.hf_config.vision_config.outer_hidden_size,
                },
            )

        return inputs

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<EOI>"
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs) -> torch.FloatTensor:
        image_inputs = self._parse_and_validate_input(**kwargs)

        hidden_size: int = self.hf_config.hidden_size
        sequence_length: int = self.num_vision_tokens + self.num_cross_vision_tokens

        if image_inputs is None:
            return None

        elif isinstance(image_inputs, CogAgentImageEmbeddingInputs):
            return image_inputs.image_embeds

        # B, C, H, W -> L, D
        image_inputs: CogAgentImagePixelInputs
        cross_embeds: torch.Tensor = self.cross_vision(image_inputs.cross_pixel_values)
        image_embeds: torch.Tensor = self.vision(image_inputs.pixel_values)

        # Pad and concat to enforce output constraints.
        # This introduces an overhead (with defaults, around 40 MB)
        batch = image_embeds.shape[0] if image_embeds.ndim >= 3 else 1
        embed_shape = [batch, sequence_length, hidden_size]

        multimodal_embeddings = image_embeds.new_zeros(*embed_shape)
        multimodal_embeddings[:, : self.num_vision_tokens, :] = image_embeds
        multimodal_embeddings[:, self.num_vision_tokens :, : self.cross_hidden_size] = (
            cross_embeds
        )

        return multimodal_embeddings

    def get_language_model(self):
        return self.model

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model.layers",
            connector="model.vision.linear_proj",
            tower_model=["model.vision.transformer", "model.cross_vision.vit"],
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        skip_prefixes = ["cross_vision.vit.model.rope"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)

        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
