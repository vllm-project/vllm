# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/THUDM/CogAgent
"""Inference-only CogAgent model compatible with THUDM weights."""
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Literal, Optional, TypedDict, Union

import torch
from torch import nn
from torch.nn import LayerNorm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature, PreTrainedTokenizer, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention.layer import MultiHeadAttention
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import ChatGLMConfig

from .chatglm import ChatGLMBaseModel, ChatGLMModel
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import flatten_bn, merge_multimodal_embeddings


class GLMVImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""


class EVA2CLIPPatchEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels,
                              config.hidden_size,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions,
                                               config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        images = images.to(device=self.proj.weight.device,
                           dtype=self.proj.weight.dtype)
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class EVA2CLIPAttention(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = config.num_heads // self.tp_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.attn = MultiHeadAttention(self.num_heads_per_rank, self.head_dim,
                                       self.scale)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.query_key_value(x)  # B, L, 3 * H * D
        q, k, v = qkv.chunk(3, dim=-1)

        out = self.attn(q, k, v)
        output, _ = self.dense(out)
        output = self.output_dropout(output)
        return output


class EVA2CLIPMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x, _ = self.fc2(x)
        return x


class EVA2CLIPTransformerLayer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size,
                                         eps=config.layer_norm_eps)
        self.attention = EVA2CLIPAttention(config,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.attention")
        self.mlp = EVA2CLIPMLP(config,
                               quant_config=quant_config,
                               prefix=f"{prefix}.mlp")
        self.post_attention_layernorm = LayerNorm(config.hidden_size,
                                                  eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(
            self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class EVA2CLIPTransformer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EVA2CLIPTransformerLayer(config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class EVA2CLIPGLU(nn.Module):

    def __init__(
        self,
        config,
        in_features,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        """
        The original implementation is the same as:
        ```python
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config
        )

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config
        )
        ```
        ```
        gate_proj_output, _ = self.gate_proj(x)
        dense_h_to_4h_output, _ = self.dense_h_to_4h(x)
        x = torch.cat([gate_proj_output, dense_h_to_4h_output], dim=-1)
        ```

        We merge two ColumnParallelLinear into one MergedColumnParallelLinear:
        ```
        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config
        )
        ```
        ```
        x, _ = self.merged_proj(x)
        ```
        """
        super().__init__()
        self.linear_proj = ReplicatedLinear(in_features,
                                            config.hidden_size,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.linear_proj")
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = SiluAndMul()

        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.merged_proj")

        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h")

    def forward(self, x):
        x, _ = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x, _ = self.merged_proj(x)
        x = self.act2(x)
        x, _ = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = EVA2CLIPPatchEmbedding(vision_config)
        self.transformer = EVA2CLIPTransformer(vision_config,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.transformer")
        self.linear_proj = EVA2CLIPGLU(config,
                                       in_features=config.hidden_size,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.linear_proj")
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size,
                              out_channels=config.hidden_size,
                              kernel_size=2,
                              stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x


class GLM4VModel(ChatGLMModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = vllm_config.quant_config

        self.vision = EVA2CLIPModel(self.config,
                                    quant_config,
                                    prefix=f"{prefix}.vision")


class GLM4VProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        vision_config = config.vision_config
        image_size = vision_config["image_size"]

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        text_inputs = self.tokenizer(text)

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values = [self.image_transform(image) for image in images]
            image_inputs = {"pixel_values": torch.stack(pixel_values)}

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class GLM4VProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(ChatGLMConfig)

    def get_hf_processor(self, **kwargs: object) -> GLM4VProcessor:
        return self.ctx.init_processor(
            GLM4VProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        image_size = vision_config["image_size"]
        patch_size = vision_config["patch_size"]
        grid_length = image_size // patch_size // 2
        return grid_length * grid_length

    def get_num_image_feature_tokens(self) -> int:
        # EVA2CLIPModel has embeddings for boi and eoi tokens as well
        return self.get_num_image_tokens() + 2


class GLM4VDummyInputsBuilder(BaseDummyInputsBuilder[GLM4VProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        base_text = "<|begin_of_image|><|endoftext|><|end_of_image|>"

        return base_text * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config

        target_width = target_height = vision_config["image_size"]
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class GLM4VMultiModalProcessor(BaseMultiModalProcessor[GLM4VProcessingInfo]):

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()

        boi_token_id = hf_config.boi_token_id
        image_token_id = hf_config.pad_token_id
        eoi_token_id = hf_config.eoi_token_id

        def get_replacement(item_idx: int):
            num_image_tokens = self.info.get_num_image_tokens()
            image_tokens = [image_token_id] * num_image_tokens

            return [boi_token_id] + image_tokens + [eoi_token_id]

        return [
            PromptReplacement(
                modality="image",
                target=[boi_token_id, image_token_id, eoi_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(GLM4VMultiModalProcessor,
                                        info=GLM4VProcessingInfo,
                                        dummy_inputs=GLM4VDummyInputsBuilder)
class GLM4VForCausalLM(ChatGLMBaseModel, SupportsLoRA, SupportsPP,
                       SupportsMultiModal):

    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"],
        "merged_proj": ["gate_proj", "dense_h_to_4h"]
    }

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="transformer.encoder",
            connector="transformer.vision.linear_proj",
            tower_model="transformer.vision.transformer")

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[GLM4VModel] = GLM4VModel,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            transformer_type=transformer_type,
        )

        self.transformer: GLM4VModel

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config["image_size"]
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[GLMVImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return GLMVImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        return None

    def _process_image_input(
            self, image_input: GLMVImagePixelInputs) -> torch.Tensor:
        pixel_values = image_input["data"].to(dtype=self.config.torch_dtype)

        return self.transformer.vision(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.transformer

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.transformer.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=[
                    self.config.boi_token_id,
                    self.config.pad_token_id,
                    self.config.eoi_token_id,
                ],
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.transformer(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)

        return hidden_states
