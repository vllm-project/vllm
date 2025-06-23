# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/Qwen/Qwen-VL/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
"""Inference-only Qwen-VL model compatible with HuggingFace weights."""

import copy
import math
import unicodedata
from collections.abc import Collection, Mapping, Sequence, Set
from functools import lru_cache, partial
from typing import Callable, Literal, Optional, TypedDict, Union

import regex as re
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (BatchFeature, PretrainedConfig, PreTrainedTokenizer,
                          TensorType)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import Resampler2, get_abs_pos
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .qwen import QWenBaseModel, QWenModel
from .utils import flatten_bn, merge_multimodal_embeddings


class QwenImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, 3, image_size, image_size)`

    Note that image_size is the value in the vision config to which we resize
    the image to in the normalization transform. Currently multi-image support
    can only be leveraged by passing image embeddings directly.
    """


class QwenImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, 256, hidden_size)`

    `hidden_size` must match the hidden size of the language model backbone
    and is stored in the visual config of the model if we have one.
    """


QwenImageInputs = Union[QwenImagePixelInputs, QwenImageEmbeddingInputs]


class VisualAttention(nn.Module):
    """self-attention layer class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim \
            and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, \
                'Visual Attention implementation only supports self-attention'
        self.in_proj = ReplicatedLinear(embed_dim, 3 * embed_dim)
        self.out_proj = ReplicatedLinear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query/key/value: [sq, b, h]
        sq, b, _ = x.size()
        mixed_x_layer, _ = self.in_proj(x)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled,
                                            key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(
            b, self.num_attention_heads_per_partition, sq,
            self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output, _ = self.out_proj(context_layer)

        return output


class QwenVLMLP(nn.Module):
    """MLP for the visual component of the Qwen model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.c_fc = ColumnParallelLinear(hidden_size,
                                         intermediate_size,
                                         bias=True,
                                         quant_config=quant_config)
        self.act_fn = get_act_fn("gelu")
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x):
        x, _ = self.c_fc(x)
        x = self.act_fn(x)
        x, _ = self.c_proj(x)
        return x


class VisualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = QwenVLMLP(
            hidden_size=d_model,
            intermediate_size=mlp_width,
            quant_config=quant_config,
        )

    def attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, attn_mask=attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(width,
                                 heads,
                                 mlp_ratio,
                                 norm_layer=norm_layer,
                                 quant_config=quant_config)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 n_queries: int = 256,
                 output_dim: int = 512,
                 image_start_id: int = 151857,
                 quant_config: Optional[QuantizationConfig] = None,
                 **kwargs):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale *
                                                 torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(width,
                                            layers,
                                            heads,
                                            mlp_ratio,
                                            norm_layer=norm_layer,
                                            quant_config=quant_config)

        self.attn_pool = Resampler2(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
            adaptive=False,
            do_post_projection=False,
        ).to(
            device=self.positional_embedding.device,
            dtype=self.positional_embedding.dtype,
        )

        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter(
            (output_dim**-0.5) * torch.randn(output_dim, output_dim))

        self.image_start_id = image_start_id
        self.image_end_id = image_start_id + 1
        self.image_pad_id = image_start_id + 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )

        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + get_abs_pos(self.positional_embedding, int(math.sqrt(
            x.size(1))))

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj

        return x


class QwenVLModel(QWenModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.visual = VisionTransformer(**config.visual,
                                        quant_config=quant_config)


@lru_cache(maxsize=1)
def _get_tokenizer_without_image_pad(
        tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    The logic of adding image pad tokens should only be applied in
    [`QwenVLProcessor`][vllm.model_executor.models.qwen_vl.QwenVLProcessor],
    so they are patched out here.

    The definition of the wrapped tokenizer can be found here:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/tokenization_qwen.py
    """
    new_tokenizer = copy.deepcopy(tokenizer)

    class TokenizerWithoutImagePad(tokenizer.__class__):  # type: ignore

        def tokenize(
            self,
            text: str,
            allowed_special: Union[Set[str], str] = "all",
            disallowed_special: Union[Collection[str], str] = (),
            **kwargs,
        ) -> list[Union[bytes, str]]:
            text = unicodedata.normalize("NFC", text)

            return [
                self.decoder[t] for t in self.tokenizer.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            ]

        def _decode(
            self,
            token_ids: Union[int, list[int]],
            skip_special_tokens: bool = False,
            errors: Optional[str] = None,
            **kwargs,
        ) -> str:
            if isinstance(token_ids, int):
                token_ids = [token_ids]

            return self.tokenizer.decode(
                token_ids,
                errors=errors or self.errors,
            )

    TokenizerWithoutImagePad.__name__ = \
        f"{tokenizer.__class__.__name__}WithoutImagePad"

    new_tokenizer.__class__ = TokenizerWithoutImagePad
    return new_tokenizer


class QwenVLProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    We call the wrapped tokenizer to automatically insert image pad tokens:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/tokenization_qwen.py#L245

    The image processor is defined here:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py#L354
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        vision_config = config.visual
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

    @property
    def image_start_tag(self) -> str:
        return self.tokenizer.image_start_tag  # type: ignore

    @property
    def image_end_tag(self) -> str:
        return self.tokenizer.image_end_tag  # type: ignore

    @property
    def image_pad_tag(self) -> str:
        return self.tokenizer.image_pad_tag  # type: ignore

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


class QwenVLProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer)

        return _get_tokenizer_without_image_pad(tokenizer)

    def get_hf_processor(self, **kwargs: object) -> QwenVLProcessor:
        return self.ctx.init_processor(
            QwenVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.visual

        image_size = vision_config["image_size"]
        patch_size = vision_config["patch_size"]
        grid_length = image_size // patch_size // 2
        return grid_length * grid_length


class QwenVLDummyInputsBuilder(BaseDummyInputsBuilder[QwenVLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        hf_processor = self.info.get_hf_processor()
        img_start = hf_processor.image_start_tag
        img_end = hf_processor.image_end_tag

        return "".join(f"Picture {i}: {img_start}{img_end}\n"
                       for i in range(1, num_images + 1))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.visual

        target_width = target_height = vision_config["image_size"]
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class QwenVLMultiModalProcessor(BaseMultiModalProcessor[QwenVLProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Drops anything between <img>/</img> tags; encoding with the tokenizer
        # will automatically add the image pads for the context.
        prompt, num_matched_images = re.subn(
            r"(Picture \d*: <img>).*?(<\/img>\n)",
            r"\1\2",
            prompt,
        )

        image_data = mm_data.get("images")
        if image_data is not None:
            assert isinstance(image_data, list)

            num_images = len(image_data)
            assert num_matched_images == num_images

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

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
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        special_tokens: dict[str,
                             int] = tokenizer.special_tokens  # type: ignore

        processor = self.info.get_hf_processor()
        img_start_id = special_tokens[processor.image_start_tag]
        img_end_id = special_tokens[processor.image_end_tag]
        img_pad_id = special_tokens[processor.image_pad_tag]

        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [img_pad_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[img_start_id, img_end_id],
                replacement=PromptUpdateDetails.select_token_id(
                    [img_start_id] + image_tokens + [img_end_id],
                    embed_token_id=img_pad_id,
                ),
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(QwenVLMultiModalProcessor,
                                        info=QwenVLProcessingInfo,
                                        dummy_inputs=QwenVLDummyInputsBuilder)
class QwenVLForConditionalGeneration(QWenBaseModel, SupportsPP, SupportsLoRA,
                                     SupportsMultiModal):
    packed_modules_mapping = {
        "c_attn": ["c_attn"],
        "gate_up_proj": [
            "w2",
            "w1",
        ],
    }

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="transformer.h",
            connector="transformer.visual.attn_pool",
            tower_model="transformer.visual.transformer")

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[QwenVLModel] = QwenVLModel,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            transformer_type=transformer_type,
        )

        self.transformer: QwenVLModel

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.visual["image_size"]
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[QwenImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return QwenImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return QwenImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        return None

    def _process_image_input(self,
                             image_input: QwenImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        return self.transformer.visual(image_input["data"])

    def get_language_model(self) -> torch.nn.Module:
        return self.transformer

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.transformer.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.transformer.visual.image_pad_id)

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
