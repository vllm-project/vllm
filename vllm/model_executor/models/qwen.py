# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""

import math
import re
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Tuple, TypedDict, Union)

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import Resampler2, get_abs_pos
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.utils import is_list_of

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

logger = init_logger(__name__)

# NOTE: Qwen models have a few other special tags, e.g., ref, bbox, quad;
# for the time being, these tags are not considered as special at encoding
# time. This may change as VLLMs multimodal API changes in the future.
IMG_START = "<img>"
IMG_END = "</img>"
IMG_PAD = "<imgpad>"
# Image context is fixed at 256 for all images
MAX_QWEN_IMG_TOKENS = 256
# Image normalization params
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query/key/value: [sq, b, h]
        sq, b, _ = x.size()
        mixed_x_layer = self.in_proj(x)

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

        output = self.out_proj(context_layer)

        return output


class QwenVMLP(nn.Module):
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
        self.act_fn = get_act_fn("gelu", quant_config, intermediate_size)
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
        norm_layer: Callable = nn.LayerNorm,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = QwenVMLP(
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
        norm_layer: Callable = nn.LayerNorm,
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

    def get_image_positions(self,
                            input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Given the input IDs, extracts start/stop points corresponding to
        images.

        args:
        Returns:
            Optional torch tensor corresponding to start/stop pairs of images.
        """
        if torch.any(input_ids == self.image_start_id):
            bos_pos = torch.where(input_ids == self.image_start_id)
            eos_pos = torch.where(input_ids == self.image_end_id)
            return torch.stack((bos_pos[0], eos_pos[0]), dim=1)
        return None


class QWenMLP(nn.Module):
    """MLP for the language component of the Qwen model, which contains a
    MergedColumnParallelLinear merging 2 outputs via silu activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.c_proj = RowParallelLinear(intermediate_size,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size(
        )
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(config.hidden_size,
                                  config.num_attention_heads,
                                  config.max_position_embeddings,
                                  rope_theta=rope_theta,
                                  rope_scaling=rope_scaling,
                                  cache_config=cache_config,
                                  quant_config=quant_config)

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(config.hidden_size,
                           config.intermediate_size // 2,
                           quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QWenModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: QWenBlock(config, cache_config, quant_config),
            prefix=f"{prefix}.h")
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.visual = VisionTransformer(**config.visual,
                                        quant_config=quant_config) if hasattr(
                                            config, "visual") else None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        pixel_values: Optional[QwenImageInputs],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        img_pos = None
        # If pixel / visual embeddings are provided, this is a visual model
        if pixel_values is not None and self.visual is not None:
            if pixel_values["type"] != "image_embeds":
                image_embeds = self.visual(pixel_values["data"])
            else:
                image_embeds = pixel_values["data"]

            # features should be of shape (# images, 256, hidden_dim)
            img_pos = self.visual.get_image_positions(input_ids)
            if isinstance(
                    img_pos,
                    np.ndarray) and img_pos.shape[0] != image_embeds.shape[0]:
                raise ValueError(
                    f"Number of placeholders: {img_pos.shape[0]} "
                    f"does not match number of images {image_embeds.shape[0]}."
                )

        if get_pp_group().is_first_rank:
            hidden_states = self.wte(input_ids)
            # Merge the image embeddings into the hidden states if actually have
            # visual features and the corresponding image tokens
            if img_pos is not None:
                for idx, (img_bos, img_eos) in enumerate(img_pos):
                    hidden_states[img_bos + 1:img_eos] = image_embeds[idx]
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


def get_image_text(image_num: int, padding: bool) -> str:
    """Retrieves a placeholder text that when tokenized, will be expanded with
    image pads.

    Args:
        image_num: The number of the image that we want a text prompt for.
            Images should be indexed starting at 1.
        padding: Whether or not padding should be manually added.

    Returns:
        Text placeholder prompt for the image being considered.
    """
    image_start = f"Picture {image_num}: {IMG_START}"
    image_end = f"{IMG_END}\n"
    if not padding:
        return f"{image_start}{image_end}"
    return f"{image_start}{MAX_QWEN_IMG_TOKENS * IMG_PAD}{image_end}"


def input_processor_for_qwen(ctx: InputContext,
                             llm_inputs: LLMInputs) -> LLMInputs:
    """Processes the inputs, which may or may not be multimodal.
    Multimodal inputs will only be processed if the model has a "visual"
    component in its model config, otherwise they'll be ignored.

    Args:
        ctx: Context of the loaded model.
        llm_inputs: LLM inputs which may have a multi_modal_data attribute.

    Returns:
        If the model is language only or not multimodal inputs were provided,
        returns llm_inputs unmodified. Otherwise, processes the multimodal
        images / image embeddings and adds the fixed-length image placeholders.
    """
    multi_modal_data = llm_inputs.get("multi_modal_data")

    # Only process images if we have multimodal data and a visual config
    hf_config = ctx.get_hf_config()
    if (multi_modal_data is None or "image" not in multi_modal_data
            or not hasattr(hf_config, "visual")):
        return llm_inputs

    prompt = llm_inputs.get("prompt")
    prompt_token_ids = llm_inputs["prompt_token_ids"]
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)
    image_data = multi_modal_data["image"]
    if isinstance(image_data, torch.Tensor):
        num_dims = len(image_data.shape)
        if num_dims < 2 or num_dims > 3:
            raise ValueError(
                f"Expected img embeds to be have 3 dimensions, got {num_dims}")
        num_images = 1 if num_dims == 2 else image_data.shape[0]
    elif isinstance(image_data, Image.Image):
        num_images = 1
    elif is_list_of(image_data, Image.Image):
        num_images = len(image_data)
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    if prompt is None:
        prompt = tokenizer.decode(prompt_token_ids)

    # Drops anything between <img>/</img> tags; encoding with the tokenizer
    # will automatically add the image pads for the context.
    new_prompt, num_matched_images = re.subn(
        r"(Picture \d*: <img>).*?(<\/img>\n)",
        r"\1\2",
        prompt,
    )

    if num_matched_images != num_images:
        logger.warning(
            "Number of matched image placeholders %s doesn't match the number "
            "of expected images %s; check your placeholder formatting.",
            num_matched_images, num_images)

    new_prompt_token_ids = tokenizer.encode(new_prompt)

    return LLMInputs(prompt=new_prompt,
                     prompt_token_ids=new_prompt_token_ids,
                     multi_modal_data=multi_modal_data)


def input_mapper_for_qwen(ctx: InputContext, data: object) -> MultiModalInputs:
    """Maps the input data to its MultiModalInputs (if any).

    Args:
        ctx: Context of the loaded model.
        data: data potentially containing image/image embeddings to be mapped
            to pixel_values in .forward() for a visual QWenLMHeadModel model.

    Returns:
        MultiModalInputs containing the stacked normalized images tensor or
        image embeddings.
    """
    # Early exit if we have provided an image to a language only Qwen model
    hf_config = ctx.get_hf_config()
    if not hasattr(hf_config, "visual"):
        logger.warning(
            "Images were provided but this model has no visual config; "
            "multimodal inputs will not be forwarded to the model.")
        return MultiModalInputs()

    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)

    image_pair_tok = tokenizer.encode(IMG_START + IMG_END,
                                      add_special_tokens=False,
                                      return_tensors="pt").squeeze()
    image_start_id = image_pair_tok[0]
    image_end_id = image_pair_tok[-1]
    if (image_start_id + 1) != image_end_id:
        raise ValueError(
            f"Found image end ID {image_end_id}, but expected {IMG_START} + 1")
    if len(image_pair_tok) != (MAX_QWEN_IMG_TOKENS + 2):
        raise ValueError(
            f"Expected image context length of {MAX_QWEN_IMG_TOKENS}, "
            f"but got {image_pair_tok - 2}")

    hf_config = ctx.get_hf_config()
    image_size = hf_config.visual["image_size"]
    img_emb_size = hf_config.visual["output_dim"]

    if isinstance(data, torch.Tensor):
        # It's expected that our values have already been processed
        # by the visual transformer; shape is expected to be:
        # (# images, 256, hidden_size)
        if len(data.shape) == 2:
            # Assume only one image embed was provided; unsqueeze the extra dim
            data = data.unsqueeze(0)
        if len(data.shape) != 3 or data.shape[
                1] != MAX_QWEN_IMG_TOKENS or data.shape[2] != img_emb_size:
            raise ValueError(
                "Expected image embeds to be a tensor of shape"
                f"[# images, {MAX_QWEN_IMG_TOKENS}, {img_emb_size}], but "
                f"received shape [{data.shape}]")
        pixel_values = data
    else:
        transform = build_normalization_transform(image_size)
        if not isinstance(data, (list, tuple)):
            data = [data]
        transformed_images = [transform(datum) for datum in data]
        pixel_values = torch.stack(transformed_images, dim=0)
    return MultiModalInputs({"pixel_values": pixel_values})


def build_normalization_transform(image_size: int) -> transforms.Compose:
    """Builds a normalization transform which can be applied to one or
    more input images from which we want to extract visual features.

    Args:
        image_size: size of the image to be processed for visual embeddings.
    
    Returns:
        Callable transform for normalizing and resizing one RGB image.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def dummy_data_for_qwen(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> Tuple[SequenceData, Optional[Dict]]:
    """Build dummy data for warming up Qwen models; this will only contain text
    matching the defaults for VLLM unless the model has a visual config.

    Args:
        ctx: Context of the loaded model.
        seq_len: Number of tokens in the text sequence.
        mm_counts: multimodal data counts.
    
    Returns:
        Tuple containing sequential and multimodal data.
    """
    hf_config = ctx.get_hf_config()

    # The presence of a visual config indicates this is a multimodal model.
    # If we don't have it, the model is considered an LLM for warmup purposes.
    if not hasattr(hf_config, "visual"):
        seq_data = SequenceData.from_token_counts((0, seq_len))
        mm_data = None
        return seq_data, mm_data

    # We have a visual component - use images to warm up
    num_images = mm_counts["image"]
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)

    # Build the image prompts with no imgpads; the tokenizer will add img pads
    image_prompt = ''.join(
        [get_image_text(idx, False) for idx in range(1, num_images + 1)])
    toks = tokenizer.encode(image_prompt, add_special_tokens=False)

    # Make sure we actually get the fixed context size per tok padding
    num_pads = toks.count(tokenizer.encode(IMG_PAD)[0])
    if num_pads != (num_images * MAX_QWEN_IMG_TOKENS):
        raise ValueError(
            f"Tokenized dummy data should encode {MAX_QWEN_IMG_TOKENS} pads"
            f" per image, but got {num_pads} pads for {num_images} image(s)"
            " in total. Are you using a qwen tokenizer?")

    # Ensure the number of tokens is at minimum the sequence length provided
    if len(toks) < seq_len:
        toks += [0] * (seq_len - len(toks))

    seq_data = SequenceData.from_seqs(toks)

    # Build the input images; width/height doesn't actually matter here since
    # the data will get resized and the # of tokens per image is constant
    image = Image.new("RGB", (224, 224), color=0)
    mm_data = {"image": image if num_images == 1 else [image] * num_images}
    return seq_data, mm_data


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_qwen)
@MULTIMODAL_REGISTRY.register_max_image_tokens(MAX_QWEN_IMG_TOKENS)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen)
class QWenLMHeadModel(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.transformer = QWenModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def _get_image_input_type(
            self,
            pixel_values: Optional[torch.Tensor]) -> Optional[QwenImageInputs]:
        """Determines if the provided pixel_values are normalized pixel values
        or image embeddings.

        Args:
            pixel_values: Optional data to processed into visual embeddings.

        Returns:
            None of the QwenImageInputs type used to determine whether or not
            the visual transformer needs to process the pixel_values.
        """
        if pixel_values is not None and self.transformer.visual is not None:
            pixel_values = flatten_bn(pixel_values)
            if len(pixel_values.shape) == 3 and pixel_values.shape[
                    1] == MAX_QWEN_IMG_TOKENS and pixel_values.shape[
                        2] == self.config.visual["output_dim"]:
                return QwenImageEmbeddingInputs(
                    type="image_embeds",
                    data=pixel_values,
                )
            else:
                # If we have the wrong shape, assume we still need to process
                return QwenImagePixelInputs(
                    type="pixel_values",
                    data=pixel_values,
                )
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        pixel_values: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            pixel_values = None
        else:
            pixel_values = self._get_image_input_type(pixel_values)

        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors,
                                         pixel_values)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
