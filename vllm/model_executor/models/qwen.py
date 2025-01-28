# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""

import copy
import math
import re
import unicodedata
from functools import lru_cache, partial
from typing import (AbstractSet, Any, Callable, Collection, Dict, Iterable,
                    List, Literal, Mapping, Optional, Set, Tuple, TypedDict,
                    Union)

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (BatchFeature, PretrainedConfig, PreTrainedTokenizer,
                          TensorType)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import Resampler2, get_abs_pos
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptReplacementDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, merge_multimodal_embeddings)

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
        prefix: str = "",
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
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

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
        prefix: str = "",
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
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.attn")

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


@support_torch_compile
class QWenModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: QWenBlock(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if (vision_config := getattr(config, "visual", None)):
            self.visual = VisionTransformer(**vision_config,
                                            quant_config=quant_config)
        else:
            self.visual = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
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


def build_normalization_transform(image_size: int) -> transforms.Compose:
    """
    Build a normalization transform which can be applied to one or
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


@lru_cache(maxsize=1)
def _get_tokenizer_without_image_pad(
        tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    The logic of adding image pad tokens should only be applied in
    :class:`QWenVLProcessor`, so they are patched out here.

    The definition of the wrapped tokenizer can be found here:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/tokenization_qwen.py
    """
    new_tokenizer = copy.deepcopy(tokenizer)

    class TokenizerWithoutImagePad(tokenizer.__class__):  # type: ignore

        def tokenize(
            self,
            text: str,
            allowed_special: Union[AbstractSet[str], str] = "all",
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
            token_ids: Union[int, List[int]],
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


class QWenVLProcessor:
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

        if hasattr(self.config, "visual"):
            self.image_transform = build_normalization_transform(
                config.visual["image_size"])
        else:
            self.image_transform = None

        special_tokens: dict[str,
                             int] = tokenizer.special_tokens  # type: ignore
        self.img_start_id = special_tokens[IMG_START]
        self.img_end_id = special_tokens[IMG_END]

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
            if self.image_transform is None:
                raise ValueError("This model does not support image inputs")

            pixel_values = [self.image_transform(image) for image in images]
            image_inputs = {"pixel_values": torch.stack(pixel_values)}

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class QWenVLProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer)

        return _get_tokenizer_without_image_pad(tokenizer)

    def get_hf_processor(self) -> QWenVLProcessor:
        tokenizer = self.ctx.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer)

        return QWenVLProcessor(self.get_hf_config(), tokenizer)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        return {"image": self.get_num_image_tokens()}

    def get_num_image_tokens(self) -> int:
        return MAX_QWEN_IMG_TOKENS


class QWenVLDummyInputsBuilder(BaseDummyInputsBuilder[QWenVLProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        hf_config = self.info.get_hf_config()
        if not hasattr(hf_config, "visual"):
            return ProcessorInputs(prompt_text="", mm_data={})

        vision_config = hf_config.visual

        max_image_size = vision_config["image_size"]
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=max_image_size,
                                   height=max_image_size,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="".join(f"Picture {i}: {IMG_START}{IMG_END}\n"
                                for i in range(1, num_images + 1)),
            mm_data=mm_data,
        )


class QWenVLMultiModalProcessor(BaseMultiModalProcessor[QWenVLProcessingInfo]):

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
            if num_matched_images != num_images:
                logger.warning(
                    "Number of matched image placeholders %s doesn't match "
                    "the number of expected images %s; check your placeholder "
                    "formatting.", num_matched_images, num_images)

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()
        special_tokens: dict[str,
                             int] = tokenizer.special_tokens  # type: ignore

        img_start_id = special_tokens[IMG_START]
        img_end_id = special_tokens[IMG_END]
        img_pad_id = special_tokens[IMG_PAD]

        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [img_pad_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[img_start_id, img_end_id],
                replacement=PromptReplacementDetails(
                    full=[img_start_id] + image_tokens + [img_end_id],
                    features=image_tokens,
                ),
            )
        ]


class QWenBaseModel(nn.Module, SupportsPP, SupportsLoRA):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.transformer = QWenModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(
                                         prefix, "transformer"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

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
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return QwenImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return QwenImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        return None

    def _process_image_input(self,
                             image_input: QwenImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.transformer.visual is not None
        return self.transformer.visual(image_input["data"])

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.transformer.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            assert self.transformer.visual is not None
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.transformer.visual.image_pad_id)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
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

        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors,
                                         inputs_embeds)
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
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
            loaded_params.add(name)
        return loaded_params


class QWenLLM(QWenBaseModel):
    packed_modules_mapping = {
        "c_attn": ["c_attn"],
        "gate_up_proj": [
            "w2",
            "w1",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "c_attn",
        "gate_up_proj",
        "c_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []


class QWenVL(QWenBaseModel, SupportsMultiModal):
    packed_modules_mapping = {
        "c_attn": ["c_attn"],
        "gate_up_proj": [
            "w2",
            "w1",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "c_attn",
        "gate_up_proj",
        "c_proj",
        # visual module
        "out_proj",
        "in_proj",
        "c_fc",
        # resampler
        "kv_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="transformer.h",
            connector="transformer.visual.attn_pool",
            tower_model="transformer.visual.transformer")


@MULTIMODAL_REGISTRY.register_processor(QWenVLMultiModalProcessor,
                                        info=QWenVLProcessingInfo,
                                        dummy_inputs=QWenVLDummyInputsBuilder)
class QWenLMHeadModel(QWenBaseModel, SupportsMultiModal, SupportsLoRA):
    """
    QWenLMHeadModel is not only applicable to LLM  but also to VL, which is not 
    conducive to the current integration logic of LoRA in vLLM. Therefore, it 
    is necessary to separate them.
    """
    # Ensure that the LoRA support check passes when the class is not
    # initialized, but set all these attributes to empty.
    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    def __new__(
        cls,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> QWenBaseModel:
        config = vllm_config.model_config.hf_config
        # Initialize VL
        if hasattr(config, "visual"):
            return QWenVL(vllm_config=vllm_config, prefix=prefix)
        # Initialize LLM
        else:
            return QWenLLM(vllm_config=vllm_config, prefix=prefix)
