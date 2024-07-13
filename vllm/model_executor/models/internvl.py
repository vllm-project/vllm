# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, List, Iterable, Tuple, Literal, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import MultiModalConfig, CacheConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SamplerOutput

from .clip import (get_clip_num_patches, dummy_image_for_clip, dummy_seq_data_for_clip)
from .interfaces import SupportsVision
from .utils import merge_vision_embeddings


IMG_START='<img>'
IMG_END='</img>'
IMG_CONTEXT='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MAX_IMAGE_FEATURE_SIZE_WIDTH = 3000
MAX_IMAGE_FEATURE_SIZE_HEIGHT = 500

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}

class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        # x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        return x


class InternMLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config, quant_config=quant_config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2

        return hidden_states


class InternVisionEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, quant_config=quant_config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        vision_feature_layer: int = -1
    ):
        hidden_states = inputs_embeds

        num_layer = len(self.layers) + vision_feature_layer + 1
        for encoder_layer in self.layers[:num_layer]:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class InternVisionModel(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config, quant_config=quant_config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            vision_feature_layer: int = -1,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            vision_feature_layer=vision_feature_layer
        )

        return encoder_outputs
    

class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: BatchedTensors
    """
    Shape: `(batch_size, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.
    """


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_num_blocks(orig_width: int, orig_height: int, min_num=1, max_num=6, image_size=448):
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    return blocks, target_width, target_height


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size

    blocks, target_width, target_height = calculate_num_blocks(orig_width, orig_height, min_num, max_num, image_size)
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values(image: Image.Image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_internvl_num_patches(image_size: int, patch_size: int, downsample_ratio: float):
    return int(get_clip_num_patches(image_size=image_size, patch_size=patch_size) * (downsample_ratio ** 2))


def get_max_internvl_image_tokens(ctx: InputContext):
    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PretrainedConfig)
    vision_config = hf_config.vision_config
    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    downsample_ratio = model_config.downsample_ratio
    num_patches = get_internvl_num_patches(image_size, patch_size, downsample_ratio)
    return num_patches * 7


def input_processor_for_internvl(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PretrainedConfig)
    vision_config = hf_config.vision_config

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        width, height = image_data.size
        num_blocks, _, _ = calculate_num_blocks(width, height)
    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    downsample_ratio = hf_config.downsample_ratio
    num_patches = get_internvl_num_patches(image_size, patch_size, downsample_ratio)

    tokenizer = cached_get_tokenizer(model_config.tokenizer, trust_remote_code=True)

    prompt = llm_inputs["prompt"]
    image_prompt = IMG_START + IMG_CONTEXT * (num_blocks+1) * num_patches + IMG_END
    new_prompt = prompt.replace('<image>', image_prompt, 1)
    new_prompt_token_ids = tokenizer.encode(new_prompt, add_special_tokens=False)

    return LLMInputs(prompt=new_prompt,
                     prompt_token_ids=new_prompt_token_ids,
                     multi_modal_data=multi_modal_data)


def input_mapper_for_internvl(ctx: InputContext, data: object):
    if isinstance(data, Image.Image):
        data = image_to_pixel_values(data)
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer, trust_remote_code=True)
    imgae_token_id = tokenizer.encode(IMG_CONTEXT, add_special_tokens=False, return_tensors="pt")[0]

    return MultiModalInputs({"pixel_values": data, "image_token_id": imgae_token_id})


def dummy_data_for_internvl(ctx: InputContext, seq_len: int):

    image_feature_size = get_max_internvl_image_tokens(ctx)
    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PretrainedConfig)
    tokenizer = cached_get_tokenizer(model_config.tokenizer, trust_remote_code=True)

    seq_data = dummy_seq_data_for_clip(
        hf_config,
        seq_len,
        image_token_id=tokenizer.encode(IMG_CONTEXT, add_special_tokens=False)[0],
        image_feature_size_override=image_feature_size,
    )
    mm_data = dummy_image_for_clip(
        hf_config,
        image_width_override=MAX_IMAGE_FEATURE_SIZE_WIDTH,
        image_height_override=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
    )

    return seq_data, mm_data
    

@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_internvl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_internvl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_internvl)
class InternVLChatModel(nn.Module, SupportsVision):
    def __init__(self, config: PretrainedConfig, 
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        
        self.vision_model = InternVisionModel(config.vision_config)
        llm_class = ModelRegistry.load_model_cls(config.llm_config.architectures[0])
        self.language_model = llm_class(config.llm_config, cache_config, quant_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        vit_embeds = self.vision_model(
            vision_feature_layer=self.select_layer,
            pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected image sizes shape is batch dimension plus "
                f"{[2]}. You supplied {data.shape}.")

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values in each batch element "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_token_id = kwargs.pop("image_token_id", None)

        if pixel_values is None:
            return None
        
        self.img_context_token_id = image_token_id[0]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return InternVLImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
            # inputs_embeds = self.language_model.model.tok_embeddings(input_ids)
            vit_embeds = self.extract_feature(image_input["data"])
            inputs_embeds = merge_vision_embeddings(input_ids, inputs_embeds, vit_embeds, self.img_context_token_id)
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            None,
                                            inputs_embeds=inputs_embeds)
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.llm_config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # We only do sharding for language model
                # and not vision model for now.
                if "vision_embed_tokens" in name and self.vision_embed_tokens:
                    continue
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
