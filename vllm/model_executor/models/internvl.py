# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import itertools
import re
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_pp_group
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.intern_vit import InternVisionModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   get_clip_num_patches)
from .interfaces import SupportsMultiModal
from .utils import (filter_weights, flatten_bn, init_vllm_registered_model,
                    merge_multimodal_embeddings)

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape:
    `(batch_size * num_images * (1 + num_patches), num_channels, height, width)`
    """


class InternVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


InternVLImageInputs = Union[InternVLImagePixelInputs,
                            InternVLImageEmbeddingInputs]


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
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


def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
                         max_num: int, image_size: int,
                         use_thumbnail: bool) -> Tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # add thumbnail image if num_blocks > 1
    if use_thumbnail and blocks > 1:
        blocks += 1
    return blocks, target_width, target_height


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess(image: Image.Image, min_num: int, max_num: int,
                       image_size: int,
                       use_thumbnail: bool) -> List[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_num_blocks(
        orig_width,
        orig_height,
        min_num,
        max_num,
        image_size,
        use_thumbnail=False)
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values(image: Image.Image, input_size: int, min_num: int,
                          max_num: int, use_thumbnail: bool) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image,
                                min_num=min_num,
                                max_num=max_num,
                                image_size=input_size,
                                use_thumbnail=use_thumbnail)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_internvl_num_patches(image_size: int, patch_size: int,
                             downsample_ratio: float):
    return int(
        get_clip_num_patches(image_size=image_size, patch_size=patch_size) *
        (downsample_ratio**2))


def get_max_internvl_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config()
    vision_config = hf_config.vision_config

    use_thumbnail = hf_config.use_thumbnail
    max_dynamic_patch = hf_config.max_dynamic_patch
    if use_thumbnail:
        max_dynamic_patch += 1
    downsample_ratio = hf_config.downsample_ratio

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    num_patches = get_internvl_num_patches(image_size, patch_size,
                                           downsample_ratio)
    return num_patches * max_dynamic_patch


def input_processor_for_internvl(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config()
    vision_config = hf_config.vision_config

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    downsample_ratio = hf_config.downsample_ratio
    num_patches = get_internvl_num_patches(image_size, patch_size,
                                           downsample_ratio)

    image_data = multi_modal_data["image"]
    min_num = hf_config.min_dynamic_patch
    max_num = hf_config.max_dynamic_patch
    use_thumbnail = hf_config.use_thumbnail
    if isinstance(image_data, Image.Image):
        width, height = image_data.size
        num_blocks, _, _ = calculate_num_blocks(width, height, min_num,
                                                max_num, image_size,
                                                use_thumbnail)
        image_feature_size = [num_blocks * num_patches]
    elif is_list_of(image_data, Image.Image):
        image_feature_size = []
        for image in image_data:
            width, height = image.size
            num_blocks, _, _ = calculate_num_blocks(width, height, min_num,
                                                    max_num, image_size,
                                                    use_thumbnail)
            image_feature_size.append(num_blocks * num_patches)
    elif isinstance(image_data, torch.Tensor):
        num_images, image_feature_size, hidden_size = image_data.shape
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    tokenizer = cached_get_tokenizer(model_config.tokenizer,
                                     trust_remote_code=True)

    prompt = llm_inputs.get("prompt")
    prompt_token_ids = llm_inputs["prompt_token_ids"]
    if prompt is None:
        prompt = tokenizer.decode(prompt_token_ids)

    new_prompt = prompt
    image_idx = sorted(map(int, re.findall(r"Image-(\d+): <image>\n", prompt)))
    for idx, feature_size in enumerate(image_feature_size, start=1):
        image_prompt = IMG_START + IMG_CONTEXT * feature_size + IMG_END
        if not image_idx:
            image_prompt = f"Image-{idx}: {image_prompt}"
        new_prompt = new_prompt.replace('<image>', image_prompt, 1)
    new_prompt_token_ids = tokenizer.encode(new_prompt)

    return LLMInputs(prompt=prompt,
                     prompt_token_ids=new_prompt_token_ids,
                     multi_modal_data=multi_modal_data)


def input_mapper_for_internvl(ctx: InputContext, data: object):
    hf_config = ctx.get_hf_config()

    use_thumbnail = hf_config.use_thumbnail
    min_num = hf_config.min_dynamic_patch
    max_num = hf_config.max_dynamic_patch
    image_size = hf_config.vision_config.image_size

    if isinstance(data, Image.Image):
        data = image_to_pixel_values(data,
                                     image_size,
                                     min_num,
                                     max_num,
                                     use_thumbnail=use_thumbnail)
        # Add an N dimension for number of images per prompt (currently 1).
        data = data.unsqueeze(0)
    elif is_list_of(data, Image.Image):
        # we can't stack here because the images may have different num_patches
        data = [
            image_to_pixel_values(img,
                                  image_size,
                                  min_num,
                                  max_num,
                                  use_thumbnail=use_thumbnail) for img in data
        ]
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer,
                                     trust_remote_code=True)
    image_token_id = tokenizer.encode(IMG_CONTEXT,
                                      add_special_tokens=False,
                                      return_tensors="pt")[0]

    return MultiModalInputs({
        "pixel_values": data,
        "image_token_id": image_token_id
    })


def dummy_data_for_internvl(ctx: InputContext, seq_len: int,
                            mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]

    image_feature_size = get_max_internvl_image_tokens(ctx)
    model_config = ctx.model_config
    hf_config = ctx.get_hf_config()
    vision_config = hf_config.vision_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer,
                                     trust_remote_code=True)

    seq_data = dummy_seq_data_for_clip(
        vision_config,
        seq_len,
        num_images,
        image_token_id=tokenizer.encode(IMG_CONTEXT,
                                        add_special_tokens=False)[0],
        image_feature_size_override=image_feature_size,
    )

    image_size = vision_config.image_size
    min_num = hf_config.min_dynamic_patch
    max_num = hf_config.max_dynamic_patch
    max_image_width = max_num * image_size
    max_image_height = min_num * image_size

    mm_data = dummy_image_for_clip(
        vision_config,
        num_images,
        image_width_override=max_image_width,
        image_height_override=max_image_height,
    )

    return seq_data, mm_data


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_internvl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_internvl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_internvl)
class InternVLChatModel(nn.Module, SupportsMultiModal):

    def __init__(self,
                 config: PretrainedConfig,
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
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        vision_feature_layer = self.select_layer
        if vision_feature_layer < 0:
            num_hidden_layers = config.vision_config.num_hidden_layers \
                + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1
        self.vision_model = InternVisionModel(
            config.vision_config, num_hidden_layers_override=num_hidden_layers)

        self.language_model = init_vllm_registered_model(
            config.text_config, cache_config, quant_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size), nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size))

        self.img_context_token_id = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_token_id = kwargs.pop("image_token_id", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        self.img_context_token_id = image_token_id[0]

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")
            # We need to flatten (B, N, P) to (B*N*P),
            # so we call flatten_bn twice.
            return InternVLImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(flatten_bn(pixel_values), concat=True)),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None
        image_embeds = self.extract_feature(image_input["data"])

        return image_embeds

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
        if image_input is not None and get_pp_group().is_first_rank:
            inputs_embeds = self.language_model.model.get_input_embeddings(
                input_ids)
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.img_context_token_id)
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # prepare weight iterators for components
        vit_weights, mlp_weights, llm_weights = itertools.tee(weights, 3)

        # load vision encoder
        vit_weights = filter_weights(vit_weights, "vision_model")
        self.vision_model.load_weights(vit_weights)

        # load mlp projector
        mlp_weights = filter_weights(mlp_weights, "mlp1")
        mlp_params_dict = dict(self.mlp1.named_parameters())
        for name, loaded_weight in mlp_weights:
            param = mlp_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        # load llm backbone
        llm_weights = filter_weights(llm_weights, "language_model")
        self.language_model.load_weights(llm_weights)
