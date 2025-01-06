# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import re
from functools import cached_property, partial
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.intern_vit import (InternVisionModel,
                                                   InternVisionPatchModel)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import NestedTensors, PlaceholderRange
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   get_clip_num_patches)
from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

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
    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    """


class InternVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: NestedTensors
    """ 
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

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


def calculate_num_blocks_wrapper(
    hf_config: PretrainedConfig,
    max_dynamic_patch: Optional[int] = None,
    dynamic_image_size: Optional[bool] = None,
):
    if dynamic_image_size is None:
        dynamic_image_size = hf_config.dynamic_image_size

    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1
    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch
    min_num = hf_config.min_dynamic_patch
    image_size = hf_config.vision_config.image_size
    use_thumbnail = hf_config.use_thumbnail
    return partial(calculate_num_blocks,
                   min_num=min_num,
                   max_num=max_dynamic_patch,
                   image_size=image_size,
                   use_thumbnail=use_thumbnail)


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


def image_to_pixel_values_wrapper(
    hf_config: PretrainedConfig,
    max_dynamic_patch: Optional[int] = None,
    dynamic_image_size: Optional[bool] = None,
):
    image_size = hf_config.vision_config.image_size
    min_num = hf_config.min_dynamic_patch
    if dynamic_image_size is None:
        dynamic_image_size = hf_config.dynamic_image_size

    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1
    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch
    use_thumbnail = hf_config.use_thumbnail
    return partial(image_to_pixel_values,
                   input_size=image_size,
                   min_num=min_num,
                   max_num=max_dynamic_patch,
                   use_thumbnail=use_thumbnail)


def get_internvl_num_patches(hf_config: PretrainedConfig):
    vision_config = hf_config.vision_config
    downsample_ratio = hf_config.downsample_ratio
    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    return int(
        get_clip_num_patches(image_size=image_size, patch_size=patch_size) *
        (downsample_ratio**2))


def get_max_internvl_image_tokens(
    ctx: InputContext,
    *,
    max_dynamic_patch: Optional[int] = None,
    dynamic_image_size: Optional[bool] = None,
):
    hf_config = ctx.get_hf_config()
    if dynamic_image_size is None:
        dynamic_image_size = hf_config.dynamic_image_size

    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1
    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch
    use_thumbnail = hf_config.use_thumbnail
    if use_thumbnail and max_dynamic_patch > 1:
        max_dynamic_patch += 1

    num_patches = get_internvl_num_patches(hf_config)
    return num_patches * max_dynamic_patch


def get_max_internvl_image_size(
    ctx: InputContext,
    *,
    max_dynamic_patch: Optional[int] = None,
    dynamic_image_size: Optional[bool] = None,
):
    hf_config = ctx.get_hf_config()
    image_size = hf_config.vision_config.image_size
    if dynamic_image_size is None:
        dynamic_image_size = hf_config.dynamic_image_size

    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1
    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch
    use_thumbnail = hf_config.use_thumbnail
    if use_thumbnail and max_dynamic_patch > 1:
        max_dynamic_patch += 1
    width = image_size * max_dynamic_patch
    height = image_size
    return width, height


class InternVLInputPipeline:

    def __init__(
        self,
        img_start_token: str,
        img_end_token: str,
        img_context_token: str,
    ) -> None:
        super().__init__()

        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        self.img_context_token = img_context_token

    def _create_image_prompt(self, feature_size: int, num_patches: int) -> str:
        return (self.img_start_token + self.img_context_token * feature_size +
                self.img_end_token)

    def _expand_image_prompt(
        self,
        prompt: str,
        feature_sizes: List[int],
        num_patches: int,
    ) -> str:
        image_idx = sorted(
            map(int, re.findall(r"Image-(\d+): <image>\n", prompt)))

        new_prompt = prompt
        for idx, feature_size in enumerate(feature_sizes, start=1):
            image_prompt = self._create_image_prompt(feature_size, num_patches)
            if not image_idx:
                image_prompt = f"Image-{idx}: {image_prompt}"

            new_prompt = new_prompt.replace('<image>', image_prompt, 1)

        return new_prompt

    def input_processor(
        self,
        ctx: InputContext,
        inputs: DecoderOnlyInputs,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> DecoderOnlyInputs:
        multi_modal_data = inputs.get("multi_modal_data")
        if multi_modal_data is None or "image" not in multi_modal_data:
            return inputs

        model_config = ctx.model_config
        hf_config = ctx.get_hf_config()

        image_data = multi_modal_data["image"]
        num_patches = get_internvl_num_patches(hf_config)
        num_blocks_calculator = calculate_num_blocks_wrapper(
            hf_config, max_dynamic_patch, dynamic_image_size)
        if isinstance(image_data, Image.Image):
            width, height = image_data.size
            num_blocks, _, _ = num_blocks_calculator(width, height)
            image_feature_sizes = [num_blocks * num_patches]
        elif is_list_of(image_data, Image.Image):
            image_feature_sizes = []
            for image in image_data:
                width, height = image.size
                num_blocks, _, _ = num_blocks_calculator(width, height)
                image_feature_sizes.append(num_blocks * num_patches)
        elif isinstance(image_data, torch.Tensor):
            num_images, image_feature_size, hidden_size = image_data.shape
            image_feature_sizes = [image_feature_size]
        else:
            raise TypeError(f"Invalid image type: {type(image_data)}")

        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code)

        prompt = inputs.get("prompt")
        prompt_token_ids = inputs["prompt_token_ids"]
        if prompt is None:
            prompt = tokenizer.decode(prompt_token_ids)

        new_prompt = self._expand_image_prompt(prompt, image_feature_sizes,
                                               num_patches)
        new_prompt_token_ids = tokenizer.encode(new_prompt)
        img_context_token_id = tokenizer.encode(self.img_context_token,
                                                add_special_tokens=False)
        assert len(img_context_token_id) == 1, \
            (f"Invalid image token '{self.img_context_token}': A valid image "
            f"token encodes to a single token ID, got {img_context_token_id}.")
        img_context_token_id = img_context_token_id[0]

        # Get precise tracking of placeholder positions
        token_idx = image_idx = 0
        placeholder_ranges = []
        while token_idx < len(new_prompt_token_ids):
            if new_prompt_token_ids[token_idx] == img_context_token_id:
                curr_image_featue_size = image_feature_sizes[image_idx]
                placeholder_ranges.append(
                    PlaceholderRange(offset=token_idx,
                                     length=curr_image_featue_size))
                image_idx += 1
                token_idx += curr_image_featue_size
            else:
                token_idx += 1

        return token_inputs(
            prompt=prompt,
            prompt_token_ids=new_prompt_token_ids,
            multi_modal_data=multi_modal_data,
            multi_modal_placeholders={"image": placeholder_ranges})

    def input_mapper(
        self,
        ctx: InputContext,
        data: object,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ):
        hf_config = ctx.get_hf_config()

        image_pixel_values_mapper = image_to_pixel_values_wrapper(
            hf_config, max_dynamic_patch, dynamic_image_size)
        if isinstance(data, Image.Image):
            data = image_pixel_values_mapper(data)
            # Add an N dimension for number of images per prompt (currently 1).
            data = data.unsqueeze(0)
        elif is_list_of(data, Image.Image):
            # we can't stack here because images may have different num_patches
            data = [image_pixel_values_mapper(img) for img in data]
        else:
            return MultiModalKwargs({"image_embeds": data})
        model_config = ctx.model_config
        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code)
        image_token_id = tokenizer.encode(self.img_context_token,
                                          add_special_tokens=False,
                                          return_tensors="pt")[0]

        return MultiModalKwargs({
            "pixel_values": data,
            "image_token_id": image_token_id
        })

    def dummy_data(
        self,
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ):
        num_images = mm_counts["image"]

        hf_config = ctx.get_hf_config()

        image_feature_size = get_max_internvl_image_tokens(
            ctx,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )
        model_config = ctx.model_config
        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code)

        seq_data, ranges = dummy_seq_data_for_clip(
            hf_config.vision_config,
            seq_len,
            num_images,
            image_token_id=tokenizer.encode(self.img_context_token,
                                            add_special_tokens=False)[0],
            image_feature_size_override=image_feature_size,
        )

        max_image_width, max_image_height = get_max_internvl_image_size(
            ctx,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

        mm_data = dummy_image_for_clip(
            hf_config.vision_config,
            num_images,
            image_width_override=max_image_width,
            image_height_override=max_image_height,
        )

        return DummyData(seq_data, mm_data, ranges)


input_pipeline = InternVLInputPipeline(IMG_START, IMG_END, IMG_CONTEXT)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_pipeline.input_mapper)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(input_pipeline.dummy_data)
@INPUT_REGISTRY.register_input_processor(input_pipeline.input_processor)
class InternVLChatModel(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.llm_arch_name = config.text_config.architectures[0]
        self.is_mono = self.llm_arch_name == 'InternLM2VEForCausalLM'
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            is_mono=self.is_mono,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None
        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and \
                (llm_quant_config is not None):
                quant_config.modules_to_not_convert.append("vision_model")

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = config.vision_config.num_hidden_layers \
                    + vision_feature_layer + 1
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            return InternVisionPatchModel(config.vision_config)

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

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

            patches_per_image = []
            for request_pixel_values in pixel_values:
                for image_pixel_values in request_pixel_values:
                    patches_per_image.append(image_pixel_values.shape[0])
            # We need to flatten (B, N, P) to (B*N*P),
            # so we call flatten_bn twice.
            return InternVLImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(flatten_bn(pixel_values), concat=True)),
                patches_per_image=patches_per_image)

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> Tuple[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["data"])

        patches_per_image = image_input["patches_per_image"]

        # Only one image in the current batch
        if len(patches_per_image) == 1:
            image_embeds = image_embeds.view(
                -1, self.config.text_config.hidden_size).unsqueeze(0)
            return image_embeds

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in patches_per_image
        ]
        image_embeds = image_embeds.split(image_feature_sizes)
        return image_embeds

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.is_mono:
            self.visual_token_mask = (
                input_ids == self.img_context_token_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None

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
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            assert self.img_context_token_id is not None
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.img_context_token_id)
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
    ) -> Union[SamplerOutput, IntermediateTensors]:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
