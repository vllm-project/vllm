# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from functools import cached_property
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.intern_vit import (InternVisionModel,
                                                   InternVisionPatchModel)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptReplacementDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .clip import get_clip_num_patches
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


def _get_max_dynamic_patch(
    *,
    dynamic_image_size: bool,
    max_dynamic_patch: int,
    use_thumbnail: bool,
) -> int:
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch > 1:
        max_dynamic_patch += 1

    return max_dynamic_patch


def get_internvl_num_patches(hf_config: PretrainedConfig):
    vision_config = hf_config.vision_config
    downsample_ratio = hf_config.downsample_ratio
    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    return int(
        get_clip_num_patches(image_size=image_size, patch_size=patch_size) *
        (downsample_ratio**2))


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
    return torch.stack(pixel_values)


class InternVLProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        if max_dynamic_patch is None:
            max_dynamic_patch = config.max_dynamic_patch
        assert isinstance(max_dynamic_patch, int)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_patch: int = config.min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: int,
    ) -> str:
        return IMG_CONTEXT * feature_size

    def get_image_repl_full(self, feature_size: int, num_patches: int) -> str:
        features = self.get_image_repl_features(feature_size, num_patches)
        return IMG_START + features + IMG_END

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
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

        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        max_dynamic_patch = _get_max_dynamic_patch(
            dynamic_image_size=dynamic_image_size,
            max_dynamic_patch=max_dynamic_patch,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values = [
                image_to_pixel_values(
                    image,
                    input_size=self.image_size,
                    min_num=self.min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    use_thumbnail=self.use_thumbnail,
                ) for image in images
            ]
            image_inputs = {"pixel_values": torch.stack(pixel_values)}

            for image, item in zip(images, image_inputs["pixel_values"]):
                num_patches = item.shape[0]
                width, height = image.size
                num_blocks, _, _ = calculate_num_blocks(
                    orig_width=width,
                    orig_height=height,
                    image_size=self.image_size,
                    min_num=self.min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    use_thumbnail=self.use_thumbnail,
                )
                feature_size = num_blocks * num_patches

                image_repl = self.get_image_repl_full(feature_size,
                                                      num_patches)
                text = [t.replace('<image>', image_repl, 1) for t in text]

        text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class InternVLProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> InternVLProcessor:
        return InternVLProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[InternVLProcessor],
    ) -> int:
        hf_config = self.get_hf_config()

        if processor is None:
            min_dynamic_patch = hf_config.min_dynamic_patch
            max_dynamic_patch = hf_config.max_dynamic_patch
            dynamic_image_size = hf_config.dynamic_image_size
        else:
            min_dynamic_patch = processor.min_dynamic_patch
            max_dynamic_patch = processor.max_dynamic_patch
            dynamic_image_size = processor.dynamic_image_size

        max_dynamic_patch = _get_max_dynamic_patch(
            dynamic_image_size=dynamic_image_size,
            max_dynamic_patch=max_dynamic_patch,
            use_thumbnail=False,  # Applied in calculate_num_blocks
        )

        num_patches = get_internvl_num_patches(hf_config)
        num_blocks, _, _ = calculate_num_blocks(
            orig_width=image_width,
            orig_height=image_height,
            min_num=min_dynamic_patch,
            max_num=max_dynamic_patch,
            image_size=hf_config.vision_config.image_size,
            use_thumbnail=hf_config.use_thumbnail,
        )

        return num_blocks * num_patches

    def get_max_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        num_patches = get_internvl_num_patches(hf_config)
        max_dynamic_patch = _get_max_dynamic_patch(
            dynamic_image_size=hf_config.dynamic_image_size,
            max_dynamic_patch=hf_config.max_dynamic_patch,
            use_thumbnail=hf_config.use_thumbnail,
        )

        return num_patches * max_dynamic_patch

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        image_size = hf_config.vision_config.image_size
        max_dynamic_patch = _get_max_dynamic_patch(
            dynamic_image_size=hf_config.dynamic_image_size,
            max_dynamic_patch=hf_config.max_dynamic_patch,
            use_thumbnail=hf_config.use_thumbnail,
        )

        width = image_size * max_dynamic_patch
        height = image_size
        return ImageSize(width=width, height=height)


class InternVLDummyInputsBuilder(BaseDummyInputsBuilder[InternVLProcessingInfo]
                                 ):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="".join(f"Image-{i}: <image>\n"
                                for i in range(1, num_images + 1)),
            mm_data=mm_data,
        )


class InternVLMultiModalProcessor(
        BaseMultiModalProcessor[InternVLProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        image_token_id = self.info.get_hf_processor(**mm_kwargs).image_token_id
        image_data = mm_data.get("images", [])
        assert isinstance(image_data, list)

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = [image_token_id
                                               ] * len(image_data)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        hf_config = self.info.get_hf_config()
        num_patches = get_internvl_num_patches(hf_config)

        def get_replacement_internvl(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            return PromptReplacementDetails(
                full=hf_processor.get_image_repl_full(feature_size,
                                                      num_patches),
                features=hf_processor.get_image_repl_features(
                    feature_size, num_patches),
            )

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_internvl,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=InternVLProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder)
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
