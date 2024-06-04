from typing import Dict, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)

from vllm.config import CacheConfig, ModelConfig, VisionLanguageConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.image import ImagePixelData, get_dummy_image_data
from vllm.sequence import SequenceData

from .llava import (LlavaForConditionalGeneration, LlavaImageFeatureInputs,
                    LlavaImagePixelInputs)


class ImageSizesMixin(TypedDict, total=False):
    image_sizes: torch.Tensor
    """Shape: (batch_size, 2)"""


class LlavaNextImagePixelInputs(ImageSizesMixin, LlavaImagePixelInputs):
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, num_channels, height, width)"""


class LlavaNextImageFeatureInputs(ImageSizesMixin, LlavaImageFeatureInputs):
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, image_feature_size, hidden_size)"""


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageFeatureInputs]


def _get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    seq_data, fake_mm_data = get_dummy_image_data(seq_len, model_config,
                                                  vlm_config)

    config_input_type = vlm_config.image_input_type
    ImageInputType = VisionLanguageConfig.ImageInputType

    if config_input_type == ImageInputType.PIXEL_VALUES:
        _, c, h, w = vlm_config.image_input_shape
        mode = {1: "L", 3: "RGB"}[c]
        fake_mm_data = ImagePixelData(Image.new(mode, (w, h), color=0))

    return seq_data, fake_mm_data


def _image_pixel_processor(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Dict[str, torch.Tensor]:
    image = data.image

    if isinstance(image, torch.Tensor):
        pixel_values = image.to(model_config.dtype)
        batch_size, _, _, h, w = pixel_values.shape
        image_sizes = torch.tensor([(w, h) for _ in range(batch_size)])

        return {"pixel_values": pixel_values, "image_sizes": image_sizes}

    return MULTIMODAL_REGISTRY._get_plugin_for_data_type(ImagePixelData) \
        ._default_input_processor(data, model_config, vlm_config)


@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input(_image_pixel_processor)
@MULTIMODAL_REGISTRY.register_dummy_data(_get_dummy_image_data)
class LlavaNextForConditionalGeneration(LlavaForConditionalGeneration):
    """
    Args to `forward()`:
        input_ids: Flattened (concatenated) input_ids corresponding to a
            batch.
        pixel_values: For PIXEL_VALUES, expects a batch with shape
            [1, num_patches, 3, 336, 336].
        image_features: For IMAGE_FEATURES, expects a batch with shape
            [1, num_patches, 1176, 1024].
    """

    def __init__(self,
                 config: LlavaNextConfig,
                 vision_language_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(
            config,  # type: ignore
            vision_language_config,
            cache_config,
            quant_config,
        )

        # Update the type annotation from that of its superclass
        self.config = config

        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))

    def _validate_image_pixels(self, data: torch.Tensor) -> torch.Tensor:
        _, num_channels, _, _ = self.vision_language_config.image_input_shape

        # Note that this is different from that of vLLM vision_language_config
        # since the image is resized by the HuggingFace preprocessor
        height = width = self.config.vision_config.image_size

        if list(data.shape[2:]) != [num_channels, height, width]:
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"num_patches plus {[num_channels, height, width]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected image sizes shape is batch dimension plus "
                f"{[2]}. You supplied {data.shape}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_features = kwargs.pop("image_features", None)

        expected_input_type = self.vision_language_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if expected_input_type == ImageInputType.PIXEL_VALUES:
            if image_features is not None:
                raise ValueError(
                    "Expected pixel values but got image features")
            if pixel_values is None:
                return None

            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, torch.Tensor):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaNextImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixels(pixel_values),
                image_sizes=self._validate_image_sizes(image_sizes),
            )

        if expected_input_type == ImageInputType.IMAGE_FEATURES:
            if pixel_values is not None:
                raise ValueError(
                    "Expected image features but got pixel values")
            if image_features is None:
                return None

            if not isinstance(image_features, torch.Tensor):
                raise ValueError("Incorrect type of image features. "
                                 f"Got type: {type(image_features)}")

            return LlavaNextImageFeatureInputs(
                type="image_features",
                data=self._validate_image_data(image_features),
            )

        return None

    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            orig_width, orig_height = image_size
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # image_aspect_ratio == "anyres"
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    (orig_width, orig_height),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                other_patch_embeds = other_patch_embeds \
                    .view(num_patch_width, num_patch_height, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     image_size)
                    other_patch_embeds = torch.cat((
                        other_patch_embeds,
                        self.image_newline[:, None, None] \
                            .expand(*other_patch_embeds.shape[:-1], 1) \
                            .to(other_patch_embeds.device),
                    ), dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
            self, inputs: LlavaNextImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        b, num_patches, c, h, w = pixel_values.shape
        stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)

        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return stacked_image_features.view(b, num_patches,
                                           *stacked_image_features.shape[-2:])

    def _process_image_input(
            self, image_input: LlavaNextImageInputs) -> torch.Tensor:
        patch_embeddings = super()._process_image_input(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = image_input["data"].shape[0]
            default_width, default_height = self.config.vision_config.image_size
            image_sizes = torch.as_tensor([[default_width, default_height]
                                           for _ in range(batch_size)])

        merged_patch_embeddings = [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features,
                                               strategy="spatial_unpad")
            for i, patch_features in enumerate(patch_embeddings)
        ]

        return torch.stack(merged_patch_embeddings, dim=0)
