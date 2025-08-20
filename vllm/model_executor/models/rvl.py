# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Literal, Optional, Protocol, TypedDict, Union
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, MultiModalData
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from transformers.models.auto import AutoConfig


from vllm.config import VllmConfig
from .utils import WeightsMapper
                    

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   VideoEmbeddingItems, VideoProcessorItems)
from vllm.multimodal import MultiModalDataDict
from .llava_onevision import LlavaOnevisionForConditionalGeneration, LlavaOnevisionMultiModalProcessor, LlavaOnevisionProcessingInfo, LlavaOnevisionLikeConfig
from .llava_next import LlavaNextProcessingInfo, LlavaDummyInputsBuilder


class RProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,

        },
        "image_kwargs": {},
    }


class RProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_processor_type",
        "vision_feature_select_strategy",
        "image_token",
        "vision_aspect_ratio",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        vision_aspect_ratio= "anyres",
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.vision_aspect_ratio = vision_aspect_ratio
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        **kwargs: Unpack[RProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            RProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            batch_num_images = iter(image_inputs["batch_num_images"])
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0]),
                channel_dim=output_kwargs["images_kwargs"].get("data_format"),
            )
            text, num_image_tokens = self._expand_image_tokens(
                text, image_sizes, height, width, self.image_token, batch_num_images
            )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

    
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        image_sizes: Iterable[Union[list[int], int]],
        height: int,
        width: int,
        special_token: str,
        batch_num_images: Iterable[int],
    ):
  
        prompt_strings = []
        max_num_vision_tokens = 0
        for sample in text:
            if special_token in sample:
                is_multi_image = next(batch_num_images) != 1
            else:
                is_multi_image = False
            while special_token in sample:
                if is_multi_image:
                    num_image_tokens = self.num_image_tokens + 1  # one for image_newline
                else:
                    original_size = next(image_sizes)
                    if not isinstance(original_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        original_size = original_size.tolist()
                    orig_height, orig_width = original_size
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                max_num_vision_tokens = max(max_num_vision_tokens, num_image_tokens)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                sample = sample.replace(special_token, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        text = [sample.replace("<placeholder>", special_token) for sample in prompt_strings]
        return text, max_num_vision_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
  
        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens 
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        
        return (unpadded_features, newline_features)


    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = RProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class RVLProcessingInfo(LlavaNextProcessingInfo):
    
    def get_hf_config(self) -> LlavaOnevisionLikeConfig:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(RProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}
    
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = int(
                round(original_height * (current_width / original_width), 7))
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7))
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

    
        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        # NOTE: This hardcoded value is found via processor tests
        return ImageSize(width=1153, height=944)

    def _get_num_frame_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        spatial_pool_stride = getattr(hf_config, "spatial_pool_stride", 2)

        vision_encoder_info = self.get_vision_encoder_info()
        patch_grid_length = vision_encoder_info.get_patch_grid_length()
        pooled_grid_length = math.ceil(patch_grid_length / spatial_pool_stride)

        return pooled_grid_length * pooled_grid_length
        
    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )

class RVLDummyInputsBuilder(
        LlavaDummyInputsBuilder[RVLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images 

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()


        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
        }



from transformers.activations import GELUActivation
class RVLMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("Using MultiModalProjector_withLayerNorm")
        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-06)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_feature: torch.Tensor) -> torch.Tensor:
        image_feature = self.pre_norm(image_feature)
        hidden_states = self.linear_1(image_feature)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states



@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevisionMultiModalProcessor,
    info=RVLProcessingInfo,
    dummy_inputs=RVLDummyInputsBuilder)
class RForConditionalGeneration(LlavaOnevisionForConditionalGeneration):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.multi_modal_projector = RVLMultiModalProjector(config)
 