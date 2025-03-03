# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Idefics3 model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import (BatchFeature, Idefics3Config, Idefics3ImageProcessor,
                          Idefics3Processor)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        MultiModalDataItems,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

# yapf: disable
from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics3VisionTransformer)
# yapf: enable
from .interfaces import SupportsLoRA, SupportsMultiModal
from .llama import LlamaModel
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)


class Idefics3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_patches, 
             num_channels, height, width)`
    """
    pixel_attention_mask: Optional[torch.BoolTensor]


class Idefics3ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
    `hidden_size` must match the hidden size of language model backbone.
    """


ImageInputs = Union[Idefics3ImagePixelInputs, Idefics3ImageEmbeddingInputs]


class Idefics3ProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        size: Optional[Dict[str, int]] = None,
        **kwargs: object,
    ) -> Idefics3Processor:
        if size is not None:
            kwargs["size"] = size

        return self.ctx.get_hf_processor(Idefics3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_processor = self.get_hf_processor()
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        grid_w, grid_h = self._get_image_feature_grid_size(
            image_width=image_processor.size['longest_edge'],
            image_height=image_processor.size['longest_edge'],
        )
        num_image_token = (grid_w * grid_h + 1) * hf_processor.image_seq_len
        # Calculate Non-image-token length
        # NOTE: <row_1_col_1> and <global-img> are special token for SmolVLM
        # but not for Idefic3, so we need to tokenize them to get actual length.
        tokenizer = self.get_tokenizer()
        tile_token_len = len(tokenizer.tokenize("<row_1_col_1>"))
        glob_token_len = len(tokenizer.tokenize(hf_processor.global_image_tag))
        # linebreak and <fake_token_around_image> always cost 1 token
        fake_token_len = lb_len = 1
        non_image_token = (grid_w * grid_h) * (
            tile_token_len + fake_token_len) + glob_token_len + (
                grid_h + 1) * lb_len + fake_token_len
        return {"image": num_image_token + non_image_token}

    def _resize_output_size(self,
                            *,
                            height: int,
                            width: int,
                            max_len: Optional[int] = None,
                            min_len: Optional[int] = 1,
                            max_size: Optional[int] = None) -> tuple[int, int]:
        # Set default value for max_len if not provided
        max_len = max(height, width) if max_len is None else max_len
        aspect_ratio = width / height

        # Handle the maximum size constraint
        if max_size is not None:
            max_len = min(max_len, max_size)

        # Adjust dimensions according to the aspect ratio
        if width >= height:
            width = max_len
            height = int(width / aspect_ratio)
        else:
            height = max_len
            width = int(height * aspect_ratio)

        # Ensure both width and height are even (if needed)
        height += height % 2
        width += width % 2

        # Ensure dimensions are not smaller than the minimum length
        height = max(height, min_len)
        width = max(width, min_len)

        return height, width

    def _get_resize_output_image_size(
        self,
        *,
        image_width: int,
        image_height: int,
        resolution_max_side: int,
    ) -> tuple[int, int]:
        hf_processor = self.get_hf_processor()
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        max_image_size = image_processor.size['longest_edge']
        if resolution_max_side > max_image_size:
            raise ValueError(
                "`resolution_max_side` cannot be larger than `max_image_size`")

        height, width = image_height, image_width

        # Find the output size, when rescaling the longest edge to max_len and
        # preserving the aspect ratio
        height, width = self._resize_output_size(height=height,
                                                 width=width,
                                                 max_len=resolution_max_side)
        return height, width

    def _get_image_feature_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        size: Optional[dict[str, object]] = None,
    ) -> tuple[int, int]:
        hf_processor = self.get_hf_processor(size=size)
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        max_image_size = image_processor.max_image_size['longest_edge']
        size = image_processor.size['longest_edge']
        assert size % max_image_size == 0, (
            "`longest_edge` in image_processor's `size` must be divisible by "
            "`longest_edge` in `max_image_size`, this may be caused by "
            "incorrect mm_kwargs override.")

        resized_height, resized_width = self._get_resize_output_image_size(
            image_width=image_width,
            image_height=image_height,
            resolution_max_side=size,
        )
        if resized_height > max_image_size or resized_width > max_image_size:
            grid_h = math.ceil(resized_height / max_image_size)
            grid_w = math.ceil(resized_width / max_image_size)
        else:
            grid_h = grid_w = 0
        return grid_w, grid_h


class Idefics3DummyInputsBuilder(BaseDummyInputsBuilder[Idefics3ProcessingInfo]
                                 ):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        longest_edge = image_processor.max_image_size['longest_edge']
        image_token: str = hf_processor.image_token.content

        mm_data = {
            "image":
            self._get_dummy_images(width=longest_edge,
                                   height=longest_edge,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class Idefics3MultimodalProcessor(
        BaseMultiModalProcessor[Idefics3ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs)
            image_grids = [
                self.info._get_image_feature_grid_size(
                    image_width=img.width,
                    image_height=img.height,
                    **mm_kwargs,
                ) for img in mm_data["images"]
            ]
            image_patches = list(map(lambda x: math.prod(x) + 1, image_grids))
            for key in ("pixel_values", "pixel_attention_mask"):
                data = processed_outputs.pop(key)
                data = data.flatten(0, 1).split(image_patches)
                processed_outputs[key] = data
        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=True,
                                          return_tensors="pt")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_attention_mask=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token = hf_processor.image_token.content
        fake_image_token = hf_processor.fake_image_token.content
        global_img_token = hf_processor.global_image_tag
        image_seq_len = hf_processor.image_seq_len
        grid_placeholder = "<row_{n_h}_col_{n_w}>"

        p_img = image_token * image_seq_len
        global_img_placeholder = fake_image_token + global_img_token + p_img
        tile_img_placeholder = fake_image_token + grid_placeholder + p_img

        def get_replacement_idefics3(item_idx: int) -> str:
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)
            grid_w, grid_h = self.info._get_image_feature_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
                **hf_processor_mm_kwargs,
            )
            if grid_w == 0 and grid_h == 0:
                image_placeholder = global_img_placeholder
            else:
                tiles_placeholder = list[str]()
                for i in range(grid_h):
                    for j in range(grid_w):
                        placeholder_per_tile = tile_img_placeholder.format(
                            n_h=i + 1, n_w=j + 1)
                        tiles_placeholder.append(placeholder_per_tile)
                        # Add line break if it is the last tile in the row
                        if j == grid_w - 1:
                            tiles_placeholder.append("\n")

                image_placeholder = "".join(
                    [*tiles_placeholder, "\n", global_img_placeholder])
            return image_placeholder + fake_image_token

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_idefics3,
            )
        ]


class Idefics3SimpleMLP(nn.Module):

    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**
                                                         2)
        output_size = config.text_config.hidden_size
        self.proj = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x)
        return out


class Idefics3Connector(nn.Module):

    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )

    def pixel_shuffle(self,
                      x: torch.Tensor,
                      scale_factor: int = 2) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor),
                   embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)),
                      embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states,
                                                 self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Idefics3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Idefics3Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size
        self.vision_model = Idefics3VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"))
        self.connector = Idefics3Connector(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )
        self.text_model = LlamaModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "text_model"),
        )

        self.image_seq_len = int(
            ((config.vision_config.image_size //
              config.vision_config.patch_size)**2) / (config.scale_factor**2))
        self.image_token_id = self.config.image_token_id

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return Idefics3ImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if isinstance(pixel_values, list):
                pixel_values = torch.cat(pixel_values, dim=1)
                pixel_attention_mask = torch.cat(pixel_attention_mask, dim=1)
            else:
                pixel_values = flatten_bn(pixel_values)
                pixel_attention_mask = flatten_bn(pixel_attention_mask)

            return Idefics3ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
                pixel_attention_mask=pixel_attention_mask)

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> NestedTensors:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        num_patches = [x.size(0) for x in pixel_values]
        pixel_values = pixel_values.to(
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(
            dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2),
                      pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask[
                real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1,
                                                      size=patch_size,
                                                      step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2,
                                                 size=patch_size,
                                                 step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        return image_hidden_states.split(num_patches)

    def _process_image_pixels(
            self, inputs: Idefics3ImagePixelInputs) -> NestedTensors:
        assert self.vision_model is not None

        pixel_values = inputs["data"]
        pixel_attention_mask = inputs["pixel_attention_mask"]

        return self._image_pixels_to_features(pixel_values,
                                              pixel_attention_mask)

    def _process_image_input(self, image_input: ImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None
        image_features = self._process_image_pixels(image_input)
        num_patches = [x.size(0) for x in image_features]
        image_features = torch.cat(image_features)
        return self.connector(image_features).split(num_patches)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        hidden_states = self.text_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Idefics3MultimodalProcessor,
    info=Idefics3ProcessingInfo,
    dummy_inputs=Idefics3DummyInputsBuilder)
class Idefics3ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                       SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.model = Idefics3Model(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.image_token_id = self.config.image_token_id

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
        )
        if self.config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.text_model.wte.weight
        self.logits_processor = LogitsProcessor(config.text_config.vocab_size)
        self.sampler = get_sampler()

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self.model._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self.model._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
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

        hidden_states = self.model.text_model(input_ids,
                                              positions,
                                              intermediate_tensors,
                                              inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.text_model",
            connector="model.connector",
            tower_model="model.vision_model")
