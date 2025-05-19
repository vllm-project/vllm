# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The vLLM team.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
from transformers import (BatchFeature, CLIPVisionConfig, PretrainedConfig,
                          ProcessorMixin)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, BoundPromptUpdate,
                                        PlaceholderFeaturesInfo,
                                        PromptReplacement, PromptUpdate)
# yapf: enable
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import CLIPVisionModel
from .interfaces import (MultiModalEmbeddings, SupportsMultiModal, SupportsPP,
                         SupportsQuant)
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 32044

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(dropout=0.0,
                                                     hidden_act="quick_gelu",
                                                     hidden_size=1024,
                                                     image_size=336,
                                                     intermediate_size=4096,
                                                     num_attention_heads=16,
                                                     num_channels=3,
                                                     num_hidden_layers=24,
                                                     patch_size=14,
                                                     projection_dim=768)


def _init_img_processor(hf_config: PretrainedConfig,
                        quant_config: Optional[QuantizationConfig],
                        prefix: str = "") -> CLIPVisionModel:
    clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
    layer_idx = hf_config.img_processor.get('layer_idx', -2)

    # Initialize the CLIP only up to the required feature layer
    if layer_idx < 0:
        num_hidden_layers = clip_config.num_hidden_layers + \
            layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    img_processor = CLIPVisionModel(
        clip_config,
        quant_config,
        num_hidden_layers_override=num_hidden_layers,
        prefix=prefix,
    )

    return img_processor


class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class Phi3VImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


Phi3VImageInputs = Union[Phi3VImagePixelInputs, Phi3VImageEmbeddingInputs]


class Phi3ImageEmbeddingBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer_idx: int
        self.type_feature: str
        self.img_processor: CLIPVisionModel

    def get_img_features(self,
                         img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        TYPE_FEATURE = self.type_feature

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the img_processor
        img_feature = self.img_processor(img_embeds)

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig],
                 prefix: str = "") -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        self.img_processor = _init_img_processor(
            config, quant_config, prefix=f"{prefix}.img_processor")

        image_dim_out = config.img_processor['image_dim_out']
        self.num_img_tokens = config.img_processor['num_img_tokens']

        self.image_dim_out = image_dim_out

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.embd_layer.get('use_hd_transform',
                                                      False)
        self.with_learnable_separator = config.embd_layer.get(
            'with_learnable_separator', False)
        self.hd_transform_order = config.embd_layer.get(
            'hd_transform_order', 'glb_sub')
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform and self.with_learnable_separator

        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(torch.empty([1, 1, self.image_dim_out * 4]))
        self.sub_GN = nn.Parameter(
            torch.empty([1, 1, 1, self.image_dim_out * 4]))

        dim_projection = hidden_size
        depth = 2
        layers = [nn.Linear(image_dim_out * 4, dim_projection)]
        for _ in range(1, depth):
            layers.extend(
                [nn.GELU(),
                 nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.type_feature = config.img_processor.get('type_feature', 'patch')

    def forward(self, pixel_values: torch.FloatTensor,
                image_sizes: torch.Tensor) -> torch.FloatTensor:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        output: (num_images, num_img_tokens, hidden_size)
        """
        num_images, num_crops, c, h, w = pixel_values.shape
        pixel_values = pixel_values.flatten(0, 1)
        img_features = self.get_img_features(pixel_values)
        img_features = img_features.reshape(num_images, num_crops, -1,
                                            self.image_dim_out)
        image_features_proj = self.hd_feature_transform(
            img_features, image_sizes)
        return image_features_proj

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (
            self.hd_transform_order == 'sub_glb'
        ), f'hd_transform_order `{self.hd_transform_order}` not implemented'
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:,
                                               0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(
            global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(
            global_image_features_hd)

        batch_image_features_proj = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1:1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop)
            sub_image_features_hd_newline = self.add_image_newline(
                sub_image_features_hd)

            # [sub features, separator, global features]
            image_embeddings = torch.cat([
                sub_image_features_hd_newline.squeeze(
                    0),  # (h_crop*12*(w_crop*12+1), 4096)
                self.glb_GN.squeeze(0),
                global_image_features_hd_newline[i],
            ])
            img_proj = self.img_projection(
                image_embeddings.to(target_device, target_dtype))
            batch_image_features_proj.append(img_proj)

        return batch_image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096)
        where h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 576 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(num_images, h_crop, w_crop, H // 2, H // 2,
                     -1)  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(num_images, h_crop * H // 2, w_crop * H // 2,
                     4 * C)  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1,
                                                -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings],
            dim=2).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline


class Phi3VProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        num_crops: Optional[int] = None,
        **kwargs: object,
    ) -> ProcessorMixin:
        if num_crops is not None:
            kwargs["num_crops"] = num_crops

        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.calc_num_image_tokens_from_image_size(  # type: ignore
            width=image_width,
            height=image_height,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        # Result in the max possible feature size (h:w = 16:1)
        return ImageSize(height=8000, width=50)


class Phi3VDummyInputsBuilder(BaseDummyInputsBuilder[Phi3VProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        hf_processor = self.info.get_hf_processor()
        image_tokens: list[str] = hf_processor.img_tokens  # type: ignore

        return "".join(image_tokens[:num_images])

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
                                   num_images=num_images)
        }


class Phi3VMultiModalProcessor(BaseMultiModalProcessor[Phi3VProcessingInfo]):

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

        input_ids = processed_outputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)

        # Phi3v processor has inserted -1, -2 etc as placeholder in prompt_ids,
        # which will cause OverflowError when decoding the prompt_ids.
        # Therefore, we need to do an early replacement here
        input_ids.masked_fill_(input_ids < 0, _IMAGE_TOKEN_ID)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_tokens: list[str] = hf_processor.img_tokens  # type: ignore

        def get_replacement_phi3v(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            return [_IMAGE_TOKEN_ID] * num_image_tokens

        num_images = mm_items.get_count("image", strict=False)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_phi3v,
            ) for image_token in image_tokens[:num_images]
        ]

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
        mm_item_counts: Mapping[str, int],
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        # align to hf behavior when there are images
        if len(mm_item_counts):
            tokenizer = self.info.get_tokenizer()
            # to decode token_ids to the original text, we need to
            # 1. remove the first bos token
            # 2. remove space after each special token
            #    introduced by the tokenizer
            if len(token_ids) and token_ids[0] == tokenizer.bos_token_id:
                token_ids = token_ids[1:]
            text = tokenizer.decode(token_ids)
            for special_tokens in tokenizer.special_tokens_map.values():
                if isinstance(special_tokens, str):
                    text = text.replace(f"{special_tokens} ", special_tokens)
                elif isinstance(special_tokens, list):
                    for special_token in special_tokens:
                        text = text.replace(f"{special_token} ", special_token)
            # perform hf behavior
            # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/64f88b6/processing_phi3_v.py#L407
            pattern = r"<\|image_\d+\|>"
            prompt_chunks = [
                tokenizer(chunk).input_ids
                for chunk in re.split(pattern, text)
            ]
            image_tags = [
                tokenizer(chunk, add_special_tokens=False).input_ids
                for chunk in re.findall(pattern, text)
            ]
            if len(prompt_chunks) > len(image_tags):
                image_tags.append([])
            token_ids = [
                e for sublist in zip(prompt_chunks, image_tags)
                for ele in sublist for e in ele
            ]

        token_ids, text, placeholders = super()._apply_prompt_updates(
            token_ids=token_ids,
            mm_prompt_updates=mm_prompt_updates,
            mm_item_counts=mm_item_counts,
        )

        # Keep the behavior in line with HF processor
        if text.startswith("<s> <|image|>"):
            text = text.replace("<s> <|image|>", "<s><|image|>", 1)
            token_ids = [token_ids[0], *token_ids[2:]]
            placeholders = {
                modality: [
                    PlaceholderFeaturesInfo(
                        modality=p.modality,
                        item_idx=p.item_idx,
                        start_idx=p.start_idx - 1,
                        tokens=p.tokens,
                        is_embed=p.is_embed,
                    ) for p in ps
                ]
                for modality, ps in placeholders.items()
            }

        return token_ids, text, placeholders


@MULTIMODAL_REGISTRY.register_processor(Phi3VMultiModalProcessor,
                                        info=Phi3VProcessingInfo,
                                        dummy_inputs=Phi3VDummyInputsBuilder)
class Phi3VForCausalLM(nn.Module, SupportsMultiModal, SupportsPP,
                       SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_embed_tokens.wte": "embed_tokens",
            "model.vision_embed_tokens.": "vision_embed_tokens.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.image_token_id = _IMAGE_TOKEN_ID

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "model.embed_tokens"),
        )

        # TODO: Optionally initializes this for supporting input embeddings.
        self.vision_embed_tokens = Phi3HDImageEmbedding(
            config,
            self.quant_config,
            prefix=maybe_prefix(prefix, "model.vision_embed_tokens"))

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            # The prefix is empty intentionally because default prefix of
            # LlamaForCausalLM is "model"
            prefix="",
            # We don't directly initialize vLLM's LlamaForCausalLM so we
            # can automatically apply embedding wrapper if this model is
            # initialized as an embedding model
            architectures=["LlamaForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, list[torch.Tensor]]
    ) -> Union[torch.Tensor, list[torch.Tensor]]:

        h = w = CLIP_VIT_LARGE_PATCH14_336_CONFIG.image_size
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
            self, **kwargs: object) -> Optional[Phi3VImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return Phi3VImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)))

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return Phi3VImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Phi3VImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            image_data = image_input["data"]
            if is_list_of(image_data, torch.Tensor):
                # it's already a list of tensors
                return image_data
            if len(image_data.shape) == 3:
                # 3D tensor
                return list(torch.unbind(image_data, dim=0))
            raise ValueError(
                "We expect batched 2D tensors; "
                "this can be either a list of 2D tensors or a single 3D tensor."
            )

        assert self.vision_embed_tokens is not None
        image_embeds = self.vision_embed_tokens(image_input["data"],
                                                image_input["image_sizes"])

        return image_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_token_id)
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights,
                                                 mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")
        return autoloaded_weights
