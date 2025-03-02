# SPDX-License-Identifier: Apache-2.0
from functools import cached_property
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional, Set,
                    Tuple, TypedDict, Union)

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor, TensorType
from torch.nn.functional import gumbel_softmax, pad, softmax
from transformers import (BatchFeature, PretrainedConfig, PreTrainedTokenizer,
                          ProcessorMixin, SiglipImageProcessor,
                          SiglipVisionConfig)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.vision import get_vision_encoder_info
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ovis import (ConversationFormatter,
                                                  Llama3ConversationFormatter,
                                                  OvisConfig)
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from .interfaces import SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]


class OvisImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]


class OvisImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]


OvisImageInputs = Union[OvisImagePixelInputs, OvisImageEmbeddingInputs]


class OvisProcessor:

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
        image_processor: SiglipImageProcessor,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.visual_tokenizer = SiglipVisualTokenizer(config)

    def get_conversation_formatter(self) -> ConversationFormatter:
        return Llama3ConversationFormatter(self.tokenizer)
        # return GemmaConversationFormatter(self.tokenizer)

    @staticmethod
    def construct_image_placeholders(grid):
        image_placeholders = [
            IMAGE_INDICATOR_IDS[0],
            IMAGE_ATOM_ID,
            IMAGE_INDICATOR_IDS[1],
        ]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(IMAGE_ATOM_ID)
                    if c < grid[1] - 1:
                        image_placeholders.append(IMAGE_INDICATOR_IDS[2])
                if r < grid[0] - 1:
                    image_placeholders.append(IMAGE_INDICATOR_IDS[3])
        image_placeholders.append(IMAGE_INDICATOR_IDS[4])
        return image_placeholders

    def preprocess_image(
        self,
        image: Image.Image,
        max_partition=9,
        covering_threshold=0.9,
        convert_to_rgb=True,
    ):

        def _preprocess(img: Image.Image, side):
            # first resize and preprocess
            w, h = img.size
            if w == h:
                new_width = new_height = side
            elif w > h:
                new_width = side
                new_height = int(h / w * new_width)
            else:
                new_height = side
                new_width = int(w / h * new_height)
            new_size = dict(height=new_height, width=new_width)
            pixel_values = self.image_processor.preprocess(
                img, size=new_size, return_tensors="pt")["pixel_values"]

            # then pad to square
            square_values = torch.zeros([1, 3, side, side],
                                        dtype=pixel_values.dtype,
                                        device=pixel_values.device)
            new_height, new_width = pixel_values.shape[2:]
            if new_height == new_width:
                square_values[:, :, :, :] = pixel_values
            elif new_height > new_width:
                from_index = (side - new_width) // 2
                square_values[:, :, :, from_index:from_index +
                              new_width] = (pixel_values)
            else:
                from_index = (side - new_height) // 2
                square_values[:, :, from_index:from_index +
                              new_height, :] = (pixel_values)

            return square_values

        def _partition(img, grid):
            w, h = img.size
            row_height = h // grid[0]
            col_width = w // grid[1]

            partition = []
            for row in range(grid[0]):
                for col in range(grid[1]):
                    left = col * col_width
                    upper = row * row_height
                    right = w if col == grid[1] - 1 else (col + 1) * col_width
                    lower = h if row == grid[0] - 1 else (row + 1) * row_height
                    partition.append((left, upper, right, lower))

            return partition

        def _covering_area(left, upper, right, lower, side):
            w = right - left
            h = lower - upper
            w, h = max(w, h), min(w, h)
            if w > side:
                h = h / w * side
                w = side
            return w * h

        def _get_best_grid(img, side):
            img_area = img.size[0] * img.size[1]

            candidate_grids = []
            for i in range(1, max_partition + 1):
                for j in range(1, max_partition + 1):
                    if i * j <= max_partition:
                        candidate_grids.append((i, j))

            all_grids = []
            good_grids = []
            for grid in candidate_grids:
                partition = _partition(img, grid)
                covering_ratio = (
                    sum([_covering_area(*p, side)
                         for p in partition]) / img_area)
                assert covering_ratio <= 1.0
                all_grids.append((grid, covering_ratio))
                if covering_ratio > covering_threshold:
                    good_grids.append((grid, covering_ratio))

            if len(good_grids) > 0:
                # pick the good partition with minimum #sub_images and break the tie using covering_ratio
                return sorted(good_grids,
                              key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][0]
            else:
                # pick the partition with maximum covering_ratio and break the tie using #sub_images
                return sorted(all_grids,
                              key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]

        if convert_to_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError("get_image_size() returns non-square size")
        side = sides[0]
        grid = _get_best_grid(image, side)
        partition = _partition(image, grid)
        crops = [image.crop(p) for p in partition]
        if len(crops) > 1:
            crops.insert(0, image)
        pixel_values = torch.cat([_preprocess(crop, side) for crop in crops],
                                 dim=0)
        image_placeholders = self.construct_image_placeholders(grid)
        return pixel_values, image_placeholders

    def preprocess_inputs(
        self,
        text_or_conversations: Union[List[Dict], str],
        images: Optional[List[Image.Image]] = None,
        max_partition=9,
        generation_preface="",
        propagate_exception=True,
    ):
        images = images or []
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{"from": "human", "value": text_or_conversations}]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        else:
            raise ValueError(
                f"Invalid type of `text_or_conversations`, expected `List[Dict]` or `str`,"
                f" but got {type(text_or_conversations)}")

        # format conversations
        prompt, raw_input_ids, _ = self.get_conversation_formatter().format(
            conversations, generation_preface=generation_preface)

        # place image placeholders
        input_ids = []
        pixel_values = []
        image_token_indices = [
            i for i, v in enumerate(raw_input_ids) if v == IMAGE_TOKEN_ID
        ]
        last_image_token_index = -1
        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])
            if len(images) > 0:
                image = images[i]
                raw_pixel_values, image_placeholders = self.preprocess_image(
                    image, max_partition=max_partition)
            else:
                raw_pixel_values, image_placeholders = (
                    self.visual_tokenizer.mock_input())

            input_ids.extend(image_placeholders)
            pixel_values.append(raw_pixel_values)
        input_ids.extend(raw_input_ids[last_image_token_index + 1:])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        pixel_values = torch.cat(pixel_values,
                                 dim=0) if len(pixel_values) > 0 else None

        return prompt, input_ids, pixel_values

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        max_partition=9,
        generation_preface="",
        propagate_exception=True,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        _, text_inputs, pixel_values = self.preprocess_inputs(
            text,
            images,
            max_partition,
            generation_preface,
            propagate_exception,
        )

        image_inputs = {"pixel_values": pixel_values}
        return BatchFeature({
            **text_inputs,
            **image_inputs
        },
                            tensor_type=return_tensors)


class OvisProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> OvisConfig:
        return self.ctx.get_hf_config(OvisConfig)

    def get_hf_image_processor(self) -> ProcessorMixin:
        visual_tokenizer_config = self.get_hf_config().visual_tokenizer_config
        image_processor = visual_tokenizer_config.vision_config._name_or_path

        return cached_get_image_processor(image_processor)

    def get_hf_processor(self, **kwargs) -> OvisProcessor:
        return self.ctx.init_processor(
            OvisProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_hf_image_processor(),
            **kwargs,
        )

    def get_vision_encoder_info(self):
        visual_tokenizer_config = self.get_hf_config().visual_tokenizer_config
        return get_vision_encoder_info(visual_tokenizer_config)

    def get_num_image_tokens(self) -> int:
        vision_encoder_info = self.get_vision_encoder_info()
        image_size = vision_encoder_info.get_image_size()
        return vision_encoder_info.get_num_image_tokens(
            image_width=image_size, image_height=image_size)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len,
                                   mm_counts) -> Mapping[str, Optional[int]]:
        vision_encoder_info = self.get_vision_encoder_info()

        return {"image": vision_encoder_info.get_max_image_tokens()}

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(height=384, width=384)


class OvisDummyInputsBuilder(BaseDummyInputsBuilder[OvisProcessingInfo]):

    def get_dummy_processor_inputs(self, seq_len,
                                   mm_counts) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features(
        )

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="<image>" * num_images,
            mm_data=mm_data,
        )


class OvisMultiModalProcessor(BaseMultiModalProcessor[OvisProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"), )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return super()._call_hf_processor(prompt=prompt,
                                          mm_data=mm_data,
                                          mm_kwargs=mm_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        image_token_id = IMAGE_TOKEN_ID

        def get_replacement_ovis(image: Image.Image):
            _, image_placeholders = self.preprocess_image(image)

            return image_placeholders

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_ovis,
            )
        ]


class SiglipVisualTokenizer(nn.Module):

    def __init__(self, config: OvisConfig):
        super().__init__()
        self.config = config

        self.visual_tokenizer_config = config.visual_tokenizer_config
        self.backbone_config: SiglipVisionConfig = (
            self.visual_tokenizer_config.vision_config)

        self.hidden_stride = self.visual_tokenizer_config.hidden_stride
        self.hidden_size = self.backbone_config.hidden_size

        self.backbone = SiglipVisionModel(self.backbone_config)
        head_dim = self.visual_tokenizer_config.vocab_size - len(
            IMAGE_INDICATOR_IDS)  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * self.hidden_stride**2,
                            head_dim,
                            bias=False),
            torch.nn.LayerNorm(head_dim),
        )

    def tokenize(self, logits):

        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                y_soft, memory_format=torch.legacy_contiguous_format).scatter_(
                    dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == "softmax":
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == "gumbel_argmax":
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == "st_argmax":
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f"Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}"
            )
        return tokens

    def encode(self, pixel_values):
        output = self.backbone(pixel_values, interpolate_pos_encoding=True)
        features = output.hidden_states[-1]
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=3, this leads to a token length reduction: 729 -> 81 for siglip
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l**0.5)
            assert (sqrt_l**2 == l
                    ), "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride -
                  (sqrt_l %
                   self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(
                n,
                sqrt_l // self.config.hidden_stride,
                self.config.hidden_stride,
                sqrt_l // self.config.hidden_stride,
                self.config.hidden_stride,
                d,
            )
            features = features.permute(
                0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(
                3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1,
                self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(
        self, pixel_values
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(
            size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
            dtype=tokens.dtype,
            device=tokens.device,
            layout=tokens.layout,
            requires_grad=False,
        )
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens


class VisualEmbedding(torch.nn.Embedding):

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.long,
        ]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)


@MULTIMODAL_REGISTRY.register_processor(
    OvisMultiModalProcessor,
    info=OvisProcessingInfo,
    dummy_inputs=OvisDummyInputsBuilder,
)
class Ovis(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "llm.": "language_model.",
    })

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.text_tokenizer = cached_tokenizer_from_config(
            vllm_config.model_config)
        self.visual_tokenizer = SiglipVisualTokenizer(self.config)
        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler
        return get_sampler()

    def _validate_pixel_values(
        self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = w = self.config.visual_tokenizer.backbone_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(p: torch.Tensor):
            actual_dims = tuple(p.shape[1:])
            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(p.shape)}.")

        for p in pixel_values:
            _validate_shape(p)

        return pixel_values

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[OvisImageInputs]:
        pixel_values = kwargs.get("pixel_values")
        image_embeds = kwargs.get("image_embeds")

        if pixel_values is not None and image_embeds is not None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (list, torch.Tensor)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return OvisImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (list, torch.Tensor)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(image_embeds)}")

            return OvisImageEmbeddingInputs(type="image_embeds",
                                            data=flatten_bn(image_embeds,
                                                            concat=True))

    def _process_image_pixels(self,
                              pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual_tokenizer(pixel_values)

    def _process_image_input(self, image_input: OvisImagePixelInputs):
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.visual_tokenizer is not None
        image_tokens = self._process_image_pixels(image_input["data"])
        return self.vte(image_tokens)

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            return self._process_image_pixels(image_input)
        return None

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        input_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            input_embeds = merge_multimodal_embeddings(input_ids, input_embeds,
                                                       multimodal_embeddings,
                                                       IMAGE_INDICATOR_IDS)
        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):

        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

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
        return self.llm.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
