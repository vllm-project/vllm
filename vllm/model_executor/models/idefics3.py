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
from typing import (Dict, Iterable, List, Literal, Mapping, NamedTuple,
                    Optional, Set, Tuple, TypedDict, Union)

import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
# Temporary solution for transformers below 4.46.0.
from transformers import PretrainedConfig as Idefics3Config
from transformers import ProcessorMixin as Idefics3ImageProcessor

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.image import cached_get_image_processor
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import is_list_of

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
    Shape: `(batch_size * num_images, num_channels, height, width)`
    """
    pixel_attention_mask: Optional[torch.BoolTensor]


class Idefics3ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
    `hidden_size` must match the hidden size of language model backbone.
    """


class Idefics3ProcessorSize(NamedTuple):
    """Hashable wrapper for unhashable `size` dict of Idefics3Processor."""
    # NOTE: cached_get_processor/cached_get_image_processor uses lru_cache,
    # we need to use NamedTuple instead of TypedDict to avoid hashing issues.
    longest_edge: int

    def __contains__(self, key: str) -> bool:
        return key in self._asdict() and getattr(self, key) is not None

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


ImageInputs = Union[Idefics3ImagePixelInputs, Idefics3ImageEmbeddingInputs]


def get_mm_processor_kwargs(size: Optional[Dict[str, int]] = None) -> Dict:
    mm_processor_kwargs = {}
    if size:
        mm_processor_kwargs["size"] = Idefics3ProcessorSize(**size)
    return mm_processor_kwargs


def input_mapper_for_idefics3(
    ctx: InputContext,
    data: object,
    *,
    size: Optional[Dict[str, int]] = None,
):
    model_config = ctx.model_config
    mm_processor_kwargs = get_mm_processor_kwargs(size)
    image_processor = cached_get_image_processor(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        **mm_processor_kwargs)
    if image_processor is None:
        raise RuntimeError("No HuggingFace processor is available "
                           "to process the image object")

    if isinstance(data, Image.Image):
        images = [[data]]
    elif is_list_of(data, Image.Image):
        images = [data]
    else:
        raise TypeError(f"Invalid image type: {type(data)}")

    try:
        batch_data = image_processor(images,
                                     return_tensors="pt",
                                     return_row_col_info=True).data
    except Exception:
        logger.error("Failed to process image (%s)", data)
        raise

    return MultiModalKwargs(batch_data)


def _resize_output_size(height: int,
                        width: int,
                        max_len: Optional[int] = None,
                        min_len: Optional[int] = 1,
                        max_size: Optional[int] = None) -> Tuple[int, int]:
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
    height += 1 if height % 2 != 0 else 0
    width += 1 if width % 2 != 0 else 0

    # Ensure dimensions are not smaller than the minimum length
    height = max(height, min_len)
    width = max(width, min_len)

    return height, width


def _get_resize_output_image_size(
    image_size: Tuple[int, int],
    resolution_max_side: int,
    max_image_size: int = 1820,
) -> Tuple[int, int]:
    if resolution_max_side > max_image_size:
        raise ValueError(
            "`resolution_max_side` cannot be larger than `max_image_size`")

    height, width = image_size

    # Find the output size, when rescaling the longest edge to max_len and
    # preserving the aspect ratio
    height, width = _resize_output_size(height,
                                        width,
                                        max_len=resolution_max_side)

    return height, width


def _prompt_split_image(image_seq_len: int, image_rows: int, image_cols: int,
                        fake_token_around_image: str, image_token: str,
                        global_img_token: str) -> str:
    """
    Prompt with expanded image tokens for when the image is split 
    into patches.
    """
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (fake_token_around_image +
                                  f"<row_{n_h + 1}_col_{n_w + 1}>" +
                                  image_token * image_seq_len)
        text_split_images += "\n"

    text_split_images += "\n" + _prompt_single_image(
        image_seq_len=image_seq_len,
        fake_token_around_image=fake_token_around_image,
        image_token=image_token,
        global_img_token=global_img_token)
    return text_split_images


def _prompt_single_image(image_seq_len: int, fake_token_around_image: str,
                         image_token: str, global_img_token: str):
    """Prompt with expanded image tokens for a single image."""
    return (fake_token_around_image + global_img_token +
            image_token * image_seq_len + fake_token_around_image)


def _get_image_prompt_string(image_rows: int, image_cols: int,
                             image_seq_len: int, fake_token_around_image: str,
                             image_token: str, global_img_token: str):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len=image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(image_seq_len, image_rows, image_cols,
                               fake_token_around_image, image_token,
                               global_img_token)


def input_processor_for_idefics3(ctx: InputContext,
                                 inputs: DecoderOnlyInputs,
                                 *,
                                 size: Optional[Dict[str, int]] = None):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    model_config = ctx.model_config
    mm_processor_kwargs = get_mm_processor_kwargs(size)
    processor = cached_get_processor(model_config.model, **mm_processor_kwargs)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    size = image_processor.size['longest_edge']
    max_image_size = image_processor.max_image_size['longest_edge']

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        image_list = [image_data]
    elif is_list_of(image_data, Image.Image):
        image_list = image_data
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    image_rows = []
    image_cols = []
    for image in image_list:
        height, width = _get_resize_output_image_size(image.size, size)

        rows = math.ceil(height / max_image_size)
        cols = math.ceil(width / max_image_size)
        image_rows.append(rows)
        image_cols.append(cols)
    image_rows = [image_rows]
    image_cols = [image_cols]

    n_images_in_text = []

    text = inputs.get("prompt")
    if text is not None:
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, "
                             "or a list of strings")

        fake_image_token = processor.fake_image_token.content
        image_token = processor.image_token.content
        global_img_token = processor.global_image_tag

        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows,
                                                    image_cols):
            n_images_in_text.append(sample.count(image_token))

            # Replace the image token with fake tokens around the expanded
            # image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = _get_image_prompt_string(
                    n_rows,
                    n_cols,
                    processor.image_seq_len,
                    image_token=image_token,
                    fake_token_around_image=fake_image_token,
                    global_img_token=global_img_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(image_token)
            if len(split_sample) == 0:
                raise ValueError(
                    "The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        prompt_token_ids = tokenizer(text=prompt_strings[0]).input_ids

        return token_inputs(
            prompt_token_ids=prompt_token_ids,
            prompt=prompt_strings[0],
            multi_modal_data=multi_modal_data,
        )


def _get_max_num_image_patch(image_processor: Idefics3ImageProcessor) -> int:
    size = image_processor.size['longest_edge']
    max_image_size = image_processor.max_image_size['longest_edge']
    resized_height, resized_width = size, size

    grid_h = resized_height // max_image_size
    grid_w = resized_width // max_image_size
    return (grid_h * grid_w + 1)


def get_max_idefics3_image_tokens(ctx: InputContext,
                                  *,
                                  size: Optional[Dict[str,
                                                      int]] = None) -> int:
    model_config = ctx.model_config
    mm_processor_kwargs = get_mm_processor_kwargs(size)
    processor = cached_get_processor(model_config.model, **mm_processor_kwargs)
    image_seq_len = processor.image_seq_len
    image_processor = processor.image_processor

    max_num_image_patches = _get_max_num_image_patch(image_processor)

    return max_num_image_patches * image_seq_len


def dummy_data_for_idefics3(
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
        *,
        size: Optional[Dict[str, int]] = None) -> DummyData:
    hf_config = ctx.get_hf_config()
    num_images = mm_counts["image"]

    mm_processor_kwargs = get_mm_processor_kwargs(size)
    processor = cached_get_processor(ctx.model_config.model,
                                     **mm_processor_kwargs)
    max_num_image_patches = _get_max_num_image_patch(processor.image_processor)
    image_seq_len = processor.image_seq_len
    max_llm_image_tokens = max_num_image_patches * image_seq_len * num_images

    if seq_len - max_llm_image_tokens < 0:
        raise RuntimeError(
            f"Idefics3 cannot process {num_images} images in a prompt, "
            "please increase max_model_len or reduce image limit by "
            "--limit-mm-per-prompt.")

    seq_data = SequenceData.from_prompt_token_counts(
        (hf_config.image_token_id, max_llm_image_tokens),
        (0, seq_len - max_llm_image_tokens))

    width = height = hf_config.vision_config.image_size
    image = Image.new("RGB", (width, height), color=0)
    mm_data = {"image": [image] if num_images == 1 else [image] * num_images}

    return DummyData(seq_data, mm_data)


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

        config = vllm_config.model_config.hf_config
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

            return Idefics3ImagePixelInputs(type="pixel_values",
                                            data=self._validate_pixel_values(
                                                flatten_bn(pixel_values,
                                                           concat=True)),
                                            pixel_attention_mask=flatten_bn(
                                                pixel_attention_mask,
                                                concat=True))

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images,
                                         *pixel_values.shape[2:])

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
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:])
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

        return image_hidden_states

    def _process_image_pixels(
            self, inputs: Idefics3ImagePixelInputs) -> torch.Tensor:
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
        return self.connector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            image_input = self._parse_and_validate_image_input(**kwargs)

            if image_input is not None:
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.text_model.get_input_embeddings(input_ids)

                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.image_token_id)
            else:
                inputs_embeds = self.text_model.get_input_embeddings(input_ids)
            input_ids = None

        hidden_states = self.text_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_idefics3)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_idefics3_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_idefics3)
@INPUT_REGISTRY.register_input_processor(input_processor_for_idefics3)
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
    # LoRA specific attributes
    supported_lora_modules = [
        # vision_model
        "fc1",
        "fc2",
        "out_proj",
        # text_model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        # vision_model
        ".fc1.",
        ".fc2.",
        ".out_proj.",
        # connector
        ".proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

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
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            **kwargs,
        )
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
