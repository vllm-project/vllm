# coding=utf-8
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

from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import Idefics3Config

from vllm.logger import init_logger
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors


from .llama import LlamaModel
from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics3VisionTransformer, )
from .interfaces import SupportsMultiModal
from .siglip import (
    SiglipVisionModel,
    dummy_image_for_siglip,
    dummy_seq_data_for_siglip,
    get_max_siglip_image_tokens,
    input_processor_for_siglip,
)
from .utils import (
    AutoWeightsLoader,
    flatten_bn,
    merge_multimodal_embeddings,
)

logger = init_logger(__name__)


class ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


ImageInputs = Union[ImagePixelInputs, ImageEmbeddingInputs]


def input_processor_for_idefics3(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    model_config = ctx.model_config
    version = get_version_by_config(model_config.hf_config)
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)
    image_processor = cached_get_image_processor(model_config.tokenizer)

    def get_placeholder(image_size: Tuple[int, int], num_image: int):
        if version == (2, 0) or version == (2, 5):
            return image_processor. \
                get_slice_image_placeholder(image_size)
        return image_processor. \
            get_slice_image_placeholder(image_size, num_image)

    prompt = llm_inputs.get("prompt")
    token_ids = llm_inputs.get("prompt_token_ids")
    if prompt is None:
        prompt = tokenizer.decode(token_ids)

    pattern = "(<image>./</image>)"
    images = multi_modal_data["image"]
    image_tags = re.findall(pattern, prompt)
    if len(image_tags) == 0:
        new_token_ids = token_ids
        new_prompt = prompt
    else:
        if isinstance(images, dict):
            image_size_list = images.get("image_size_list")
            images = [images.get("image_embeds")]
        else:
            if isinstance(images, Image.Image):
                images = [images]
            image_size_list = [image.size for image in images]

        text_chunks = prompt.split(pattern)
        new_prompt_chunks: List[str] = []
        for i in range(len(image_size_list)):
            new_prompt_chunks += [
                text_chunks[i],
                get_placeholder(image_size_list[i], i)
            ]
        new_prompt_chunks.append(text_chunks[-1])
        new_prompt = "".join(new_prompt_chunks)
        new_token_ids = tokenizer.encode(new_prompt)

    multi_modal_data["image"] = [
        _build_image_input(ctx, image) for image in images
    ]

    llm_inputs = LLMInputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
    )
    return llm_inputs


class Idefics3SimpleMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**
                                                         2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics3
class Idefics3RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics3Connector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
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

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states,
                                                 self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Idefics3Model(nn.Module):

    def __init__(
        self,
        config: Idefics3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = Idefics3VisionTransformer(config.vision_config, )
        self.connector = Idefics3Connector(config)
        self.text_model = LlamaModel(config.text_config, cache_config,
                                     quant_config)

        self.image_seq_len = int(
            ((config.vision_config.image_size //
              config.vision_config.patch_size)**2) / (config.scale_factor**2))
        self.image_token_id = self.config.image_token_id

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return ImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(self, inputs: ImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(self, image_input: ImageInputs) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        image_features = self._process_image_pixels(image_input)
        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        r"""
        TODO
        ```"""
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            image_input = self._parse_and_validate_image_input(**kwargs)

            if image_input is not None:
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)

                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.config.image_token_index)
            else:
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)
            input_ids = None

        hidden_states = self.text_model(input_ids,
                                        positions,
                                        kv_caches,
                                        attn_metadata,
                                        intermediate_tensors,
                                        inputs_embeds=inputs_embeds)

        return hidden_states


def dummy_data_for_idefics3(ctx: InputContext, seq_len: int,
                            mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(Idefics3Config)
    vision_config = hf_config.vision_config
    num_images = mm_counts["image"]

    image_feature_size = get_max_llava_image_tokens(ctx)

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            num_images,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_clip(vision_config, num_images)
        return seq_data, mm_data
    elif isinstance(vision_config, SiglipVisionConfig):
        seq_data = dummy_seq_data_for_siglip(
            vision_config,
            seq_len,
            num_images,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_siglip(vision_config, num_images)
        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_idefics3)
@INPUT_REGISTRY.register_input_processor(input_processor_for_idefics3)
class Idefics3ForConditionalGeneration(nn.Module, SupportsMultiModal):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: Idefics3Config,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        self.model = Idefics3Model(config, cache_config, quant_config)
        self.image_token_id = self.config.image_token_id

        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

        self.logits_processor = LogitsProcessor(self.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        r"""
        TODO
        ```"""

        outputs = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            **kwargs,
        )

        hidden_states = outputs[0]

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
