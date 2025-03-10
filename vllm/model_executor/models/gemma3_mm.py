# SPDX-License-Identifier: Apache-2.0
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import (Any, Iterable, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
from torch import nn
# FIXME
# from transformers import Gemma3Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers import BatchFeature, ProcessorMixin

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings, flatten_bn)

logger = init_logger(__name__)

NUM_TOKENS_PER_IMAGE = 256


class Gemma3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


Gemma3ImageInputs = Gemma3ImagePixelInputs


class Gemma3ProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": NUM_TOKENS_PER_IMAGE}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin],
    ) -> int:
        return NUM_TOKENS_PER_IMAGE

    def get_image_size_with_most_features(self) -> ImageSize:
        # Result in the max possible feature size (h:w = 16:1)
        return ImageSize(height=8000, width=50)


class Gemma3DummyInputsBuilder(BaseDummyInputsBuilder[Gemma3ProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }
        return ProcessorInputs(
            prompt_text=" ".join(["<image>"] * num_images),
            mm_data=mm_data,
        )


class Gemma3MultiModalProcessor(BaseMultiModalProcessor[Gemma3ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # FIXME(woosuk): Currently, PaS is not supported.
        img_kwargs = mm_kwargs.get("images_kwargs", {})
        if img_kwargs:
            img_kwargs["do_pan_and_scan"] = False
        else:
            img_kwargs = {"do_pan_and_scan": False}
        mm_kwargs["images_kwargs"] = img_kwargs
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.tokenizer.image_token
        image_tokens_expanded = "".join([image_token] * NUM_TOKENS_PER_IMAGE)

        def get_replacement_gemma3(item_idx: int):
            return PromptUpdateDetails(
                full=hf_processor.full_image_sequence,
                features=image_tokens_expanded,
            )

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_gemma3,
            )
        ]


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = GemmaRMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        b, _, l = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(b, l, self.patches_per_image, self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.einsum("btm,md->btd", normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


@MULTIMODAL_REGISTRY.register_processor(Gemma3MultiModalProcessor,
                                        info=Gemma3ProcessingInfo,
                                        dummy_inputs=Gemma3DummyInputsBuilder)
class Gemma3ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                     SupportsPP):
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
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        config.text_config.architectures = ["Gemma3ForCausalLM"]
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.language_model.logits_processor.scale *= logit_scale

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @property
    def sampler(self):
        return self.language_model.sampler

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            if d.shape != expected_dims:
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_dims}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Gemma3ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                                f"Got type: {type(pixel_values)}")

        pixel_values = flatten_bn(pixel_values)
        return Gemma3ImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
        )

    def _image_pixels_to_features(
        self,
        vision_tower: SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = vision_tower.get_input_embeddings().weight.dtype
        image_features = vision_tower(pixel_values.to(dtype=target_dtype))
        return image_features

    def _process_image_input(
        self,
        image_input: Gemma3ImageInputs,
    ) -> torch.Tensor:
        assert self.vision_tower is not None
        pixel_values = image_input["data"]
        vision_outputs = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )
        return self.multi_modal_projector(vision_outputs)

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
        if multimodal_embeddings is None:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        else:
            # NOTE(woosuk): Gemma3 uses vocab_size as the image token index.
            # To avoid out-of-range error in the embedding layer, we replace the
            # image token index with 0.
            safe_input_ids = torch.where(
                input_ids == self.config.image_token_index, 0, input_ids)
            inputs_embeds = self.language_model.get_input_embeddings(safe_input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_index)
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object) -> Union[SamplerOutput, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
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
