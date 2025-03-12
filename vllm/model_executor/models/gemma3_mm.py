# SPDX-License-Identifier: Apache-2.0
import math
from typing import (Any, Iterable, Literal, Mapping, Optional, Sequence, Set,
                    Tuple, TypedDict, Union)

import torch
from torch import nn
from transformers import BatchFeature, Gemma3Config, Gemma3Processor
from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, encode_tokens)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

logger = init_logger(__name__)


class Gemma3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""
    num_crops: torch.Tensor
    """Shape: `(batch_size * num_images, num_crops)`"""


Gemma3ImageInputs = Gemma3ImagePixelInputs


class Gemma3ProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def _resolve_image_kwargs(
        self,
        processor: Gemma3Processor,
        keys: set[str],
    ) -> dict[str, Any]:
        image_processor = processor.image_processor
        kwargs = processor._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        )

        images_kwargs = kwargs["images_kwargs"]

        def _resolve_kw(key: str):
            val = getattr(image_processor, key)
            if val is None:
                val = images_kwargs[key]

            return val

        return {k: _resolve_kw(k) for k in keys}

    def get_num_crops(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor, {
                "do_pan_and_scan", "pan_and_scan_min_crop_size",
                "pan_and_scan_max_num_crops",
                "pan_and_scan_min_ratio_to_activate"
            })

        do_pan_and_scan = images_kwargs["do_pan_and_scan"]
        pan_and_scan_min_crop_size = images_kwargs[
            "pan_and_scan_min_crop_size"]
        pan_and_scan_max_num_crops = images_kwargs[
            "pan_and_scan_max_num_crops"]
        pan_and_scan_min_ratio_to_activate = images_kwargs[
            "pan_and_scan_min_ratio_to_activate"]

        if not do_pan_and_scan:
            return 0

        # Based on Gemma3ImageProcessor.pan_and_scan
        if image_width >= image_height:
            if image_width / image_height < pan_and_scan_min_ratio_to_activate:
                return 0

            num_crops_w = min(
                int(math.floor(image_width / pan_and_scan_min_crop_size)),
                int(math.floor(image_width / image_height + 0.5)),
            )

            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1
        else:
            if image_height / image_width < pan_and_scan_min_ratio_to_activate:
                return 0

            num_crops_h = min(
                int(math.floor(image_height / pan_and_scan_min_crop_size)),
                int(math.floor(image_height / image_width + 0.5)),
            )

            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(image_width / num_crops_w))
        crop_size_h = int(math.ceil(image_height / num_crops_h))

        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return 0

        return num_crops_w * num_crops_h

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> str:
        if processor is None:
            processor = self.get_hf_processor()

        image_token = processor.boi_token

        num_crops = self.get_num_crops(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        if num_crops == 0:
            image_text = image_token
        else:
            crops_image_tokens = " ".join(image_token
                                          for _ in range(num_crops))
            image_text = (
                f"Here is the original image {image_token} and here are some "
                f"crops to help you see better {crops_image_tokens}")

        return image_text.replace(image_token, processor.full_image_sequence)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> int:
        tokenizer = self.get_tokenizer()
        image_repl = self.get_image_repl(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        image_repl_tokens = encode_tokens(
            tokenizer,
            image_repl,
            add_special_tokens=False,
        )
        return len(image_repl_tokens)

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor, {"pan_and_scan_max_num_crops"})
        max_num_crops = images_kwargs["pan_and_scan_max_num_crops"]

        # Result in the max possible feature size (h:w = max_num_crops:1)
        return ImageSize(height=50 * max_num_crops, width=50)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=None,
        )


class Gemma3DummyInputsBuilder(BaseDummyInputsBuilder[Gemma3ProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        processor = self.info.get_hf_processor()
        image_token = processor.boi_token

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
            prompt_text=" ".join(image_token for _ in range(num_images)),
            mm_data=mm_data,
        )


class Gemma3MultiModalProcessor(BaseMultiModalProcessor[Gemma3ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
        )

        # HF processor pops the `num_crops` kwarg, which is needed by vLLM
        if (images := mm_data.get("images")) is not None:
            assert isinstance(images, list)

            parsed_images = (self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", ImageProcessorItems))
            image_sizes = [
                parsed_images.get_image_size(i)
                for i in range(len(parsed_images))
            ]
            hf_processor = self.info.get_hf_processor(**mm_kwargs)

            num_crops = [
                self.info.get_num_crops(image_width=size.width,
                                        image_height=size.height,
                                        processor=hf_processor)
                for size in image_sizes
            ]

            processed_outputs["num_crops"] = torch.tensor(num_crops)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_crops = hf_inputs.get("num_crops", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_crops + 1),
            num_crops=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.boi_token

        def get_replacement_gemma3(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)
            return self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_gemma3,
            )
        ]


class Gemma3MultiModalProjector(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size,
                        config.text_config.hidden_size))

        self.mm_soft_emb_norm = GemmaRMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps)

        self.patches_per_image = int(config.vision_config.image_size //
                                     config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size,
                                     stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image,
            self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight)
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
        self.sliding_window = config.text_config.interleaved_sliding_window

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Gemma3ForCausalLM"],
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
        num_crops = kwargs.pop("num_crops", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma3 does not support image_embeds."
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(num_crops, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_crops values. "
                             f"Got type: {type(num_crops)}")

        pixel_values = flatten_bn(pixel_values, concat=True)
        num_crops = flatten_bn(num_crops, concat=True)

        return Gemma3ImagePixelInputs(
            type="pixel_values",
            pixel_values=self._validate_pixel_values(pixel_values),
            num_crops=num_crops,
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

        pixel_values = image_input["pixel_values"]
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
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
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
            if vision_embeddings is not None:
                kwargs = self.prepare_attn_masks(
                    input_ids,
                    positions,
                    mask_dtype=vision_embeddings.dtype,
                    **kwargs)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds,
                                                  **kwargs)

        return hidden_states

    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs,
    ):
        kwargs["has_images"] = True
        # NOTE(woosuk): Here, we distinguish the sequences by the position id 0.
        # This is a HACK. Fix this.
        start_idices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_idices)
        seq_lens = []
        for i in range(num_seqs):
            start_idx = start_idices[i].item()
            if i < num_seqs - 1:
                end_idx = start_idices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)
        kwargs["seq_lens"] = seq_lens

        global_attn_masks = []
        local_attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx
            # Create a global causal mask.
            global_attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            global_attn_mask.fill_(float("-inf"))
            # Fill the lower triangle with 0.
            global_attn_mask = global_attn_mask.triu(diagonal=1)

            # Consider the bidirectional attention between image tokens.
            img_mask = torch.zeros_like(global_attn_mask)
            img_pos = (input_token_ids == self.config.image_token_index)
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            global_attn_mask = torch.where(img_mask == 2, 0, global_attn_mask)
            global_attn_masks.append(global_attn_mask)

            # Create a local causal mask with sliding window (1024).
            local_attn_mask = torch.ones_like(global_attn_mask)
            local_attn_mask = torch.tril(local_attn_mask,
                                         diagonal=-self.sliding_window)
            local_attn_mask = torch.where(local_attn_mask == 0,
                                          global_attn_mask, float("-inf"))
            local_attn_masks.append(local_attn_mask)
        kwargs["global_attn_masks"] = global_attn_masks
        kwargs["local_attn_masks"] = local_attn_masks
        return kwargs

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
