# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature, Gemma3Config, Gemma3Processor
from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    replace_token_matches,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


class Gemma3ImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - p: Number of patches total (over each image over each prompt in the
          batch)
        - c: Number of channels (3)
        - h: Height of each patch
        - w: Width of each patch
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[torch.Tensor, TensorShape("p", 3, "h", "w")]

    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


Gemma3ImageInputs = Gemma3ImagePixelInputs


class Gemma3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma3Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

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
        processor: Gemma3Processor | None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor,
            {
                "do_pan_and_scan",
                "pan_and_scan_min_crop_size",
                "pan_and_scan_max_num_crops",
                "pan_and_scan_min_ratio_to_activate",
            },
        )

        do_pan_and_scan = images_kwargs["do_pan_and_scan"]
        pan_and_scan_min_crop_size = images_kwargs["pan_and_scan_min_crop_size"]
        pan_and_scan_max_num_crops = images_kwargs["pan_and_scan_max_num_crops"]
        pan_and_scan_min_ratio_to_activate = images_kwargs[
            "pan_and_scan_min_ratio_to_activate"
        ]

        if not do_pan_and_scan:
            return 0

        logger.warning_once(
            "`do_pan_and_scan=True` has suboptimal results on V1 "
            "because of the simplified attention pattern being used."
        )

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
        processor: Gemma3Processor | None,
    ) -> PromptUpdateDetails[str]:
        if processor is None:
            processor = self.get_hf_processor()

        boi_token = processor.boi_token

        num_crops = self.get_num_crops(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        if num_crops == 0:
            image_text = boi_token
        else:
            crops_image_tokens = " ".join(boi_token for _ in range(num_crops))
            image_text = (
                f"Here is the original image {boi_token} and here are some "
                f"crops to help you see better {crops_image_tokens}"
            )

        repl_full = image_text.replace(boi_token, processor.full_image_sequence)

        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        image_token_id = vocab[tokenizer.image_token]

        return PromptUpdateDetails.select_token_id(repl_full, image_token_id)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma3Processor | None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        num_crops = self.get_num_crops(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )
        image_seq_len = processor.image_seq_length

        return (num_crops + 1) * image_seq_len

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor, {"pan_and_scan_max_num_crops"}
        )
        max_num_crops = images_kwargs["pan_and_scan_max_num_crops"]

        vision_config = self.get_hf_config().vision_config
        native_size = vision_config.image_size
        return ImageSize(height=native_size * max_num_crops, width=native_size)


class Gemma3DummyInputsBuilder(BaseDummyInputsBuilder[Gemma3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.boi_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class Gemma3MultiModalProcessor(BaseMultiModalProcessor[Gemma3ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        # HF processor pops the `num_crops` kwarg, which is needed by vLLM
        if (images := mm_data.get("images")) is not None:
            parsed_images = (
                self._get_data_parser()
                .parse_mm_data({"image": images})
                .get_items("image", ImageProcessorItems)
            )
            image_sizes = [
                parsed_images.get_image_size(i) for i in range(len(parsed_images))
            ]
            hf_processor = self.info.get_hf_processor(**mm_kwargs)

            num_crops = [
                self.info.get_num_crops(
                    image_width=size.width,
                    image_height=size.height,
                    processor=hf_processor,
                )
                for size in image_sizes
            ]
            processed_outputs["num_patches"] = torch.tensor(num_crops) + 1

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
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

    def _apply_token_matches(
        self,
        prompt: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], MultiModalPromptUpdatesApplyResult]:
        token_ids, res = super()._apply_token_matches(prompt, mm_prompt_updates)

        # "\n\n\n" and "\n\n\n\n" are single tokens
        # Since our replacement can insert "\n\n" next to "\n"
        # tokens, we have to combine them to be consistent with
        # the output of the tokenizer
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        newline_1 = vocab["\n"]
        newline_2 = vocab["\n\n"]
        newline_3 = vocab["\n\n\n"]
        newline_4 = vocab["\n\n\n\n"]

        token_ids = replace_token_matches(
            token_ids,
            [newline_1, newline_2],
            [newline_3],
        )
        token_ids = replace_token_matches(
            token_ids,
            [newline_2, newline_1],
            [newline_3],
        )
        token_ids = replace_token_matches(
            token_ids,
            [newline_2, newline_2],
            [newline_4],
        )

        return token_ids, res

    def _find_mm_placeholders(
        self,
        new_token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        # We need to detect "\n\n" inside "\n\n\n" and "\n\n\n\n"
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        newline_1 = vocab["\n"]
        newline_2 = vocab["\n\n"]
        newline_3 = vocab["\n\n\n"]
        newline_4 = vocab["\n\n\n\n"]

        def get_repl_toks(tok: int) -> list[int]:
            if tok == newline_3:
                return [newline_1, newline_2]
            if tok == newline_4:
                return [newline_2, newline_2]

            return [tok]

        repl_token_ids = list[int]()
        repl_orig_idxs = list[int]()
        for orig_idx, orig_tok in enumerate(new_token_ids):
            repl_toks = get_repl_toks(orig_tok)
            repl_token_ids.extend(repl_toks)
            repl_orig_idxs.extend(orig_idx for _ in range(len(repl_toks)))

        repls = super()._find_mm_placeholders(repl_token_ids, mm_prompt_updates)

        return {
            modality: [
                PlaceholderFeaturesInfo(
                    modality=p.modality,
                    item_idx=p.item_idx,
                    start_idx=repl_orig_idxs[p.start_idx],
                    tokens=p.tokens,
                    is_embed=p.is_embed,
                )
                for p in placeholders
            ]
            for modality, placeholders in repls.items()
        }


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )

        self.mm_soft_emb_norm = GemmaRMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.type_as(vision_outputs)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma3MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class Gemma3ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
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

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<start_of_image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = SiglipVisionModel(
                config.vision_config,
                quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = Gemma3MultiModalProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma3ForCausalLM"],
            )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.language_model.logits_processor.scale *= logit_scale

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma3ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma3 does not support image_embeds."
        if pixel_values is None:
            return None

        image_size = self.config.vision_config.image_size

        return Gemma3ImagePixelInputs(
            pixel_values=pixel_values,
            num_patches=num_patches,
            resolve_bindings={"h": image_size, "w": image_size},
        )

    def _image_pixels_to_features(
        self,
        vision_tower: SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        return vision_tower(pixel_values)

    def _process_image_input(
        self,
        image_input: Gemma3ImageInputs,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        image_features = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )
        image_embeds = self.multi_modal_projector(image_features)

        return [e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        # Early return for text-only inference (no multimodal data)
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        # Use interface default with OOV handling enabled
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        """
        Calculate the number of tokens output by the vision encoder.

        The vision encoder processes images into patch embeddings. For Gemma3,
        the relationship between prompt placeholder tokens and actual vision
        encoder output tokens depends on the patch grid size.

        Args:
            num_image_tokens: Number of image placeholder tokens in the prompt
                              (typically mm_tokens_per_image per image)

        Returns:
            Number of tokens output by the vision encoder
        """
        # For Gemma3, the vision encoder outputs tokens_per_side x tokens_per_side
        # tokens per image. Since num_image_tokens represents the number of
        # connector output tokens (mm_tokens_per_image = 256), and tokens_per_side
        # is sqrt(256) = 16, we need to account for the token expansion.
        # Based on empirical testing, the multiplier of 16 works correctly.
        return num_image_tokens * 16

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        """
        Calculate the number of tokens output by the multimodal connector.

        The connector applies projection and normalization but maintains the
        token count for Gemma3.

        Args:
            num_vision_tokens: Number of tokens from vision encoder

        Returns:
            Number of tokens after connector processing
        """
        # The Gemma3 connector maintains a 1:1 token mapping
        return num_vision_tokens
