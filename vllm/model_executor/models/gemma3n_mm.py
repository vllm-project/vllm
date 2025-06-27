# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict

import torch
from torch import nn
from transformers import BatchFeature, Gemma3nConfig, Gemma3nProcessor
from transformers.models.gemma3n.processing_gemma3n import Gemma3nProcessorKwargs
from transformers import AutoModel

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
# yapf: disable
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, BoundPromptUpdate,
                                        PlaceholderFeaturesInfo,
                                        PromptReplacement, PromptTargetMatch,
                                        PromptUpdate, PromptUpdateDetails,
                                        find_mm_placeholders,
                                        replace_token_matches)
# yapf: enable
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)


class Gemma3nImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""

class Gemma3nAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""
    input_features_mask: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length)`"""


Gemma3nImageInputs = Gemma3nImagePixelInputs


class Gemma3ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma3nConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma3nProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
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

        if envs.VLLM_USE_V1:
            logger.warning_once(
                "`do_pan_and_scan=True` has suboptimal results on V1 "
                "because of the simplified attention pattern being used.")

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
                f"crops to help you see better {crops_image_tokens}")

        repl_full = image_text.replace(boi_token,
                                       processor.full_image_sequence)

        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        image_token_id = vocab[tokenizer.image_token]

        return PromptUpdateDetails.select_token_id(repl_full, image_token_id)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
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
            processor, {"pan_and_scan_max_num_crops"})
        max_num_crops = images_kwargs["pan_and_scan_max_num_crops"]

        # Result in the max possible feature size (h:w = max_num_crops:1)
        return ImageSize(height=50 * max_num_crops, width=50)


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

    def _apply_token_matches(
        self,
        prompt: list[int],
        mm_matches: Mapping[str, Sequence[PromptTargetMatch]],
        mm_item_counts: Mapping[str, int],
    ) -> list[int]:
        token_ids = super()._apply_token_matches(
            prompt,
            mm_matches,
            mm_item_counts,
        )

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

        return token_ids

    def _find_mm_placeholders(
        self,
        mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
        new_token_ids: list[int],
        mm_item_counts: Mapping[str, int],
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

        repls = find_mm_placeholders(mm_prompt_updates, repl_token_ids,
                                     mm_item_counts)

        return {
            modality: [
                PlaceholderFeaturesInfo(
                    modality=p.modality,
                    item_idx=p.item_idx,
                    start_idx=repl_orig_idxs[p.start_idx],
                    tokens=p.tokens,
                    is_embed=p.is_embed,
                ) for p in placeholders
            ]
            for modality, placeholders in repls.items()
        }


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size


        self.embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.multimodal_hidden_size,
        )

        self.hard_embedding_norm = RMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.soft_embedding_norm = RMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.embedding_projection = RowParallelLinear(
            self.multimodal_hidden_size,
            self.text_hidden_size,
            bias=False,
        )

        self.embedding_post_projection_norm = RMSNorm(
            self.text_hidden_size,
            eps=self.eps,
            has_weight=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj, _ = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


@MULTIMODAL_REGISTRY.register_processor(Gemma3MultiModalProcessor,
                                        info=Gemma3ProcessingInfo,
                                        dummy_inputs=Gemma3DummyInputsBuilder)
class Gemma3nForConditionalGeneration(nn.Module, SupportsMultiModal):
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
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.sliding_window = getattr(config.text_config,
                                      "interleaved_sliding_window", None)

        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.audio_tower = AutoModel.from_config(config=config.audio_config)
        self.embed_vision = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Gemma3nForCausalLM"],
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _process_image_input(
        self,
        image_input: Gemma3nImageInputs,
    ) -> list[torch.Tensor]:
        assert self.vision_tower is not None

        pixel_values = image_input["pixel_values"]
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    def _process_audio_input(
        self, audio_input: Gemma3nAudioInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.audio_tower is not None
        input_features = audio_input["input_features"]
        input_features_mask = audio_input["input_features_mask"]
        audio_outputs, audio_mask = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object) -> IntermediateTensors:
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
                    mask_dtype=self.dtype,
                    **kwargs,
                )
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds,
                                                  **kwargs)

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
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower")
