# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, BatchFeature
from transformers.models.gemma3n import (
    Gemma3nAudioConfig,
    Gemma3nAudioFeatureExtractor,
    Gemma3nConfig,
    Gemma3nProcessor,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
)
from transformers.models.siglip import SiglipImageProcessorFast

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.gemma3n import Gemma3nForCausalLM
from vllm.model_executor.models.gemma3n_audio_utils import (
    adjust_audio_features_to_expected_length,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.vision import should_torch_compile_mm_vit
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
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

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)

# This should be based on model config but we hardcode them for now.
TOKENS_PER_IMAGE = 256
TOKENS_PER_AUDIO = 188


class Gemma3nImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each patch
        - w: Width of each patch
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]


class Gemma3nAudioInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - s: seq_length
        - f: num_features
    """

    type: Literal["audio"] = "audio"
    input_features_padded: Annotated[torch.Tensor, TensorShape("bn", "s", "f")]
    input_features_mask: Annotated[torch.Tensor, TensorShape("bn", "s")]


Gemma3nImageInputs = Gemma3nImagePixelInputs


class Gemma3nProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma3nConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma3nProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> Gemma3nAudioFeatureExtractor:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()

        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        return {"image": TOKENS_PER_IMAGE, "audio": TOKENS_PER_AUDIO}

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma3nProcessor | None,
    ) -> str:
        """
        Get the replacement text for image tokens.

        For Gemma3n, this should return the full_image_sequence which includes
        BOI token, repeated image tokens, and EOI token.
        """
        if processor is None:
            processor = self.get_hf_processor()

        return PromptUpdateDetails.select_token_id(
            processor.full_image_sequence, processor.image_token_id
        )

    def get_audio_repl(
        self,
        *,
        processor: Gemma3nProcessor | None,
    ) -> str:
        """
        Get the replacement text for audio tokens.

        For Gemma3n, this should return the full_audio_sequence which includes
        BOA token, repeated audio tokens, and EOA token.
        """
        if processor is None:
            processor = self.get_hf_processor()

        # Return the full audio sequence as defined by the processor
        return PromptUpdateDetails.select_token_id(
            processor.full_audio_sequence, processor.audio_token_id
        )


class Gemma3nDummyInputsBuilder(BaseDummyInputsBuilder[Gemma3nProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        audio_token = processor.audio_token

        return image_token * num_images + audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        processor = self.info.get_hf_processor()
        audio_feature_extractor: Gemma3nAudioFeatureExtractor = (
            processor.feature_extractor
        )
        audio_len = audio_feature_extractor.fft_length
        image_processor: SiglipImageProcessorFast = processor.image_processor
        img_width = image_processor.size.get("width", 224)
        img_height = image_processor.size.get("height", 224)

        image_overrides = mm_options.get("image") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=img_width,
                height=img_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            ),
        }


class Gemma3nMultiModalProcessor(BaseMultiModalProcessor[Gemma3nProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # HF Transformers audio processor no longer accepts `audios` key.
        # We pop `audios` and replace it with `audio` key to suppress
        # the warning.
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        if "input_features" in processed_outputs:
            # Padding enables audio_tower to run in batched mode
            processed_outputs["input_features_padded"] = processed_outputs[
                "input_features"
            ]

            # Unpad features here since we need the output of each item to be
            # independent of other items for the cache to work correctly
            unpadded_features = [
                f[mask]
                for f, mask in zip(
                    processed_outputs["input_features"],
                    processed_outputs["input_features_mask"],
                )
            ]
            processed_outputs["input_features"] = unpadded_features
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            input_features_padded=MultiModalFieldConfig.batched("audio"),
            input_features_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        prompt_updates = []

        # Handle image tokens
        if "image" in mm_items:
            image_token = hf_processor.image_token

            def get_replacement_image(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)
                image_size = images.get_image_size(item_idx)
                return self.info.get_image_repl(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=image_token,
                    replacement=get_replacement_image,
                )
            )

        # Handle audio tokens
        if "audio" in mm_items:
            audio_token = hf_processor.audio_token

            def get_replacement_audio(item_idx: int):
                return self.info.get_audio_repl(
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="audio",
                    target=audio_token,
                    replacement=get_replacement_audio,
                )
            )

        return prompt_updates

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


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "inputs_embeds": 0},
    mark_unbacked_dims={"input_ids": 0, "inputs_embeds": 0},
    enable_if=should_torch_compile_mm_vit,
)
class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language
    model space."""

    def __init__(
        self,
        multimodal_config: Gemma3nAudioConfig | Gemma3nVisionConfig,
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = torch.tensor(multimodal_config.vocab_offset)
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
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """  # noqa: E501
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


@MULTIMODAL_REGISTRY.register_processor(
    Gemma3nMultiModalProcessor,
    info=Gemma3nProcessingInfo,
    dummy_inputs=Gemma3nDummyInputsBuilder,
)
class Gemma3nForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

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
            "model.embed_audio.": "embed_audio.",
            "model.embed_vision.": "embed_vision.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.audio_tower.": "audio_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
            "model": "language_model.model",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.vocab_size = config.text_config.vocab_size

        with (
            self._mark_tower_model(vllm_config, "image"),
            set_model_tag("Gemma3nMultimodalEmbedder", is_encoder=True),
        ):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma3nMultimodalEmbedder(
                multimodal_config=config.vision_config, text_config=config.text_config
            )
            # self.embed_vision.compile(fullgraph=True)

        with (
            self._mark_tower_model(vllm_config, "audio"),
            set_model_tag("Gemma3nMultimodalEmbedder", is_encoder=True),
        ):
            self.audio_tower = AutoModel.from_config(config=config.audio_config)
            self.embed_audio = Gemma3nMultimodalEmbedder(
                multimodal_config=config.audio_config, text_config=config.text_config
            )

        with self._mark_language_model(vllm_config):
            self.language_model: Gemma3nForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma3nForCausalLM"],
            )

            # NOTE (NickLucche) In order to be compatible with cudagraph, the
            # buffer needs to be consistent, so we pre-allocate here.
            self.per_layer_embeddings = torch.zeros(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.config.text_config.num_hidden_layers,
                self.config.text_config.hidden_size_per_layer_input,
                device=self.language_model.model.embed_tokens.weight.device,
                dtype=self.language_model.model.embed_tokens.weight.dtype,
            )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma3nImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        # TODO is this the case?
        assert image_embeds is None, "Gemma3n does not support image_embeds."
        if pixel_values is None:
            return None

        return Gemma3nImagePixelInputs(pixel_values=pixel_values)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Gemma3nAudioInputs | None:
        input_features_padded = kwargs.pop("input_features_padded", None)
        if input_features_padded is None:
            return None

        input_features_mask = kwargs.pop("input_features_mask", None)
        if input_features_mask is None:
            return None

        return Gemma3nAudioInputs(
            input_features_padded=input_features_padded,
            input_features_mask=input_features_mask,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key == "input_features_padded"
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def _process_image_input(
        self,
        image_input: Gemma3nImageInputs,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # TODO try to avoid copy here
        # (batch, channels, height, width) to (batch, height * width, channels)
        vision_outputs = (
            vision_outputs.reshape(
                vision_outputs.shape[0],
                self.config.vision_config.hidden_size,
                self.config.vision_soft_tokens_per_image,
            )
            .permute(0, 2, 1)
            .contiguous()
        )
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        # Return a list of embeddings instead of a batched tensor
        return self.embed_vision(inputs_embeds=vision_outputs).unbind(0)

    def _process_audio_input(
        self,
        audio_input: Gemma3nAudioInputs,
    ) -> list[torch.Tensor]:
        # Run on padded features to enable batching
        input_features = audio_input["input_features_padded"].squeeze(1)
        input_features_mask = audio_input["input_features_mask"].squeeze(1)
        audio_outputs, audio_mask = self.audio_tower(
            input_features, ~input_features_mask
        )
        audio_features = self.embed_audio(inputs_embeds=audio_outputs)

        # The Gemma3nProcessor expects all audio will be 30s in length and
        # inserts 188 audio soft tokens into the text to account for this.
        # However, the audio preprocessing and encoder do not guarantee they
        # will produce exactly 188 soft tokens; they may produce fewer tokens
        # (for shorter audio) or more tokens (for longer audio or due to
        # BOA/EOA special tokens in the placeholder sequence).
        # We handle both cases:
        # - If fewer tokens: pad with the embedding of the last vocab token
        # - If more tokens: truncate to the expected count
        # TODO precompute and cache padding
        audio_padding_toks = torch.tensor(
            [[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device
        )
        audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
        audio_features = torch.where(
            audio_mask.unsqueeze(-1), audio_padding_embs, audio_features
        )

        expected_tokens = self.config.audio_soft_tokens_per_image
        audio_features, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, audio_padding_embs
        )
        if tokens_truncated > 0:
            logger.warning(
                "Gemma3n audio encoder produced %d extra tokens. "
                "Truncating to match placeholder count of %d.",
                tokens_truncated,
                expected_tokens,
            )

        # Return a list of embeddings instead of a batched tensor
        return audio_features.unbind(0)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if mm_input_by_modality is None:
            return []

        multimodal_embeddings: list[torch.Tensor] = []

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            with set_forward_context(None, self.vllm_config):
                if modality == "image":
                    vision_embeddings = self._process_image_input(multimodal_input)
                    multimodal_embeddings.extend(vision_embeddings)
                if modality == "audio":
                    audio_embeddings = self._process_audio_input(multimodal_input)
                    multimodal_embeddings.extend(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # NOTE (NickLucche) Each pass needs tokens to compute PLE so we cache
        # them here, as the model  forward has only access to the input_embeds.
        if input_ids is not None:
            per_layer_inputs = self.language_model.model.get_per_layer_input_embeddings(
                input_ids
            )
            per_layer_inputs = per_layer_inputs.reshape(
                -1,
                self.config.text_config.num_hidden_layers,
                self.config.text_config.hidden_size_per_layer_input,
            )
            self.per_layer_embeddings[: per_layer_inputs.shape[0]].copy_(
                per_layer_inputs
            )

        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

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

        # NOTE (NickLucche) During profiling, `embed_input_ids` is not
        # called, hence we don't have input_ids to compute PLEs. We simply
        # select a chunk of pre-allocated PLEs. During normal execution,
        # `embed_input_ids` is called before forward, hence this slice
        # will contain PLEs computed from the actual input_ids.
        per_layer_inputs = self.per_layer_embeddings[: inputs_embeds.shape[0]]

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            per_layer_inputs=per_layer_inputs,
            intermediate_tensors=intermediate_tensors,
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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        elif modality == "audio":
            return "<audio_soft_token>"
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """
        Gemma3n supports "free-form" transcription.
        We fix its prompt here to standardize transcriptions/translations
        requests.
        """
        # Transcribe this audio [into <>] | for transcription
        # Translate this audio [from <> into <>] | for translation
        prompt = "<start_of_turn>user\n"
        prompt += "Transcribe" if task_type == "transcribe" else "Translate"
        prompt += " this audio"

        # We assume the language is a valid ISO 639-1 code.
        full_lang_name = cls.supported_languages.get(language, "")
        # Translation only for now
        full_lang_name_to = cls.supported_languages.get(to_language, "")

        if task_type == "transcribe" and full_lang_name:
            prompt += f" into {full_lang_name}"
        elif task_type == "translate":
            if full_lang_name:
                prompt += f" from {full_lang_name}"
            if full_lang_name_to:
                prompt += f" into {full_lang_name_to}"

        prompt += ": <audio_soft_token><end_of_turn>\n<start_of_turn>model\n"

        audio = (audio, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio}, "prompt": prompt}
        return cast(PromptType, prompts_dict)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            # Let's set this to 30 as suggested in the docs for now, although
            # the model is only limited by its context length.
            max_audio_clip_s=30,
            sample_rate=16000,
            # TODO enable chunking after more thorough testing.
            min_energy_split_window_size=None,
        )
