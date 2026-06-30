# SPDX-License-Identifier: Apache-2.0
"""
Multimodal Apertus Pipeline optimized for native vLLM asynchronous execution.

Architecture Contract:
1. Processor (CPU): Deterministic tensor creation, dynamic batch padding, and dummy token injection.
   Zero neural network inference is executed here.
2. Worker (GPU): Native Emu3.5/WavTokenizer execution and O(1) boolean mask ID substitution.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from transformers import ApertusConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
    )
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
    )
from vllm.logger import init_logger

from .apertus import ApertusForCausalLM
from .apertus_utils import ApertusAudioTokenizer, ApertusImageTokenizer
from .interfaces import MultiModalEmbeddings, SupportsMultiModal

logger = init_logger(__name__)


class ApertusProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> ApertusConfig:
        return self.ctx.get_hf_config(ApertusConfig)

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
                target_sr=ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE,
                target_channels=1,
                expected_hidden_size=self._get_expected_hidden_size(),
                )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
            self, seq_len: int, mm_counts: Mapping[str, int]
            ) -> Mapping[str, int] | None:
        del mm_counts
        ds = ApertusImageTokenizer.EMU35_DS_FACTOR
        max_px = ApertusImageTokenizer.DEFAULT_MAX_PIXELS

        return {
            "image": min((max_px // (ds * ds)) + 512, seq_len),
            "audio": min((40 * 300) + 4, seq_len),  # 40 tokens/sec * 300 secs + formatting
            }


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (
                ApertusImageTokenizer.DEFAULT_IMAGE_PLACEHOLDER * mm_counts.get("image", 0) +
                ApertusAudioTokenizer.DEFAULT_AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)
        )

    def get_dummy_mm_data(
            self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions]
            ) -> MultiModalDataDict:
        max_side = int(ApertusImageTokenizer.DEFAULT_MAX_PIXELS ** 0.5)
        audio_length = ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE

        return {
            "image": self._get_dummy_images(
                    width=max_side, height=max_side, num_images=mm_counts.get("image", 0)
                    ),
            "audio": self._get_dummy_audios(
                    length=audio_length, num_audios=mm_counts.get("audio", 0)
                    ),
            }


class ApertusMultiModalProcessor(BaseMultiModalProcessor[ApertusProcessingInfo]):
    """CPU-bound API Processor. Strict YAGNI Rule: NO heavy neural networks run here."""

    def __init__(
            self, info: ApertusProcessingInfo, dummy_inputs: BaseDummyInputsBuilder, *, cache: object | None = None
            ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        self.image_tokenizer = ApertusImageTokenizer()
        self.audio_tokenizer = ApertusAudioTokenizer()

    def _get_mm_fields_config(
            self, hf_inputs: object, hf_processor_mm_kwargs: Mapping[str, object]
            ) -> Mapping[str, MultiModalFieldConfig]:
        """Routes batched tensors and metadata directly to the GPU Worker's embed_input_ids kwargs."""
        return {
            "pixel_values": MultiModalFieldConfig.batched("pixel_values"),
            "image_sizes": MultiModalFieldConfig.batched("image_sizes"),
            "audio_values": MultiModalFieldConfig.batched("audio_values"),
            "audio_lengths": MultiModalFieldConfig.batched("audio_lengths"),
            }

    def _get_prompt_updates(self, *args: Any, **kwargs: Any) -> Sequence[PromptUpdate]:
        return []

    def apply(self, inputs: ProcessorInputs, timing_ctx: TimingContext) -> MultiModalInput:
        tokenizer = self.info.get_tokenizer()
        prompt_text = inputs.prompt if isinstance(inputs.prompt, str) else tokenizer.decode(inputs.prompt)
        config = self.info.get_hf_config()

        # Must be defined as distinct single tokens in tokenizer.json
        dummy_img_token = getattr(config, "dummy_image_token", "<|dummy_image|>")
        dummy_aud_token = getattr(config, "dummy_audio_token", "<|dummy_audio|>")

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)

        mm_kwargs: dict[str, torch.Tensor] = {}
        prompt_replacements: list[PromptReplacement] = []
        mm_counts: dict[str, int] = {}

        # 1. Vision Preprocessing (Mathematical Layout & Padding Only)
        if num_images > 0:
            with timing_ctx.record("preprocess_apertus_images"):
                images = inputs.mm_data_items.get_items("image", ImageProcessorItems).get_all()
                pixel_values, image_layouts, image_sizes = [], [], []

                for img in images:
                    tensor, h_tok, w_tok = self.image_tokenizer.preprocess_image_to_tensor(img)
                    pixel_values.append(tensor)
                    image_sizes.append(torch.tensor([h_tok, w_tok], dtype=torch.long))

                    rows = [dummy_img_token * w_tok for _ in range(h_tok)]
                    imgstr = self.image_tokenizer.DEFAULT_EOL_TOKEN.join(rows)
                    layout = (
                        f"{self.image_tokenizer.DEFAULT_BOI_TOKEN}{h_tok * 16}*{w_tok * 16}"
                        f"{self.image_tokenizer.DEFAULT_IMG_TOKEN}{imgstr}{self.image_tokenizer.DEFAULT_EOI_TOKEN}"
                    )
                    image_layouts.append(layout)

                # Dynamic Batching: Pad variable resolution images to Max(H, W) in the batch
                max_h = max(t.shape[1] for t in pixel_values)
                max_w = max(t.shape[2] for t in pixel_values)
                padded_pixels = [
                    torch.nn.functional.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]))
                    for t in pixel_values
                    ]

                mm_kwargs["pixel_values"] = torch.stack(padded_pixels)
                mm_kwargs["image_sizes"] = torch.stack(image_sizes)
                mm_counts["image"] = len(image_layouts)

                # Use instant replacement
                for layout in image_layouts:
                    prompt_text = prompt_text.replace(self.image_tokenizer.DEFAULT_IMAGE_PLACEHOLDER, layout, 1)

        # 2. Audio Preprocessing (Mathematical Layout & Padding Only)
        if num_audios > 0:
            with timing_ctx.record("preprocess_apertus_audios"):
                audios = inputs.mm_data_items.get_items("audio", AudioProcessorItems).get_all()
                audio_values, audio_layouts, audio_lengths = [], [], []

                for audio in audios:
                    tensor, num_tok = self.audio_tokenizer.preprocess_audio_to_tensor(audio)
                    audio_values.append(tensor.squeeze(0))
                    audio_lengths.append(torch.tensor(tensor.shape[-1], dtype=torch.long))

                    layout = (
                        f"{self.audio_tokenizer.DEFAULT_AUDIO_START_TOKEN}"
                        f"{dummy_aud_token * num_tok}"
                        f"{self.audio_tokenizer.DEFAULT_AUDIO_END_TOKEN}"
                    )
                    audio_layouts.append(layout)

                # Dynamic Batching: Pad variable length audio sequences
                max_len = max(t.shape[-1] for t in audio_values)
                padded_audio = [torch.nn.functional.pad(t, (0, max_len - t.shape[-1])) for t in audio_values]

                mm_kwargs["audio_values"] = torch.stack(padded_audio)
                mm_kwargs["audio_lengths"] = torch.stack(audio_lengths)
                mm_counts["audio"] = len(audio_layouts)

                for layout in audio_layouts:
                    prompt_text = prompt_text.replace(self.audio_tokenizer.DEFAULT_AUDIO_PLACEHOLDER, layout, 1)

        with timing_ctx.record("tokenize"):
            prompt_token_ids = list(tokenizer.encode(prompt_text, **dict(inputs.tokenization_kwargs)))

        return mm_input(
                prompt_token_ids=prompt_token_ids,
                mm_kwargs=MultiModalKwargsItems(mm_kwargs),
                prompt=prompt_text,
                )


@MULTIMODAL_REGISTRY.register_processor(
        ApertusMultiModalProcessor,
        info=ApertusProcessingInfo,
        dummy_inputs=ApertusDummyInputsBuilder,
        )
class ApertusForConditionalGeneration(ApertusForCausalLM, SupportsMultiModal):
    """
    GPU Worker Domain.
    Heavy inference executes natively. Output IDs substitute dummies via instant boolean mask.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.image_tokenizer = ApertusImageTokenizer()
        self.audio_tokenizer = ApertusAudioTokenizer()
        self.vision_tower: Any | None = None
        self.audio_tower: Any | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_keys = super().load_weights(weights)

        model_path = getattr(self.config, "_name_or_path", "")
        mm_kwargs = {"model_name_or_path": model_path}
        if model_path and os.path.isdir(model_path):
            mm_kwargs["apertus_vq_hub"] = os.path.abspath(model_path)
            mm_kwargs["apertus_audio_tokenizer_path"] = os.path.abspath(model_path)

        # Strictly synchronize precision with the Backbone to prevent VRAM fragmentation
        try:
            sample_param = next(self.parameters())
            target_device, target_dtype = sample_param.device, sample_param.dtype
        except StopIteration:
            target_device, target_dtype = torch.device("cuda"), torch.bfloat16

        mm_kwargs.update(
                {
                    "apertus_vision_tokenizer_device": str(target_device),
                    "apertus_vision_tokenizer_dtype":  target_dtype,
                    "apertus_audio_tokenizer_device":  str(target_device),
                    }
                )

        logger.info("[Apertus Worker] Loading Vision Tower natively on %s (%s)", target_device, target_dtype)
        self.vision_tower = self.image_tokenizer.load_vision_tokenizer(mm_kwargs)

        logger.info("[Apertus Worker] Loading Audio Tower natively on %s", target_device)
        self.audio_tower = self.audio_tokenizer.get_audio_tokenizer(mm_kwargs)

        return loaded_keys

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Disabled: Managed explicitly via input_ids overriding before layer lookup."""
        return []

    def embed_input_ids(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings | None = None,
            *,
            pixel_values: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            audio_values: torch.Tensor | None = None,
            audio_lengths: torch.Tensor | None = None,
            ) -> torch.Tensor:
        """Executes encoders on GPU natively and injects generated tokens into the sequence."""

        # Clone array so we can mutate IDs in place safely
        input_ids = input_ids.clone()

        # 1. GPU Vision Execution & Slice Masking
        if pixel_values is not None and self.vision_tower is not None and image_sizes is not None:
            pixel_values = pixel_values.to(device=input_ids.device, dtype=next(self.vision_tower.parameters()).dtype)

            with torch.inference_mode():
                _, _, info = self.vision_tower.encode(pixel_values)
                # Recover grid shape based on the padded Max H/W
                max_h_tok = pixel_values.shape[2] // self.image_tokenizer.EMU35_DS_FACTOR
                max_w_tok = pixel_values.shape[3] // self.image_tokenizer.EMU35_DS_FACTOR
                img_codes = info[2].view(pixel_values.shape[0], max_h_tok, max_w_tok)

                # Slice the valid tokens out of the padded batch
            valid_codes = []
            for i, size in enumerate(image_sizes):
                h_tok, w_tok = size.tolist()
                valid_codes.append(img_codes[i, :h_tok, :w_tok].flatten())

            vocab_offset = getattr(self.config, "image_token_offset", 0)
            llm_img_ids = torch.cat(valid_codes).to(torch.long) + vocab_offset

            dummy_img_id = getattr(self.config, "dummy_image_token_id", -1)
            mask = (input_ids == dummy_img_id)

            if mask.sum() != llm_img_ids.shape[0]:
                logger.error(
                    "[Apertus MM] Image mask mismatch. Processor: %d, Generated: %d", mask.sum(), llm_img_ids.shape[0]
                    )
            else:
                input_ids[mask] = llm_img_ids.to(input_ids.device)

        # 2. GPU Audio Execution & Slice Masking
        if audio_values is not None and self.audio_tower is not None and audio_lengths is not None:
            audio_values = audio_values.to(device=input_ids.device)

            with torch.inference_mode():
                audio_codes = self.audio_tower.encode_audio(audio_values)
            if audio_codes.dim() == 3:
                audio_codes = audio_codes.squeeze(1)  # [B, Max_Tokens]

            # Slice the valid tokens out of the padded batch
            valid_codes = []
            for i, length in enumerate(audio_lengths):
                num_tok = (length.item() + 600 - 1) // 600
                valid_codes.append(audio_codes[i, :num_tok])

            vocab_offset = getattr(self.config, "audio_token_offset", 262344)
            llm_audio_ids = torch.cat(valid_codes).to(torch.long) + vocab_offset

            dummy_audio_id = getattr(self.config, "dummy_audio_token_id", -1)
            mask = (input_ids == dummy_audio_id)

            if mask.sum() != llm_audio_ids.shape[0]:
                logger.error(
                    "[Apertus MM] Audio mask mismatch. Processor: %d, Generated: %d", mask.sum(), llm_audio_ids.shape[0]
                    )
            else:
                input_ids[mask] = llm_audio_ids.to(input_ids.device)

        # Handoff fully resolved ID array to standard Text Embedding mechanism
        return super().embed_input_ids(input_ids)