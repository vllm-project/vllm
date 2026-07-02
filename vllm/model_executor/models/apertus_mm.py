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

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
    )
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
    )
from vllm.sequence import IntermediateTensors
from .apertus import ApertusForCausalLM
from .apertus_utils import ApertusAudioTokenizer, ApertusImageTokenizer
from .interfaces import MultiModalEmbeddings, SupportsMultiModal

logger = init_logger(__name__)


class ApertusProcessingInfo(BaseProcessingInfo):

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
                target_sr=ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE,
                target_channels=1,
                expected_hidden_size=self._get_expected_hidden_size(),
                )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
            self, seq_len: int, mm_counts: Mapping[str, int],
            ) -> Mapping[str, int] | None:
        del mm_counts
        vision_config = self.get_hf_config().vision_config
        ds = vision_config.get("ds_factor", 16)
        max_px = vision_config.get("max_pixels", 1400 * 1400)

        return {
            "image": min((max_px // (ds * ds)) + 512, seq_len),
            "audio": min((40 * 300) + 4, seq_len),  # 40 tokens/sec * 300 secs + formatting
            }


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        vision_config = self.info.get_hf_config().vision_config
        image_placeholder = vision_config.get("image_placeholder", "<|image|>")
        return (
                image_placeholder * mm_counts.get("image", 0) +
                ApertusAudioTokenizer.DEFAULT_AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)
        )

    def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
            ) -> MultiModalDataDict:
        vision_config = self.info.get_hf_config().vision_config
        max_px = vision_config.get("max_pixels", 1400 * 1400)
        max_side = int(max_px ** 0.5)
        audio_length = ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE
        image_overrides = mm_options.get("image")
        audio_overrides = mm_options.get("audio")

        return {
            "image": self._get_dummy_images(
                    width=max_side,
                    height=max_side,
                    num_images=mm_counts.get("image", 0),
                    overrides=image_overrides,
                    ),
            "audio": self._get_dummy_audios(
                    length=audio_length,
                    num_audios=mm_counts.get("audio", 0),
                    overrides=audio_overrides,
                    ),
            }


class ApertusMultiModalProcessor(BaseMultiModalProcessor[ApertusProcessingInfo]):
    """CPU-bound API Processor. Strict YAGNI Rule: NO heavy neural networks run here."""

    def __init__(
            self,
            info: ApertusProcessingInfo,
            dummy_inputs: BaseDummyInputsBuilder,
            *,
            cache: object | None = None,
            ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        vision_config = info.get_hf_config().vision_config
        self.image_tokenizer = ApertusImageTokenizer(vision_config)
        self.audio_tokenizer = ApertusAudioTokenizer()

    def _get_mm_fields_config(
            self, hf_inputs: object, hf_processor_mm_kwargs: Mapping[str, object],
            ) -> Mapping[str, MultiModalFieldConfig]:
        """Routes batched tensors and metadata directly to the GPU Worker's embed_multimodal kwargs."""
        return {
            "pixel_values":  MultiModalFieldConfig.batched("pixel_values"),
            "image_sizes":   MultiModalFieldConfig.batched("image_sizes"),
            "audio_values":  MultiModalFieldConfig.batched("audio_values"),
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
                        f"{self.image_tokenizer.boi_token}{h_tok * 16}*{w_tok * 16}"
                        f"{self.image_tokenizer.img_token}{imgstr}{self.image_tokenizer.eoi_token}"
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
                    prompt_text = prompt_text.replace(self.image_tokenizer.image_placeholder, layout, 1)

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
    Heavy inference executes natively on GPU. Returns embeddings of substituted tokens.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.image_tokenizer = ApertusImageTokenizer(config.vision_config)
        self.audio_tokenizer = ApertusAudioTokenizer()
        self.vision_tower: Any | None = None
        self.audio_tower: Any | None = None

        config = vllm_config.model_config.hf_config
        dummy_image_token_id = getattr(config, "dummy_image_token_id", -1)
        dummy_audio_token_id = getattr(config, "dummy_audio_token_id", -1)
        self.image_token_offset = self.config.vision_config["token_offset"]
        assert self.image_token_offset, "vision_config.token_offset must be set"
        self.audio_token_offset = self.config.audio_config["token_offset"]
        assert self.audio_token_offset, "audio_config.token_offset must be set"
        # Register the dummy placeholder token IDs to vLLM. 
        # This tells the vLLM engine to automatically construct the boolean mask (is_multimodal) 
        # matching these positions, allowing the engine to slice and route multimodal embeddings 
        # to the correct locations in the input sequence.
        self.configure_mm_token_handling(
                vocab_size=config.vocab_size,
                mm_token_ids=[dummy_image_token_id, dummy_audio_token_id],
                )

    def get_language_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: object,
            ) -> torch.Tensor | IntermediateTensors:
        # Absorb multimodal kwargs (e.g. pixel_values, etc.) which are already processed in embed_multimodal
        return self.model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                )

    def load_weights(
            self, weights: Iterable[tuple[str, torch.Tensor]],
            ) -> set[str]:
        loaded_keys = super().load_weights(weights)

        # Encoders are only executed on the first pipeline stage (first PP rank)
        if get_pp_group().is_first_rank:
            # Resolve the local path of the model directory (local or cached)
            model_id = self.vllm_config.model_config.model
            revision = self.vllm_config.model_config.revision

            if os.path.isdir(model_id):
                model_path = os.path.abspath(model_id)
            else:
                try:
                    from huggingface_hub import try_to_load_from_cache
                    cached_file = try_to_load_from_cache(
                            repo_id=model_id,
                            filename="config.json",
                            revision=revision,
                            )
                    if cached_file:
                        model_path = os.path.dirname(cached_file)
                    else:
                        model_path = getattr(self.config, "_name_or_path", model_id)
                except Exception:
                    # Fallback to the HF configuration's _name_or_path
                    model_path = getattr(self.config, "_name_or_path", model_id)

            mm_kwargs = {"model_name_or_path": model_path}
            if model_path and os.path.isdir(model_path):
                mm_kwargs["apertus_vq_hub"] = os.path.abspath(model_path)
                mm_kwargs["apertus_audio_tokenizer_path"] = os.path.abspath(
                        model_path,
                        )

            # Strictly synchronize precision with the Backbone to prevent VRAM fragmentation
            try:
                sample_param = next(self.parameters())
                target_device = sample_param.device
                target_dtype = sample_param.dtype
            except StopIteration:
                target_device = torch.device("cuda")
                target_dtype = torch.bfloat16

            mm_kwargs.update(
                    {
                        "apertus_vision_tokenizer_device": str(target_device),
                        "apertus_vision_tokenizer_dtype":  target_dtype,
                        "apertus_audio_tokenizer_device":  str(target_device),
                        },
                    )

            logger.info(
                    "[Apertus Worker] Loading Vision Tower natively on %s (%s)",
                    target_device,
                    target_dtype,
                    )
            self.vision_tower = self.image_tokenizer.load_vision_tokenizer(
                    mm_kwargs,
                    )

            logger.info(
                    "[Apertus Worker] Loading Audio Tower natively on %s",
                    target_device,
                    )
            self.audio_tower = self.audio_tokenizer.get_audio_tokenizer(
                    mm_kwargs,
                    )

        return loaded_keys

    def embed_multimodal(
            self,
            *,
            pixel_values: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            audio_values: torch.Tensor | None = None,
            audio_lengths: torch.Tensor | None = None,
            **kwargs: object,
            ) -> MultiModalEmbeddings:
        """Executes encoders on GPU natively and returns the embeddings of generated tokens."""

        # Get device of model parameters to ensure correct placement
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda")

        # 1. GPU Vision Execution & Embedding Retrieval
        if pixel_values is not None and self.vision_tower is not None and image_sizes is not None:
            # Move pixel_values to the same device and dtype as vision_tower parameters
            try:
                target_device = next(self.vision_tower.parameters()).device
                target_dtype = next(self.vision_tower.parameters()).dtype
            except StopIteration:
                target_device = pixel_values.device
                target_dtype = torch.bfloat16

            pixel_values = pixel_values.to(device=target_device, dtype=target_dtype)

            with torch.inference_mode():
                _, _, info = self.vision_tower.encode(pixel_values)
                # Recover 2D grid shape based on the padded Max H/W of the batch
                max_h_tok = pixel_values.shape[2] // self.image_tokenizer.ds_factor
                max_w_tok = pixel_values.shape[3] // self.image_tokenizer.ds_factor
                img_codes = info[2].view(pixel_values.shape[0], max_h_tok, max_w_tok)

            valid_codes_embeds = []

            for i, size in enumerate(image_sizes):
                h_tok, w_tok = size.tolist()
                # Emu3.5-VisionTokenizer is a fully convolutional network that preserves 
                # spatial layout. Because padding was added at the right/bottom during 
                # CPU data prep, we slice the top-left [:h_tok, :w_tok] active region.
                # This completely discards the padded margins and recovers the original grid.
                valid_codes_i = img_codes[i, :h_tok, :w_tok].flatten()

                # Convert codebook indices to the language model vocabulary space
                llm_img_ids_i = valid_codes_i.to(torch.long) + self.image_token_offset

                # Fetch embeddings from text embedding layer for the resolved token IDs
                img_embeds_i = super().embed_input_ids(llm_img_ids_i.to(device))
                valid_codes_embeds.append(img_embeds_i)

            return valid_codes_embeds

        # 2. GPU Audio Execution & Embedding Retrieval
        if audio_values is not None and self.audio_tower is not None and audio_lengths is not None:
            audio_values = audio_values.to(device=device)

            with torch.inference_mode():
                audio_codes = self.audio_tower.encode_audio(audio_values)
            if audio_codes.dim() == 3:
                audio_codes = audio_codes.squeeze(1)  # [B, Max_Tokens]

            valid_codes_embeds = []

            for i, length in enumerate(audio_lengths):
                # Calculate the exact number of valid tokens (1 token per 600 audio samples)
                num_tok = (length.item() + 600 - 1) // 600

                # Slice the active region to discard padded audio frames at the end
                valid_codes_i = audio_codes[i, :num_tok]

                # Convert WavTokenizer codes to language model vocabulary space
                llm_audio_ids_i = valid_codes_i.to(torch.long) + self.audio_token_offset

                # Fetch embeddings from text embedding layer for the resolved token IDs
                audio_embeds_i = super().embed_input_ids(llm_audio_ids_i.to(device))
                valid_codes_embeds.append(audio_embeds_i)

            return valid_codes_embeds

        return []

    def embed_input_ids(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings | None = None,
            *,
            is_multimodal: torch.Tensor | None = None,
            **kwargs: object,
            ) -> torch.Tensor:
        # Route to standard vLLM multi-modal merge. 
        # This takes the text embeddings generated from input_ids and overwrites the rows 
        # corresponding to the dummy placeholders (where is_multimodal is True) with 
        # the high-fidelity visual/audio embeddings generated in embed_multimodal.
        return SupportsMultiModal.embed_input_ids(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
                )
