# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multimodal Apertus Pipeline optimized for native vLLM asynchronous execution.

Architecture Contract:
1. Processor (CPU): Deterministic per-item tensor creation and dummy token
   injection. Zero neural network inference is executed here.
2. Worker (GPU): Native Emu3.5/WavTokenizer execution and O(1) boolean mask
   ID substitution.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
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
from .apertus_emu35 import IBQ
from .apertus_utils import ApertusAudioTokenizer, ApertusImageTokenizer
from .apertus_wavetokenizer import WavTokenizer40
from .interfaces import MultiModalEmbeddings, SupportsMultiModal

logger = init_logger(__name__)


class ApertusProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        audio_config = self.get_hf_config().audio_config
        target_sr = audio_config.get("target_sampling_rate", 24000)
        return MultiModalDataParser(
            target_sr=target_sr,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del mm_counts
        vision_config = self.get_hf_config().vision_config
        ds = vision_config.get("ds_factor", 16)
        max_px = vision_config.get("max_pixels", 1400 * 1400)

        return {
            "image": min((max_px // (ds * ds)) + 512, seq_len),
            "audio": min(
                (40 * 300) + 4, seq_len
            ),  # 40 tokens/sec * 300 secs + formatting
        }


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        vision_config = self.info.get_hf_config().vision_config
        image_placeholder = vision_config.get("image_placeholder", "<|image|>")
        audio_config = self.info.get_hf_config().audio_config
        audio_placeholder = audio_config.get("audio_placeholder", "<|audio|>")
        return image_placeholder * mm_counts.get(
            "image", 0
        ) + audio_placeholder * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        vision_config = self.info.get_hf_config().vision_config
        max_px = vision_config.get("max_pixels", 1400 * 1400)
        max_side = int(max_px**0.5)
        audio_config = self.info.get_hf_config().audio_config
        audio_length = audio_config.get("target_sampling_rate", 24000)
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
        audio_config = info.get_hf_config().audio_config
        self.image_tokenizer = ApertusImageTokenizer(vision_config)
        self.audio_tokenizer = ApertusAudioTokenizer(audio_config)

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Routes per-item tensors to the GPU Worker's embed_multimodal kwargs."""
        # The first argument of batched() is the MODALITY
        # ("image"/"audio"), not the field name -- the engine looks items up
        # by modality during profiling and scheduling.
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "audio_values": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(self, *args: Any, **kwargs: Any) -> Sequence[PromptUpdate]:
        return []

    def _preprocess_image_item(
        self,
        image: Any,
        dummy_token: str,
    ) -> tuple[torch.Tensor, str]:
        tensor, h_tok, w_tok = self.image_tokenizer.preprocess_image_to_tensor(image)
        rows = [dummy_token * w_tok for _ in range(h_tok)]
        image_tokens = self.image_tokenizer.eol_token.join(rows)
        layout = (
            f"{self.image_tokenizer.boi_token}{h_tok}*{w_tok}"
            f"{self.image_tokenizer.img_token}{image_tokens}"
            f"{self.image_tokenizer.eoi_token}"
        )
        return tensor, layout

    def _preprocess_audio_item(
        self,
        audio: Any,
        dummy_token: str,
    ) -> tuple[torch.Tensor, str]:
        tensor, num_tokens = self.audio_tokenizer.preprocess_audio_to_tensor(audio)
        layout = (
            f"{self.audio_tokenizer.audio_start_token}"
            f"{dummy_token * num_tokens}"
            f"{self.audio_tokenizer.audio_end_token}"
        )
        return tensor.squeeze(0), layout

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt
            if isinstance(inputs.prompt, str)
            else tokenizer.decode(inputs.prompt)
        )
        config = self.info.get_hf_config()

        # Must be defined as distinct single tokens in tokenizer.json
        dummy_img_token = getattr(config, "dummy_image_token", "<|visual token 0|>")
        dummy_aud_token = getattr(config, "dummy_audio_token", "<|audio token 0|>")

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)

        mm_kwargs: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        mm_counts: dict[str, int] = {}

        # 1. Vision Preprocessing (Mathematical Layout Only)
        if num_images > 0:
            with timing_ctx.record("preprocess_apertus_images"):
                images = inputs.mm_data_items.get_items(
                    "image", ImageProcessorItems
                ).get_all()
                pixel_values, image_layouts = [], []
                for image in images:
                    value, layout = self._preprocess_image_item(
                        image, dummy_img_token
                    )
                    pixel_values.append(value)
                    image_layouts.append(layout)

                mm_kwargs["pixel_values"] = pixel_values
                mm_counts["image"] = len(image_layouts)
                for layout in image_layouts:
                    prompt_text = prompt_text.replace(
                        self.image_tokenizer.image_placeholder, layout, 1
                    )

        # 2. Audio Preprocessing (Mathematical Layout Only)
        if num_audios > 0:
            with timing_ctx.record("preprocess_apertus_audios"):
                audios = inputs.mm_data_items.get_items(
                    "audio", AudioProcessorItems
                ).get_all()
                audio_values, audio_layouts = [], []
                for audio in audios:
                    value, layout = self._preprocess_audio_item(audio, dummy_aud_token)
                    audio_values.append(value)
                    audio_layouts.append(layout)

                mm_kwargs["audio_values"] = audio_values
                mm_counts["audio"] = len(audio_layouts)

                for layout in audio_layouts:
                    prompt_text = prompt_text.replace(
                        self.audio_tokenizer.audio_placeholder, layout, 1
                    )

        with timing_ctx.record("tokenize"):
            prompt_token_ids = list(
                tokenizer.encode(prompt_text, **dict(inputs.tokenization_kwargs))
            )

        # mm_input() requires mm_hashes and mm_placeholders
        # since upstream 08a8a4af. Placeholders are one PlaceholderRange per
        # item spanning its layout tokens, with is_embed marking the dummy
        # positions the GPU worker overwrites.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        def _single_token_id(token_text: str) -> int:
            ids = list(tokenizer.encode(token_text, add_special_tokens=False))
            if len(ids) != 1:
                raise ValueError(
                    f"Apertus MM: expected {token_text!r} to be one special "
                    f"token, got {ids}"
                )
            return ids[0]

        def _span_ranges(
            start_token: str, end_token: str, dummy_token_id: int, count: int
        ) -> list[PlaceholderRange]:
            # Anchor on the atomic start/end special tokens directly in the
            # prompt ids: unlike re-encoding the layout string standalone,
            # special tokens cannot merge with surrounding text under BPE.
            start_id = _single_token_id(start_token)
            end_id = _single_token_id(end_token)
            ranges: list[PlaceholderRange] = []
            pos = 0
            for _ in range(count):
                try:
                    s = prompt_token_ids.index(start_id, pos)
                    e = prompt_token_ids.index(end_id, s)
                except ValueError as exc:
                    raise ValueError(
                        f"Apertus MM: {start_token!r}/{end_token!r} pair not "
                        f"found in prompt (search from {pos})"
                    ) from exc
                span = prompt_token_ids[s : e + 1]
                is_embed = torch.tensor(
                    [tok == dummy_token_id for tok in span], dtype=torch.bool
                )
                ranges.append(
                    PlaceholderRange(offset=s, length=e - s + 1, is_embed=is_embed)
                )
                pos = e + 1
            return ranges

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if num_images > 0:
            mm_placeholders["image"] = _span_ranges(
                self.image_tokenizer.boi_token,
                self.image_tokenizer.eoi_token,
                getattr(config, "dummy_image_token_id", 131272),
                num_images,
            )
        if num_audios > 0:
            mm_placeholders["audio"] = _span_ranges(
                self.audio_tokenizer.audio_start_token,
                self.audio_tokenizer.audio_end_token,
                getattr(config, "dummy_audio_token_id", 262344),
                num_audios,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems.from_hf_inputs(
                mm_kwargs, self._get_mm_fields_config(mm_kwargs, {})
            ),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=prompt_text,
        )


@MULTIMODAL_REGISTRY.register_processor(
    ApertusMultiModalProcessor,
    info=ApertusProcessingInfo,
    dummy_inputs=ApertusDummyInputsBuilder,
)
class ApertusForConditionalGeneration(ApertusForCausalLM, SupportsMultiModal):
    # Required by vLLM's chat serving to insert the
    # modality placeholder when flattening OpenAI content parts. Without it
    # image/audio parts silently vanish from the prompt.
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError(f"Unsupported modality: {modality}")

    """
    GPU Worker Domain.
    Heavy inference executes natively on GPU. Returns embeddings of substituted tokens.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.image_tokenizer = ApertusImageTokenizer(config.vision_config)
        self.audio_tokenizer = ApertusAudioTokenizer(config.audio_config)
        self.vision_tower: IBQ | None = None
        self.audio_tower: WavTokenizer40 | None = None

        dummy_image_token_id = getattr(config, "dummy_image_token_id", 131272)
        dummy_audio_token_id = getattr(config, "dummy_audio_token_id", 262344)
        self.image_token_offset = self.config.vision_config["token_offset"]
        assert self.image_token_offset, "vision_config.token_offset must be set"
        self.audio_token_offset = self.config.audio_config["token_offset"]
        assert self.audio_token_offset, "audio_config.token_offset must be set"
        # Register the dummy placeholder token IDs to vLLM.
        # This tells the vLLM engine to automatically construct the boolean mask
        # (is_multimodal) matching these positions, allowing the engine to slice
        # and route multimodal embeddings to the correct locations in the input
        # sequence.
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
        # Absorb multimodal kwargs (e.g. pixel_values, etc.) which are
        # already processed in embed_multimodal
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
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

            # Multimodal tokenizers use FP32 independently of the backbone.
            try:
                sample_param = next(self.parameters())
                target_device = sample_param.device
            except StopIteration:
                target_device = torch.device("cuda")
            tokenizer_dtype = torch.float32

            logger.info(
                "[Apertus Worker] Loading Vision Tower natively on %s (%s)",
                target_device,
                tokenizer_dtype,
            )
            self.vision_tower = self.image_tokenizer.load_vision_tokenizer(
                model_path=model_path,
                device=str(target_device),
                dtype=tokenizer_dtype,
                vision_config=self.config.vision_config,
            )

            logger.info(
                "[Apertus Worker] Loading Audio Tower natively on %s",
                target_device,
            )
            self.audio_tower = self.audio_tokenizer.load_audio_tokenizer(
                model_path=model_path,
                device=str(target_device),
                dtype=tokenizer_dtype,
                audio_config=self.config.audio_config,
            )

            loaded_keys.update(
                    name
                    for name, _ in self.vision_tower.named_parameters(
                            prefix="vision_tower")
                    )
            loaded_keys.update(
                    name
                    for name, _ in self.audio_tower.named_parameters(
                            prefix="audio_tower")
                    )

        return loaded_keys

    def _get_module_device_dtype(
        self,
        module: torch.nn.Module,
    ) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return parameter.device, parameter.dtype

    def _encode_image_to_llm_ids(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        target_device, target_dtype = self._get_module_device_dtype(self.vision_tower)
        image = image.unsqueeze(0).to(device=target_device, dtype=target_dtype)
        with torch.inference_mode():
            _, _, info = self.vision_tower.encode(image)

        h_tok = image.shape[2] // self.image_tokenizer.ds_factor
        w_tok = image.shape[3] // self.image_tokenizer.ds_factor
        valid_codes = info[2].view(h_tok, w_tok).flatten()
        return valid_codes.to(torch.long) + self.image_token_offset

    def _encode_audio_to_llm_ids(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        target_device, target_dtype = self._get_module_device_dtype(self.audio_tower)
        with torch.inference_mode():
            valid_codes = self.audio_tower.encode_audio(
                audio.unsqueeze(0).to(device=target_device, dtype=target_dtype)
            ).squeeze(0)

        return valid_codes.to(torch.long) + self.audio_token_offset

    def _parse_mm_modality_and_encoder(
        self,
        **kwargs: object,
    ) -> tuple[
        torch.Tensor | list[torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ] | None:
        pixel_values = kwargs.get("pixel_values")
        audio_values = kwargs.get("audio_values")

        # vLLM batches each encoder call by modality. Both fields cannot be set.
        if pixel_values is not None:
            if self.vision_tower is None:
                return None
            return pixel_values, self._encode_image_to_llm_ids
        if audio_values is not None:
            if self.audio_tower is None:
                return None
            return audio_values, self._encode_audio_to_llm_ids

        return None

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Encode one modality batch into language embeddings."""
        selected_encoder = self._parse_mm_modality_and_encoder(**kwargs)
        if selected_encoder is None:
            return []
        values, encode_to_llm_ids = selected_encoder

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda")

        items = list(values.unbind(0)) if isinstance(values, torch.Tensor) else values

        ids_per_item = [encode_to_llm_ids(item) for item in items]
        lengths = [ids.shape[0] for ids in ids_per_item]

        all_ids = torch.cat(ids_per_item).to(device)
        all_embeds = super().embed_input_ids(all_ids)
        return list(all_embeds.split(lengths))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Route to standard vLLM multi-modal merge.
        # This takes the text embeddings generated from input_ids and
        # overwrites the rows corresponding to the dummy placeholders (where
        # is_multimodal is True) with the high-fidelity visual/audio
        # embeddings generated in embed_multimodal.
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
