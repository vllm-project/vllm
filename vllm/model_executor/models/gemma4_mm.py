# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 multimodal model (image + audio + video support).

Adds vision tower, audio tower, and multimodal embedders on top of the
text-only Gemma4ForCausalLM.  The vision/audio encoders are loaded via
AutoModel.from_config and run in eager mode while the language model uses
the vLLM-optimized path.

Video support:  Gemma4 does **not** have a native video tower.  Videos are
decomposed into timestamped image frames (up to 32 frames at 70 soft tokens
each) and fed through the same vision tower as regular images.  The
processor inserts ``mm:ss`` timestamps between frames so the model can
reason about temporal order.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from transformers import AutoModel, BatchFeature
from transformers.models.gemma4 import (
    Gemma4Config,
    Gemma4Processor,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
)

from vllm.config import VllmConfig
from vllm.config.model import get_served_model_name
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

# Video constants — match transformers Gemma4VideoProcessor defaults.
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
_VIDEO_MAX_SOFT_TOKENS = 70  # soft tokens per video frame (vs 280 for images)
_VIDEO_MAX_FRAMES = 32  # max sampled frames per video


def _get_max_soft_tokens(
    merged_kwargs: Mapping[str, object],
) -> tuple[object | None, bool]:
    """Return configured image max_soft_tokens and whether it is top-level."""
    val = merged_kwargs.get("max_soft_tokens")
    if val is not None:
        return val, True

    images_kwargs = merged_kwargs.get("images_kwargs")
    if isinstance(images_kwargs, Mapping):
        return images_kwargs.get("max_soft_tokens"), False

    return None, False


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class Gemma4ImagePixelInputs(TensorSchema):
    """
    Pre-patchified image inputs from the Gemma4 image processor.

    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches (max_patches = max_soft_tokens * pooling_kernel_size²)
        - pp: Patch pixels (patch_size² * 3)

    The Gemma4 image processor outputs pixel_values as
    (batch, max_patches, patch_pixels) — already patchified with
    zero-padding for patches beyond the real image content.
    pixel_position_ids provides (x, y) coordinates per patch,
    with (-1, -1) for padding patches.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "np", "pp", dynamic_dims={"np"}),
    ]
    pixel_position_ids: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "np", 2, dynamic_dims={"np"}),
    ]


class Gemma4AudioInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - s: Sequence length (MEL spectrogram frames)
        - f: Number of features (MEL bins)
    """

    type: Literal["audio"] = "audio"
    input_features_padded: Annotated[
        torch.Tensor, TensorShape("bn", "s", "f", dynamic_dims={"s"})
    ]
    input_features_mask: Annotated[
        torch.Tensor, TensorShape("bn", "s", dynamic_dims={"s"})
    ]


Gemma4ImageInputs = Gemma4ImagePixelInputs


class Gemma4VideoInputs(TensorSchema):
    """Video frame inputs — same tensor format as image inputs.

    Gemma4 has no separate video tower; video frames are processed
    through the vision tower at lower resolution (max_soft_tokens=70).
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", "pp"),
    ]
    pixel_position_ids_videos: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", 2),
    ]


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------


class Gemma4ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma4Config)

    def get_default_tok_params(self):
        """Gemma4's chat template already embeds a literal ``<bos>`` token in
        the rendered text.  If ``add_special_tokens=True`` (the base-class
        default), the tokenizer prepends *another* BOS, producing a
        ``[2, 2, ...]`` double-BOS sequence that the model was not trained on.

        Setting ``add_special_tokens=False`` here prevents the duplicate and
        ensures both ``llm.generate()`` and the chat/completions API behave
        correctly for IT models. For PT models (without chat template), we
        keep the default (True) to ensure BOS is added for raw prompts.
        """
        tokenizer = self.ctx.get_tokenizer()
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        params = super().get_default_tok_params()
        if has_chat_template:
            params = params.with_kwargs(add_special_tokens=False)
        return params

    def get_hf_processor(self, **kwargs: object) -> Gemma4Processor:
        return self.ctx.get_hf_processor(
            Gemma4Processor,
            **kwargs,
        )

    def validate_num_items(self, modality: str, num_items: int) -> None:
        if (
            modality == "audio"
            and num_items > 0
            and self.get_hf_config().audio_config is None
        ):
            model_config = self.ctx.model_config
            model = get_served_model_name(
                model_config.model, model_config.served_model_name
            )
            raise ValueError(
                f"Audio input was provided but the model "
                f"'{model}' does not have an audio tower. "
                f"Audio inference is only supported for Gemma4 "
                f"models that include an audio_config "
                f"(i.e., models that include an audio_config)."
            )
        super().validate_num_items(modality, num_items)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None}
        if self.get_hf_config().audio_config is not None:
            limits["audio"] = None
        limits["video"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        # Upper bound: the pooler outputs max_soft_tokens slots per image.
        # After padding is stripped the actual count is ≤ this value, but
        # vLLM needs the max for memory planning.
        tokens_per_image = config.vision_config.default_output_length
        merged_kwargs = self.ctx.get_merged_mm_kwargs({})
        val, _ = _get_max_soft_tokens(merged_kwargs)
        if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS:
            tokens_per_image = val
        tokens: dict[str, int] = {"image": tokens_per_image}
        if config.audio_config is not None:
            # Audio max tokens from the processor's audio_seq_length.
            processor = self.get_hf_processor()
            tokens["audio"] = processor.audio_seq_length
        # Video: each frame ≤ 70 soft tokens + boi + eoi + ~6 ts tokens.
        num_frames = _VIDEO_MAX_FRAMES
        mm_config = self.ctx.model_config.get_multimodal_config()
        video_opts = mm_config.limit_per_prompt.get("video")
        if (
            isinstance(video_opts, VideoDummyOptions)
            and video_opts.num_frames is not None
        ):
            num_frames = min(num_frames, video_opts.num_frames)
        tokens["video"] = num_frames * (_VIDEO_MAX_SOFT_TOKENS + 2 + 6)
        return tokens

    def get_data_parser(self) -> MultiModalDataParser:
        config = self.get_hf_config()
        kwargs: dict[str, Any] = {"video_needs_metadata": True}
        if getattr(config, "audio_config", None) is not None:
            processor = self.get_hf_processor()
            kwargs["target_sr"] = processor.feature_extractor.sampling_rate
        return MultiModalDataParser(**kwargs)

    def _compute_num_soft_tokens(
        self,
        image_width: int,
        image_height: int,
        max_soft_tokens: int | None = None,
    ) -> int:
        """Compute the number of soft tokens the vision tower produces
        for an image of the given dimensions, after padding is stripped.

        Args:
            max_soft_tokens: Override for the vision config's
                ``default_output_length``.  When *None*, the value from
                the model config is used.
        """
        vision_cfg = self.get_hf_config().vision_config
        patch_size = vision_cfg.patch_size
        pooling_kernel_size = vision_cfg.pooling_kernel_size

        if max_soft_tokens is None:
            max_soft_tokens = vision_cfg.default_output_length

        unit = patch_size * pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2
        num_patches_orig = (image_height / patch_size) * (image_width / patch_size)
        scale = math.sqrt(max_patches / num_patches_orig)
        target_h = max(unit, int(math.floor(image_height * scale / unit)) * unit)
        target_w = max(unit, int(math.floor(image_width * scale / unit)) * unit)
        num_patches = (target_h // patch_size) * (target_w // patch_size)
        # Clamp to ``max_soft_tokens``: extreme aspect ratios (e.g. 3x900)
        # cause the floor() above to round one dim up to ``unit`` while the
        # other scales freely, which over-shoots ``max_patches``. The HF
        # Gemma 4 image processor caps its vision-tower output at
        # ``max_soft_tokens``, so without this clamp the prompt-side
        # placeholder count exceeds the encoder output and
        # ``_merge_multimodal_embeddings`` crashes.
        return min(num_patches // (pooling_kernel_size**2), max_soft_tokens)

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma4Processor | None,
        max_soft_tokens: int | None = None,
    ) -> PromptUpdateDetails[list[int]]:
        """Return the dynamic image token sequence for this image.

        Computes the exact number of soft tokens the vision tower will
        produce after stripping padding.

        Args:
            max_soft_tokens: Override for the default token budget.
                When *None*, falls back to the model config value.
        """
        if processor is None:
            processor = self.get_hf_processor()

        num_soft = self._compute_num_soft_tokens(
            image_width,
            image_height,
            max_soft_tokens=max_soft_tokens,
        )
        config = self.get_hf_config()
        token_ids = (
            [config.boi_token_id]
            + [processor.image_token_id] * num_soft
            + [config.eoi_token_id]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.image_token_id)

    @staticmethod
    def _compute_audio_num_tokens(
        num_samples: int, sampling_rate: int, audio_seq_length: int
    ) -> int:
        """Replicate the audio encoder's sequence-length arithmetic.

        Mirrors: mel framing (_unfold in Gemma4AudioFeatureExtractor)
        followed by two Conv2d subsampling layers (kernel=3, stride=2,
        semicausal padding top=1, bottom=1), capped at audio_seq_length.
        """
        frame_length = int(round(sampling_rate * 20.0 / 1000.0))
        hop_length = int(round(sampling_rate * 10.0 / 1000.0))
        frame_size_for_unfold = frame_length + 1
        pad_left = frame_length // 2
        padded_samples = num_samples + pad_left
        num_mel_frames = (padded_samples - frame_size_for_unfold) // hop_length + 1
        if num_mel_frames <= 0:
            return 0
        t = num_mel_frames
        for _ in range(2):
            t = (t + 2 - 3) // 2 + 1
        return min(t, audio_seq_length)

    def get_audio_repl(
        self,
        *,
        audio_len: int,
        processor: Gemma4Processor | None,
    ) -> PromptUpdateDetails[list[int]]:
        """Return the dynamic audio token sequence for this audio.

        Computes the number of soft tokens from the audio waveform
        length by replicating the audio encoder's sequence-length
        arithmetic (mel framing + two Conv2d subsampling layers).
        """
        if processor is None:
            processor = self.get_hf_processor()

        sampling_rate = processor.feature_extractor.sampling_rate
        num_tokens = self._compute_audio_num_tokens(
            audio_len, sampling_rate, processor.audio_seq_length
        )
        config = self.get_hf_config()
        token_ids = (
            [config.boa_token_id]
            + [processor.audio_token_id] * num_tokens
            + [getattr(config, "eoa_token_id", config.eoa_token_index)]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.audio_token_id)

    def get_video_repl(
        self,
        *,
        timestamps: list[float],
        num_soft_tokens_per_frame: list[int],
        processor: Gemma4Processor,
    ) -> PromptUpdateDetails[list[int]]:
        """Build the full token replacement for one video.

        Produces the same interleaved sequence as the HF Gemma4Processor:
            mm:ss <boi><|video|>*N<eoi> mm:ss <boi><|video|>*N<eoi> ...
        """
        tokenizer = self.ctx.get_tokenizer()
        config = self.get_hf_config()

        boi_token_id = config.boi_token_id
        eoi_token_id = config.eoi_token_id
        video_token_id = processor.video_token_id

        all_token_ids: list[int] = []
        for i, (ts, n_tokens) in enumerate(zip(timestamps, num_soft_tokens_per_frame)):
            # mm:ss timestamp — matches transformers: int-truncated,
            # zero-padded.
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            ts_str = f"{minutes:02d}:{seconds:02d}"

            prefix = f" {ts_str} " if i > 0 else f"{ts_str} "
            ts_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
            all_token_ids.extend(ts_token_ids)

            all_token_ids.append(boi_token_id)
            all_token_ids.extend([video_token_id] * n_tokens)
            all_token_ids.append(eoi_token_id)

        return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)


# ---------------------------------------------------------------------------
# Dummy inputs builder
# ---------------------------------------------------------------------------


class Gemma4DummyInputsBuilder(BaseDummyInputsBuilder[Gemma4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        # Use image_token (<|image|>) with tab prefix — this is what the
        # Gemma4 chat template inserts per image (\t<|image|>).
        # _get_prompt_updates targets image_token and expands it to the
        # full_image_sequence.
        text = ("\t" + processor.image_token) * num_images
        if num_audios > 0 and processor.audio_token:
            text += processor.audio_token * num_audios
        if num_videos > 0:
            text += processor.video_token * num_videos
        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        image_processor = processor.image_processor
        # Use processor's configured image size for dummies.
        # Gemma4ImageProcessor sets size=None (it uses patch_size /
        # max_soft_tokens instead of the standard size dict), so we
        # guard against None with `or {}`.
        size = getattr(image_processor, "size", None) or {}
        img_width = size.get("width", 224)
        img_height = size.get("height", 224)

        image_overrides = mm_options.get("image") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=img_width,
                height=img_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }

        if num_audios > 0:
            audio_len = processor.feature_extractor.fft_length
            data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )

        if num_videos > 0:
            data["video"] = self._get_dummy_videos(
                width=img_width,
                height=img_height,
                num_frames=_VIDEO_MAX_FRAMES,
                num_videos=num_videos,
                overrides=video_overrides,
            )

        return data

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        videos = super()._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        videos = [v.copy() for v in videos]

        video_items: list[VideoItem] = []
        for video in videos:
            video_num_frames = video.shape[0]
            video_metadata = {
                "fps": 2.0,
                "duration": video_num_frames / 2.0,
                "total_num_frames": video_num_frames,
                "frames_indices": list(range(video_num_frames)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_items.append((video, video_metadata))

        return video_items


# ---------------------------------------------------------------------------
# Multimodal processor
# ---------------------------------------------------------------------------


class Gemma4MultiModalProcessor(BaseMultiModalProcessor[Gemma4ProcessingInfo]):
    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        # Bypass the HF processor and tokenize directly.  The HF
        # processor expands multimodal placeholders (<|video|>, etc.)
        # via get_text_with_replacements, which raises StopIteration
        # when the prompt contains placeholders without matching data.
        # The text-only path only needs token IDs, so the tokenizer
        # alone is sufficient.
        processor = self.info.get_hf_processor()
        text_inputs = processor.tokenizer([prompt_text], **tokenization_kwargs)
        input_ids = text_inputs["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        (prompt_ids,) = input_ids
        return prompt_ids

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        merged_kwargs = self.info.ctx.get_merged_mm_kwargs(mm_kwargs)
        val, is_top_level_max_soft_tokens = _get_max_soft_tokens(merged_kwargs)

        if val is not None and val not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"Unsupported max_soft_tokens value: {val}. "
                f"Valid values are {_SUPPORTED_SOFT_TOKENS}."
            )

        mm_data = dict(mm_data)

        # ---- VIDEO HANDLING ----
        # Gemma4 decomposes video into timestamped image frames.
        # Each frame is processed with max_soft_tokens=70 through the
        # same vision tower, matching transformers processing_gemma4.py.
        video_outputs: dict[str, Any] = {}
        if videos := mm_data.pop("videos", []):
            processor = self.info.get_hf_processor()

            all_video_pixel_values: list[torch.Tensor] = []
            all_video_position_ids: list[torch.Tensor] = []
            video_num_soft_tokens_per_video: list[list[int]] = []
            video_timestamps_per_video: list[list[float]] = []
            video_frame_counts: list[int] = []

            video_replacements: list[str] = []

            for item in videos:
                video_array, metadata = item

                # Convert frames to PIL images
                if isinstance(video_array, np.ndarray):
                    frames = [
                        PILImage.fromarray(video_array[i])
                        for i in range(video_array.shape[0])
                    ]
                else:
                    frames = list(video_array)

                # Compute timestamps from metadata (same as transformers)
                fps = metadata.get("fps") or 24
                frame_indices = metadata.get("frames_indices", list(range(len(frames))))
                timestamps = [idx / fps for idx in frame_indices]

                # Process frames as images with max_soft_tokens=70
                video_mm_kwargs = dict(mm_kwargs)
                video_mm_kwargs["max_soft_tokens"] = _VIDEO_MAX_SOFT_TOKENS

                dummy_prompt = ("\t" + processor.image_token) * len(frames)

                frame_outputs = super()._call_hf_processor(
                    prompt=dummy_prompt,
                    mm_data={"images": frames},
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                # Remap HF key name
                if "image_position_ids" in frame_outputs:
                    frame_outputs["pixel_position_ids"] = frame_outputs.pop(
                        "image_position_ids"
                    )

                all_video_pixel_values.append(frame_outputs["pixel_values"])
                all_video_position_ids.append(frame_outputs["pixel_position_ids"])

                # Compute soft tokens per frame
                num_soft_per_frame = []
                for img in frames:
                    w, h = img.size
                    n = self.info._compute_num_soft_tokens(
                        w, h, max_soft_tokens=_VIDEO_MAX_SOFT_TOKENS
                    )
                    num_soft_per_frame.append(n)

                video_num_soft_tokens_per_video.append(num_soft_per_frame)
                video_timestamps_per_video.append(timestamps)
                video_frame_counts.append(len(frames))

                # Build expanded replacement text for this video.
                ts_strs = [f"{int(s // 60):02d}:{int(s % 60):02d}" for s in timestamps]
                replacement = " ".join(
                    f"{t} {processor.boi_token}"
                    f"{processor.video_token * n}"
                    f"{processor.eoi_token}"
                    for t, n in zip(ts_strs, num_soft_per_frame)
                )
                video_replacements.append(replacement)

            # Replace all <|video|> placeholders at once. We split on
            # video_token to get N+1 parts, then interleave with the
            # N replacement strings. This avoids the iterative
            # split-replace bug where replacement text (which itself
            # contains <|video|> tokens) collides with later splits.
            vt = processor.video_token
            parts = prompt.split(vt, len(video_replacements))

            # NOTE: len(parts) <= len(video_replacements) + 1
            parts_with_repl: list[str] = []
            for part, repl in zip(parts, video_replacements):
                parts_with_repl.extend([part, repl])
            parts_with_repl.extend(parts[len(video_replacements) :])

            prompt = "".join(parts_with_repl)

            video_outputs = {
                "pixel_values_videos": torch.cat(all_video_pixel_values, dim=0),
                "pixel_position_ids_videos": torch.cat(all_video_position_ids, dim=0),
                "video_frame_counts": torch.tensor(video_frame_counts),
                "video_num_soft_tokens": video_num_soft_tokens_per_video,
                "video_timestamps": video_timestamps_per_video,
            }

        # The processor accepts 'audio' not 'audios'.
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        # Warn if any audio waveform exceeds the model's max duration.
        if "audio" in mm_data:
            processor = self.info.get_hf_processor()
            sr = processor.feature_extractor.sampling_rate
            max_tokens = processor.audio_seq_length
            ms_per_tok = processor.audio_ms_per_token
            max_duration_s = max_tokens * ms_per_tok / 1000.0
            audios = mm_data["audio"]
            if not isinstance(audios, (list, tuple)):
                audios = [audios]
            for i, waveform in enumerate(audios):
                duration_s = len(waveform) / sr
                if duration_s > max_duration_s:
                    logger.warning(
                        "Audio duration exceeds max: %f > %f seconds",
                        duration_s,
                        max_duration_s,
                    )
        # vLLM's call_hf_processor (context.py) re-merges
        # mm_processor_kwargs from the model config on every call via:
        #   config_kwargs | incoming_kwargs  (right side wins)
        #
        # If we strip max_soft_tokens from incoming, the re-merge puts
        # back the config's global default (e.g. 280), ignoring any
        # per-prompt override.  Instead, we keep it in the kwargs with
        # the validated per-prompt value so it wins during the merge.
        #
        # NOTE: This requires a corresponding type annotation on the
        # HF side (Gemma4ProcessorKwargs.images_kwargs) so that
        # _merge_kwargs routes max_soft_tokens into images_kwargs.
        patched_mm_kwargs = dict(mm_kwargs)
        if val is not None and is_top_level_max_soft_tokens:
            patched_mm_kwargs["max_soft_tokens"] = val

        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            patched_mm_kwargs,
            tok_kwargs,
        )

        # HF uses 'image_position_ids'; vLLM uses 'pixel_position_ids'.
        # Remap here to keep a single translation point.
        if "image_position_ids" in processed_outputs:
            processed_outputs["pixel_position_ids"] = processed_outputs.pop(
                "image_position_ids"
            )

        if "input_features" in processed_outputs:
            # Unpad per-item so each item's cache entry is
            # self-contained. The batched() field config in
            # _get_mm_fields_config will re-pad all fields to the
            # batch's max length at batch time, ensuring consistent
            # padding regardless of cache history.
            masks = processed_outputs["input_features_mask"]
            unpadded_features = [
                f[mask]
                for f, mask in zip(
                    processed_outputs["input_features"],
                    masks,
                )
            ]
            unpadded_masks = [mask[mask] for mask in masks]
            processed_outputs["input_features"] = unpadded_features
            processed_outputs["input_features_padded"] = unpadded_features
            processed_outputs["input_features_mask"] = unpadded_masks

        # Merge video outputs into the final result
        combined_outputs = dict(processed_outputs, **video_outputs)
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_position_ids=MultiModalFieldConfig.batched("image"),
            input_features_padded=MultiModalFieldConfig.batched("audio"),
            input_features_mask=MultiModalFieldConfig.batched("audio"),
        )

        # Video fields: frames stored flat, split per video by
        # video_frame_counts.
        video_frame_counts = hf_inputs.get("video_frame_counts")
        if video_frame_counts is not None:
            vfc = video_frame_counts
            if not isinstance(vfc, torch.Tensor):
                vfc = torch.tensor(vfc)
            fields.update(
                pixel_values_videos=(
                    MultiModalFieldConfig.flat_from_sizes("video", vfc)
                ),
                pixel_position_ids_videos=(
                    MultiModalFieldConfig.flat_from_sizes("video", vfc)
                ),
                video_frame_counts=MultiModalFieldConfig.batched(
                    "video",
                ),
                video_num_soft_tokens=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
                video_timestamps=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        prompt_updates = []

        if "image" in mm_items:
            # Target image_token (<|image|>) — the single placeholder the
            # Gemma4 chat template inserts once per image in the prompt.
            # vLLM tokenizes the prompt without token expansion, so only
            # one image_token exists per image in the token stream.
            # The replacement expands it to the full image sequence
            # (boi + N×image_token + eoi, where N = max_soft_tokens).
            image_token = hf_processor.image_token

            def get_replacement_image(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)
                image_size = images.get_image_size(item_idx)
                # Resolve the effective max_soft_tokens by merging
                # per-prompt kwargs with the config-level defaults,
                # consistent with how _call_hf_processor resolves it.
                # Without this merge, a missing per-prompt override
                # would fall back to vision_cfg.default_output_length
                # instead of the config's mm_processor_kwargs default.
                merged_kwargs = self.info.ctx.get_merged_mm_kwargs(
                    hf_processor_mm_kwargs,
                )
                val, _ = _get_max_soft_tokens(merged_kwargs)
                max_soft_tokens = (
                    val
                    if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS
                    else None
                )
                return self.info.get_image_repl(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                    max_soft_tokens=max_soft_tokens,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=image_token,
                    replacement=get_replacement_image,
                )
            )

        if "video" in mm_items:
            video_token = hf_processor.video_token

            def get_replacement_video(item_idx: int):
                out_item = out_mm_kwargs["video"][item_idx]
                timestamps = out_item["video_timestamps"].data
                num_soft = out_item["video_num_soft_tokens"].data
                return self.info.get_video_repl(
                    timestamps=timestamps,
                    num_soft_tokens_per_frame=num_soft,
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="video",
                    target=video_token,
                    replacement=get_replacement_video,
                )
            )

        if "audio" in mm_items:
            audio_token = hf_processor.audio_token

            def get_replacement_audio(item_idx: int):
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                return self.info.get_audio_repl(
                    audio_len=audio_len,
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

    # NOTE: Gemma3/Gemma3n override _apply_token_matches and
    # _find_mm_placeholders to merge adjacent newline tokens that arise
    # when full_image_sequence contains "\n\n" wrappers.  Gemma4's
    # full_image_sequence has NO newlines (just BOI + 280×image_token +
    # EOI), so the base class implementations work correctly as-is.


# ---------------------------------------------------------------------------
# Multimodal embedder
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """Projects vision/audio soft tokens into LM embedding space.

    Architecture:
        inputs_embeds → embedding_projection → embedding_post_projection_norm

    Unlike Gemma3n which has separate hard/soft embedding paths with
    per-path normalization and a learned embedding table, Gemma4 uses a
    simplified 2-layer design: a linear projection followed by RMSNorm
    (without learnable scale).  The checkpoint confirms this — only
    ``embedding_projection.weight`` exists; there is no embedding table
    or pre-projection norm weights.
    """

    def __init__(
        self,
        multimodal_config: Gemma4VisionConfig | Gemma4AudioConfig,
        text_config: Gemma4TextConfig,
        *,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size

        # Audio tower uses output_proj_dims (1536) rather than hidden_size
        # (1024); vision uses hidden_size (768) directly.
        embedding_dim = (
            getattr(multimodal_config, "output_proj_dims", None)
            or multimodal_config.hidden_size
        )

        self.embedding_pre_projection_norm = RMSNorm(
            embedding_dim,
            eps=self.eps,
            has_weight=False,
        )

        self.embedding_projection = ReplicatedLinear(
            embedding_dim,
            self.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embedding_projection"),
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Project soft tokens from a multimodal tower into LM space."""
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        embs_proj, _ = self.embedding_projection(embs_normed)
        return embs_proj


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsQuant,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle3,
):
    # Gemma4 clamps mm_prefix bidirectional ranges to the sliding window
    # in-kernel (HF's (causal OR blockwise) AND sliding_window). The model
    # runner reads this to keep image bidirectional ranges that exceed the
    # window instead of dropping them (which would make image attention
    # causal-only for images larger than the sliding window).
    mm_prefix_clamp_sliding_window: bool = True

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

    # Maps checkpoint prefixes to vLLM module paths.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # vision tower
            "model.vision_tower": "vision_tower",
            "model.embed_vision": "embed_vision",
            # audio tower
            "model.audio_tower.": "audio_tower.",
            "model.embed_audio.": "embed_audio.",
            # backbone
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model": "language_model.model",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype

        # Only quantize towers when the quant method supports their
        # dimensions.  BNB/torchao handle arbitrary sizes; other methods
        # (Marlin, FP8, …) require dimensions divisible by 64, which
        # the vision tower (intermediate_size=4304) does not satisfy.
        # TODO(mgoin): remove this by fixing kernel padding.
        if quant_config and quant_config.get_name() in [
            "bitsandbytes",
            "torchao",
            "compressed-tensors",
        ]:
            tower_quant = quant_config
        else:
            vision_cfg = config.vision_config
            quantizable = (
                vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None

        # ---- Vision tower (shared by image and video) ----
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=tower_quant,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                tower_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        # ---- Audio tower (variants with audio_config) ----
        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                # AutoModel.from_config does NOT call post_init(),
                # which is needed to initialize buffers that are absent
                # from the checkpoint (e.g. inv_timescales for relative
                # position embeddings, softcap, gradient_clipping).
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        # ---- Language model (vLLM optimised) ----
        with self._mark_language_model(vllm_config):
            self.language_model: Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )

            # Pre-allocate PLE buffer for CUDA graph compatibility.
            # Some variants have hidden_size_per_layer_input=None (no PLE).
            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # --- Precompute full-attention layer indices for bidi clearing ---
        self._full_attn_layer_idxs: frozenset[int] = frozenset()
        text_config = config.text_config
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                self._full_attn_layer_idxs = frozenset(
                    i for i, lt in enumerate(layer_types) if lt != "sliding_attention"
                )

        # --- MixtureOfExperts delegation to language_model ---
        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts

        gen_cfg = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None

    # ------------------------------------------------------------------ #
    # Input parsing
    # ------------------------------------------------------------------ #

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma4ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        return Gemma4ImagePixelInputs(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Gemma4AudioInputs | None:
        input_features_padded = kwargs.pop("input_features_padded", None)
        if input_features_padded is None:
            return None
        input_features_mask = kwargs.pop("input_features_mask", None)
        if input_features_mask is None:
            return None
        return Gemma4AudioInputs(
            input_features_padded=input_features_padded,
            input_features_mask=input_features_mask,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> dict[str, torch.Tensor] | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        pixel_position_ids_videos = kwargs.pop("pixel_position_ids_videos", None)
        video_frame_counts = kwargs.pop("video_frame_counts", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "pixel_position_ids_videos": pixel_position_ids_videos,
            "video_frame_counts": video_frame_counts,
        }

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, Gemma4ImageInputs | Gemma4AudioInputs | Gemma4VideoInputs | None]:
        mm_input_by_modality = {}
        for input_key in list(kwargs):
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
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

    @staticmethod
    def _encoder_chunk(
        patches_per_item: int,
        free_bytes: int,
        total_bytes: int,
        position_embedding_size: int,
    ) -> int:
        """Max chunk size whose F.one_hot transient fits in the budget.

        The dominant transient inside HF's ``Gemma4VisionPatchEmbedder.
        _position_embeddings`` is
        ``F.one_hot(clamped_positions, num_classes=position_embedding_size)``
        with shape ``(chunk, patches, 2, position_embedding_size)``,
        int64, plus its simultaneous cast to the position embedding
        table dtype. That, not the encoder residual stream, sets peak
        memory.
        """
        if patches_per_item <= 0:
            return 1
        # Half of currently-free, capped at 10% of total so we leave room
        # for the rest of profile_run / the subsequent encoder + pooler.
        budget = min(free_bytes // 2, total_bytes // 10)
        if budget <= 0:
            return 1
        # F.one_hot allocates (chunk, patches, 2, pos_emb_size) int64
        # (the inner 2 is the (x, y) coordinate axis, 8 is sizeof(int64)).
        # Outer 2x covers the int64 buffer and its concurrent bf16 cast
        # plus the matmul output that live alongside it at peak.
        cost = patches_per_item * 4 * position_embedding_size * 8
        return max(1, budget // cost) if cost > 0 else 1

    # ------------------------------------------------------------------ #
    # Image processing
    # ------------------------------------------------------------------ #

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs,
    ) -> list[torch.Tensor]:
        """Batch-encode images through the vision tower.

        Groups images by patch count (resolution bucket) so each
        encoder call processes a uniform-shape batch with no
        cross-resolution padding.  Pooling and projection are then
        applied over a single concatenated tensor for all images.
        """
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]

        vt = self.vision_tower
        vision_cfg = self.config.vision_config
        pooling_k2 = vision_cfg.pooling_kernel_size**2

        # Concurrent requests with different image resolutions may
        # arrive as a list of per-image tensors, while same-resolution
        # batches may arrive as a stacked tensor.
        buckets: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        total_images = (
            len(pixel_values)
            if isinstance(pixel_values, list)
            else pixel_values.shape[0]
        )

        for idx in range(total_images):
            pv = pixel_values[idx]
            pp = pixel_position_ids[idx]
            buckets.setdefault(pv.shape[0], []).append((idx, pv, pp))

        # Encode each resolution bucket in memory-safe chunks. Re-read
        # free memory per bucket because the previous bucket's encoder
        # pass has already allocated activations we should account for.
        last_hidden_states_map: dict[int, torch.Tensor] = {}
        for patches, items in buckets.items():
            free, total = current_platform.mem_get_info()
            max_batch_size = min(
                len(items),
                self._encoder_chunk(
                    patches, free, total, vision_cfg.position_embedding_size
                ),
            )

            for chunk_idx in range(0, len(items), max_batch_size):
                chunk_items = items[chunk_idx : chunk_idx + max_batch_size]

                pv_tensor = torch.cat(
                    [item[1].unsqueeze(0) for item in chunk_items], dim=0
                )
                pp_tensor = torch.cat(
                    [item[2].unsqueeze(0) for item in chunk_items], dim=0
                )
                pad_tensor = (pp_tensor == -1).all(dim=-1)

                inputs_embeds = vt.patch_embedder(
                    pv_tensor,
                    pp_tensor,
                    pad_tensor,
                ).to(self.model_dtype)
                encoder_outputs = vt.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=~pad_tensor,
                    pixel_position_ids=pp_tensor,
                )
                hidden_states = encoder_outputs.last_hidden_state

                for i, (orig_idx, _, _) in enumerate(chunk_items):
                    last_hidden_states_map[orig_idx] = hidden_states[i]

        # Pool per image to strip padding and reduce spatial resolution.
        all_valid_states: list[torch.Tensor] = [None] * total_images  # type: ignore[list-item]
        valid_lens = [0] * total_images

        for orig_idx in range(total_images):
            chunk_hidden = last_hidden_states_map[orig_idx]
            output_length = chunk_hidden.shape[0] // pooling_k2

            single_hidden = chunk_hidden.unsqueeze(0)
            single_pos_ids = pixel_position_ids[orig_idx].unsqueeze(0)
            padding_positions = (single_pos_ids == -1).all(dim=-1)

            pooled_states, valid_mask = vt.pooler(
                hidden_states=single_hidden,
                pixel_position_ids=single_pos_ids,
                padding_positions=padding_positions,
                output_length=output_length,
            )
            valid_states = pooled_states[valid_mask]

            if getattr(vt.config, "standardize", False):
                valid_states = (valid_states - vt.std_bias) * vt.std_scale

            all_valid_states[orig_idx] = valid_states
            valid_lens[orig_idx] = valid_states.shape[0]

        # Project all images in a single batched call.
        flat_valid_states = torch.cat(all_valid_states, dim=0).to(self.model_dtype)
        flat_proj_embs = self.embed_vision(
            inputs_embeds=flat_valid_states.unsqueeze(0)
        ).squeeze(0)

        # Split back into per-image tensors (slicing returns views).
        per_image_embeddings: list[torch.Tensor] = []
        offset = 0
        for length in valid_lens:
            per_image_embeddings.append(flat_proj_embs[offset : offset + length])
            offset += length

        return per_image_embeddings

    # ------------------------------------------------------------------ #
    # Video processing (frames through vision tower)
    # ------------------------------------------------------------------ #

    def _process_video_input(
        self,
        video_input: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Batch-encode video frames through the vision tower.

        Gemma4 has no separate video tower; video frames are images at
        lower resolution (max_soft_tokens=70).  All frames across all
        videos in the batch are encoded together in chunks, then pooled
        and projected in a single batched call.

        Returns one concatenated embedding tensor per video (not per
        frame), matching the flat_from_sizes grouping that vLLM expects
        for embed_multimodal.
        """
        pixel_values = video_input["pixel_values_videos"]
        pixel_position_ids = video_input["pixel_position_ids_videos"]
        frame_counts = video_input["video_frame_counts"]

        vt = self.vision_tower
        vision_cfg = self.config.vision_config
        pooling_k2 = vision_cfg.pooling_kernel_size**2

        if isinstance(frame_counts, torch.Tensor):
            fc_list = frame_counts.tolist()
        else:
            fc_list = list(frame_counts)

        total_frames = pixel_values.shape[0]
        free, total = current_platform.mem_get_info()
        max_batch_size = min(
            total_frames,
            self._encoder_chunk(
                pixel_values.shape[1],
                free,
                total,
                vision_cfg.position_embedding_size,
            ),
        )

        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        # Encode frames in chunks bounded by _encoder_chunk.
        last_hidden_states_list: list[torch.Tensor] = []
        for i in range(0, total_frames, max_batch_size):
            pv_chunk = pixel_values[i : i + max_batch_size]
            pp_chunk = pixel_position_ids[i : i + max_batch_size]
            pad_chunk = padding_positions[i : i + max_batch_size]

            inputs_embeds = vt.patch_embedder(
                pv_chunk,
                pp_chunk,
                pad_chunk,
            ).to(self.model_dtype)
            encoder_outputs = vt.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=~pad_chunk,
                pixel_position_ids=pp_chunk,
            )
            last_hidden_states_list.append(encoder_outputs.last_hidden_state)

        last_hidden_states = torch.cat(last_hidden_states_list, dim=0)

        # Pool per frame to strip padding and reduce spatial resolution.
        output_length = pixel_values.shape[1] // pooling_k2
        all_frame_valid_states: list[torch.Tensor] = []
        frame_valid_lens: list[int] = []

        for i in range(total_frames):
            single_hidden = last_hidden_states[i].unsqueeze(0)
            single_pos_ids = pixel_position_ids[i].unsqueeze(0)
            single_pad_pos = padding_positions[i].unsqueeze(0)

            pooled_states, valid_mask = vt.pooler(
                hidden_states=single_hidden,
                pixel_position_ids=single_pos_ids,
                padding_positions=single_pad_pos,
                output_length=output_length,
            )
            valid_states = pooled_states[valid_mask]

            if getattr(vt.config, "standardize", False):
                valid_states = (valid_states - vt.std_bias) * vt.std_scale

            all_frame_valid_states.append(valid_states)
            frame_valid_lens.append(valid_states.shape[0])

        # Project all frames in a single batched call.
        flat_valid_states = torch.cat(all_frame_valid_states, dim=0).to(
            self.model_dtype
        )
        flat_proj_embs = self.embed_vision(
            inputs_embeds=flat_valid_states.unsqueeze(0)
        ).squeeze(0)

        # Regroup into per-video tensors (slicing returns views).
        per_video_embeddings: list[torch.Tensor] = []
        frame_idx = 0
        offset = 0
        for count in fc_list:
            video_tokens = sum(frame_valid_lens[frame_idx : frame_idx + count])
            per_video_embeddings.append(flat_proj_embs[offset : offset + video_tokens])
            offset += video_tokens
            frame_idx += count

        return per_video_embeddings

    # ------------------------------------------------------------------ #
    # Audio processing
    # ------------------------------------------------------------------ #

    def _process_audio_input(
        self,
        audio_input: Gemma4AudioInputs,
    ) -> list[torch.Tensor]:
        input_features = audio_input["input_features_padded"].squeeze(1)
        input_features_mask = audio_input["input_features_mask"].squeeze(1)

        # Run audio tower — mask convention: True=valid, False=padding.
        audio_outputs = self.audio_tower(input_features, input_features_mask)
        if isinstance(audio_outputs, tuple):
            audio_encodings, audio_mask = audio_outputs
        else:
            audio_encodings = audio_outputs.last_hidden_state
            audio_mask = audio_outputs.attention_mask

        # Project into LM embedding space.
        audio_features = self.embed_audio(inputs_embeds=audio_encodings)

        # Strip padding per-batch element: only keep valid (non-padding)
        # tokens.
        per_audio = []
        for enc, mask in zip(audio_features, audio_mask, strict=True):
            per_audio.append(enc[mask])  # [num_real, hidden_size]

        return per_audio

    # ------------------------------------------------------------------ #
    # MultiModalEmbeddings interface
    # ------------------------------------------------------------------ #

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        multimodal_embeddings: list[torch.Tensor] = []

        for modality, multimodal_input in mm_input_by_modality.items():
            if multimodal_input is None:
                continue
            if modality == "image":
                multimodal_embeddings.extend(
                    self._process_image_input(multimodal_input)
                )
            elif modality == "video":
                multimodal_embeddings.extend(
                    self._process_video_input(multimodal_input)
                )
            elif modality == "audio":
                multimodal_embeddings.extend(
                    self._process_audio_input(multimodal_input)
                )

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cache per-layer embeddings (PLE) for the language model's
        # forward pass.  During profiling embed_input_ids is not called,
        # so the pre-allocated zeros are used instead.
        if self.per_layer_embeddings is not None:
            # Mask multimodal tokens (image/audio) to 0 for PLE
            # computation (using token_type_ids == 0 as text_mask).
            # Replicate this: map image token positions to token 0.
            if is_multimodal is not None:
                ple_input_ids = torch.where(
                    is_multimodal.to(input_ids.device, non_blocking=True),
                    torch.zeros_like(input_ids),
                    input_ids,
                )
            else:
                ple_input_ids = input_ids

            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                ple_input_ids
            )
            if per_layer_inputs is not None:
                per_layer_inputs = per_layer_inputs.reshape(
                    -1,
                    self.config.text_config.num_hidden_layers,
                    self.config.text_config.hidden_size_per_layer_input,
                )
                self.per_layer_embeddings[: per_layer_inputs.shape[0]].copy_(
                    per_layer_inputs
                )

        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Select the pre-cached PLEs for this batch (None when PLE
        # is disabled for variants without PLE).
        per_layer_inputs = (
            self.per_layer_embeddings[: inputs_embeds.shape[0]]
            if self.per_layer_embeddings is not None and inputs_embeds is not None
            else None
        )

        # Gemma4 bidi: clear mm_prefix_range for full_attention layers.
        # Must run here (outside @support_torch_compile boundary) because
        # _run_decoder_layers is inside a compiled graph where Python
        # side effects are eliminated.
        self._clear_mm_prefix_for_full_attn_layers()

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
        logits = self.language_model.compute_logits(hidden_states)
        if logits is not None and self._suppress_token_ids:
            logits[:, self._suppress_token_ids] = -float("inf")
        return logits

    # ------------------------------------------------------------------ #
    # Bidirectional attention helpers
    # ------------------------------------------------------------------ #

    def _clear_mm_prefix_for_full_attn_layers(self) -> None:
        """Clear mm_prefix_range for non-sliding layers.

        Gemma4 with use_bidirectional_attention='vision' applies
        bidirectional attention only to sliding_attention layers.
        Full attention layers use plain causal masking.

        Uses _full_attn_layer_idxs (precomputed in __init__) for O(1)
        lookup instead of per-call regex parsing.
        """
        if not self._full_attn_layer_idxs:
            return

        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            return

        def _process(metadata_dict: dict) -> None:
            for layer_name, metadata in metadata_dict.items():
                if ".layers." not in layer_name:
                    continue
                try:
                    layer_idx = int(layer_name.split(".layers.")[1].split(".")[0])
                except (ValueError, IndexError):
                    continue
                if layer_idx in self._full_attn_layer_idxs:
                    if hasattr(metadata, "mm_prefix_range"):
                        metadata.mm_prefix_range = None
                    if hasattr(metadata, "mm_prefix_range_tensor"):
                        metadata.mm_prefix_range_tensor = None

        if isinstance(attn_metadata, list):
            for ub_metadata in attn_metadata:
                _process(ub_metadata)
        elif isinstance(attn_metadata, dict):
            _process(attn_metadata)

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Some checkpoints have vestigial embed_vision.embedding and
        # embed_audio.embedding weights from the Gemma3n architecture
        # that are not used by Gemma4's MultimodalEmbedder (which only
        # has embedding_projection + embedding_post_projection_norm).
        ignore_prefixes = [
            "embed_vision.embedding.",
            "embed_audio.embedding.",
        ]
        # Models without audio tower should skip audio weights entirely.
        if self.audio_tower is None:
            ignore_prefixes.extend(
                [
                    "audio_tower.",
                    "embed_audio.",
                ]
            )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ------------------------------------------------------------------ #
    # LoRA / multimodal mapping
    # ------------------------------------------------------------------ #

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        connectors = ["embed_vision"]
        tower_models = ["vision_tower"]
        if self.audio_tower is not None:
            connectors.append("embed_audio")
            tower_models.append("audio_tower")

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=connectors,
            tower_model=tower_models,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        if modality == "audio":
            return "<audio_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")
