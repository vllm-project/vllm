# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Implementation for vLLM.

Kimi-K2.5 extends Kimi-K2 with vision support.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isqrt
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.parse import MultiModalDataItems, VisionChunkProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k25 import KimiK25Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


# Dummy input dimensions for profiling.
@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK25MediaPixelInputs(TensorSchema):
    """
    Media input schema for K2-VL model.

    Dimensions:
        - np: Number of patches (flattened from all media items)
        - ps: Patch size
        - nm: Number of media items
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class KimiK25ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K2.5 model.

    Provides configuration and utilities for processing both
    images and video-chunks.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K2.5 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK25Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            media_token_id=media_token_id,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"vision_chunk": None}


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    """Builds dummy inputs for Kimi-K2.5 model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_media = mm_counts.get("vision_chunk", 0)
        return self.info.media_token * num_media

    def get_dummy_mm_items(self):
        dummy_videos = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=self.info.image_processor.num_frames_per_chunk,
        )

        video_chunk_dummy_item = VisionChunkVideo(
            type="video_chunk", video_chunk=dummy_videos
        )
        video_chunk_num_tokens = self.info.media_tokens_calculator(
            video_chunk_dummy_item
        )

        image_dummy_item = VisionChunkImage(
            type="image",
            image=self._get_dummy_images(
                height=MaxImageTokenMeta.height,
                width=MaxImageTokenMeta.width,
                num_images=1,
            )[0],
        )
        image_num_tokens = self.info.media_tokens_calculator(image_dummy_item)
        # return the larger one
        if video_chunk_num_tokens >= image_num_tokens:
            return [video_chunk_dummy_item]
        else:
            return [image_dummy_item]

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        # TODO: Support mm_options for vision_chunk to allow user configuration
        dummy_items = self.get_dummy_mm_items()
        return {"vision_chunk": dummy_items}


class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    """Multi-modal processor for Kimi-K2.5.

    Handles both image and video-chunk modalities.
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Indicates how to slice media input into multiple items.

        pixel_values: [N, 3, patch_size, patch_size],
          all patches collected from B medias
        grid_thws: [B,3], each item: [N_t, N_h ,N_w],
          indicates the grid size in time/height/width direction for current item.

        by multiplying [N_t, N_h ,N_w], we get the number of patches
        for each media item, thus we can slice pixel_values by
        pixel_values[start:start + N_t*N_h*N_w] to get patches of one item.

        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override to use the text path instead of token path because vision chunk
        # is not considered
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        media_token_id = self.info.media_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items("vision_chunk", (VisionChunkProcessorItems,))
            num_media_token = self.info.media_tokens_calculator(media[item_idx])
            return [media_token_id] * num_media_token

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=[media_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsEagle,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
):
    """Kimi-K2.5 model for conditional generation.

    Supports both image and video-chunk modalities.
    Video-chunks are temporal segments (typically 4 frames) that are
    processed with temporal pooling.
    """

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # For legacy NVFP4 checkpoint compatibility:
            # see https://github.com/vllm-project/vllm/pull/33346#issuecomment-3851475033
            "language_model.layers.": "language_model.model.layers.",
            # mm projector
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Kimi-K2.5 uses video_chunk for all media types
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        elif modality == "video":
            # return a placeholder, to be replaced in the future.
            return "<|kimi_k25_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # Check for MoonViT config compatibility
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        # Build vision tower directly with KimiK25VisionConfig
        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.vision_tower = self.vision_tower.to(
                device=self.device, dtype=model_config.dtype
            )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        # The batch dimension of pixel_values has been flattened into shape[0]
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        # In some cases (e.g. with merger), grid_thws has an extra middle dimension
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    # -- SupportsEncoderCudaGraph protocol methods --
    # The runner batches multimodal items by the string modality returned
    # from ``group_and_batch_mm_kwargs``.  Kimi-K2.5 uses a single
    # ``vision_chunk`` modality for both image and video-chunk items (see
    # ``KimiK25ProcessingInfo.get_supported_mm_limits``), so the config
    # advertises that one key.  Image vs. video items differ only in
    # ``grid_thws[:, 0]`` (``t=1`` vs ``t=num_frames_per_chunk``) and
    # share the same CUDA graph.

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        return EncoderCudaGraphConfig(
            modalities=["vision_chunk"],
            # NOTE: ``EncoderCudaGraphManager._capture_budget_graph`` in
            # ``encoder_cudagraph.py:178`` hardcodes a lookup on
            # ``input_key_by_modality["image"]`` at capture time to
            # locate the dummy input buffer.  Kimi's only runtime
            # modality is ``"vision_chunk"`` (see
            # ``KimiK25ProcessingInfo``), so we alias both keys to the
            # same ``"pixel_values"`` tensor to satisfy both the
            # capture-time hardcoded lookup and the replay-time
            # ``get_input_modality`` → ``input_key_by_modality`` lookup.
            input_key_by_modality={
                "image": "pixel_values",
                "vision_chunk": "pixel_values",
            },
            buffer_keys=[
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
                "tpool_temporal_gather_idx",
                "tpool_temporal_divisor",
                "tpool_spatial_gather_idx",
            ],
            out_hidden_size=self.config.vision_config.mm_hidden_size,
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "vision_chunk"

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        kh, kw = self.vision_tower.merge_kernel_size
        # Per-item output tokens lower bound: a small 448x448 image
        # (32x32 patches, spatial_merge=2) ≈ 256 output slots.  Use
        # that as the min budget so ``_generate_budgets`` doesn't emit
        # ``[1, 2, 4, 8, ...]`` which would inflate to 30+ graphs and
        # push ``max_batch_size = max_budget // min_budget`` to an
        # absurd value (e.g. 32k).
        min_budget = 256

        # Per-item output tokens upper bound: the RoPE table footprint.
        # ``Rope2DPosEmbRepeated(head_dim, 512, 512)`` can only address
        # (h<=512, w<=512), so a single item never has more than
        # (512//kh) * (512//kw) output slots.  Capturing graphs bigger
        # than that is wasted memory — any such request will always
        # land in the eager fallback.
        rope_2d = self.vision_tower.encoder.rope_2d
        rope_max_slots = (rope_2d.max_height // kh) * (rope_2d.max_width // kw)

        # Also cap by the scheduler budget so we never over-provision
        # relative to the runtime batch.
        scheduler_cap = vllm_config.scheduler_config.max_num_batched_tokens
        max_budget = min(rope_max_slots, scheduler_cap)
        max_budget = max(max_budget, min_budget)
        return (min_budget, max_budget)

    def _get_grid_thws(self, mm_kwargs: dict[str, Any]) -> list[tuple[int, int, int]]:
        # Cache the converted tuple list on the mm_kwargs dict itself.
        # The manager invokes ``get_encoder_cudagraph_num_items``,
        # ``get_encoder_cudagraph_per_item_output_tokens``,
        # ``get_encoder_cudagraph_per_item_input_sizes``,
        # ``select_encoder_cudagraph_items`` and
        # ``prepare_encoder_cudagraph_replay_buffers`` per request — all
        # of which need this list.  Without caching we do ``.tolist()``
        # and tuple coercion 5+ times, which shows up on the hot path.
        cached = mm_kwargs.get("_cached_grid_thws_tuples")
        if cached is not None:
            return cached
        grid = mm_kwargs["grid_thws"]
        if isinstance(grid, torch.Tensor):
            grid_list = grid.reshape(-1, grid.shape[-1]).tolist()
        else:
            grid_list = [list(x) for x in grid]
        out = [(int(t), int(h), int(w)) for t, h, w in grid_list]
        mm_kwargs["_cached_grid_thws_tuples"] = out
        return out

    def get_encoder_cudagraph_num_items(self, mm_kwargs: dict[str, Any]) -> int:
        return len(self._get_grid_thws(mm_kwargs))

    def get_encoder_cudagraph_per_item_output_tokens(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[int]:
        # Kimi temporal-pools across t, so per-item output tokens are
        # ``(h // kh) * (w // kw)``, independent of ``t``.
        kh, kw = self.vision_tower.merge_kernel_size
        return [(h // kh) * (w // kw) for _, h, w in self._get_grid_thws(mm_kwargs)]

    def get_encoder_cudagraph_per_item_input_sizes(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[int]:
        return [t * h * w for t, h, w in self._get_grid_thws(mm_kwargs)]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        grid_thws_all = self._get_grid_thws(mm_kwargs)
        pixel_values: torch.Tensor = mm_kwargs["pixel_values"]
        grid_raw = mm_kwargs["grid_thws"]

        if len(indices) == 0:
            empty_grid: torch.Tensor | list[tuple[int, int, int]]
            if isinstance(grid_raw, torch.Tensor):
                empty_grid = grid_raw.reshape(-1, grid_raw.shape[-1])[:0]
            else:
                empty_grid = []
            return {
                "pixel_values": pixel_values[:0],
                "grid_thws": empty_grid,
            }

        # pixel_values is laid out as concatenated per-item patches.
        patches_per_item = [t * h * w for t, h, w in grid_thws_all]
        cum = [0]
        for p in patches_per_item:
            cum.append(cum[-1] + p)

        selected_pv = torch.cat(
            [pixel_values[cum[i] : cum[i + 1]] for i in indices], dim=0
        )
        if isinstance(grid_raw, torch.Tensor):
            flat = grid_raw.reshape(-1, grid_raw.shape[-1])
            selected_grid: torch.Tensor | list[tuple[int, int, int]] = flat[indices]
        else:
            selected_grid = [grid_thws_all[i] for i in indices]

        return {
            "pixel_values": selected_pv,
            "grid_thws": selected_grid,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        kh, kw = self.vision_tower.merge_kernel_size
        patch_size = self.vision_tower.patch_embed.patch_size
        p_h, p_w = patch_size
        per_item_output = max(token_budget // max(max_batch_size, 1), 1)

        frames_per_item = max(max_frames_per_batch // max(max_batch_size, 1), 1)

        # Pick a (nh, nw) layout that keeps each dummy item inside the
        # 2D RoPE table bounds. ``Rope2DPosEmbRepeated`` precomputes a
        # ``max_height x max_width`` table (512x512 in the Kimi-K2.5
        # config) and asserts every item satisfies
        # ``h <= max_height`` and ``w <= max_width`` where
        # ``h = nh*kh`` and ``w = nw*kw``.  Prefer a square-ish
        # layout, but clamp each dim so the product of the clamped
        # dims still covers ``per_item_output`` output slots (if the
        # budget is too large for the RoPE table, capture the largest
        # covered sub-grid; budgets beyond that land in the eager
        # fallback at runtime, which is the correct behavior).
        rope_2d = self.vision_tower.encoder.rope_2d
        max_nh = max(rope_2d.max_height // kh, 1)
        max_nw = max(rope_2d.max_width // kw, 1)

        nh_cap = min(max(isqrt(max(per_item_output, 1)), 1), max_nh)
        nw_cap = min(
            (per_item_output + nh_cap - 1) // nh_cap,
            max_nw,
        )
        nw_cap = max(nw_cap, 1)

        # ``Learnable2DInterpPosEmbDivided_fixed`` asserts
        # ``t <= self.num_frames``; clamp the capture-time t to that
        # bound so dummy inputs are valid.  Replays with t <= num_frames
        # still fit; anything beyond would need its own eager path.
        max_num_frames = self.vision_tower.patch_embed.pos_emb.num_frames
        t_cap = frames_per_item if frames_per_item > 1 else 1
        t_cap = min(t_cap, max_num_frames)
        h_cap = nh_cap * kh
        w_cap = nw_cap * kw

        grid_config: list[list[int]] = [
            [t_cap, h_cap, w_cap] for _ in range(max(max_batch_size, 1))
        ]

        total_patches = sum(t * h * w for t, h, w in grid_config)
        total_output_slots = sum((h // kh) * (w // kw) for t, h, w in grid_config)

        dummy_pixel_values = torch.randn(
            total_patches, 3, p_h, p_w, device=device, dtype=dtype
        )

        # Worst case max_seqlen for replays: a single frame covering all
        # patches that fit in the ``token_budget`` output slots.  An
        # output slot is a ``kh*kw`` window, so a frame with
        # ``token_budget`` output slots has ``token_budget * kh * kw``
        # input patches.  Clamp to the RoPE table footprint so the
        # scalar baked into the captured graph stays consistent with
        # the metadata buffers.
        max_seqlen_override = max(
            min(token_budget * kh * kw, max_nh * kh * max_nw * kw),
            total_patches,
        )

        buffers = self.vision_tower.prepare_encoder_metadata(
            grid_config,
            max_batch_size=max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            max_total_patches=total_patches,
            max_output_slots=total_output_slots,
            max_seqlen_override=max_seqlen_override,
            device=device,
        )

        mm_kwargs = {
            "pixel_values": dummy_pixel_values,
            "grid_thws": grid_config,
        }
        return EncoderCudaGraphCaptureInputs(mm_kwargs=mm_kwargs, buffers=buffers)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        grid_thws = self._get_grid_thws(mm_kwargs)
        # Match the input/output sizing the capture used so scatter copies
        # into the captured buffers are slice-consistent.
        kh, kw = self.vision_tower.merge_kernel_size
        max_total_patches = sum(t * h * w for t, h, w in grid_thws)
        max_output_slots = sum((h // kh) * (w // kw) for t, h, w in grid_thws)
        buffers = self.vision_tower.prepare_encoder_metadata(
            grid_thws,
            max_batch_size=max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            max_total_patches=max_total_patches,
            max_output_slots=max_output_slots,
        )
        return EncoderCudaGraphReplayBuffers(buffers=buffers)

    def encoder_cudagraph_forward(
        self,
        mm_kwargs: dict[str, Any],
        buffers: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        # ``grid_thws`` is only needed by the eager patch-embed
        # (for position-embedding interpolation) and the eager
        # merger; both are bypassed when ``encoder_metadata`` is
        # provided, so we do not need to materialize a tensor here.
        # A placeholder of the right type keeps downstream code happy.
        grid_thws = mm_kwargs["grid_thws"]
        if isinstance(grid_thws, torch.Tensor):
            grid_thws_tensor = grid_thws
        else:
            # Pure-Python list: not used inside the captured ops, but
            # the tower signature still accepts it.
            grid_thws_tensor = grid_thws  # type: ignore[assignment]

        target_dtype = next(self.vision_tower.parameters()).dtype
        if pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)

        vt_out = self.vision_tower(
            pixel_values, grid_thws_tensor, encoder_metadata=buffers
        )  # (max_output_slots, kh*kw, hidden)
        # Projector expects (N, kh*kw, hidden); the internal
        # ``view(-1, hidden_size=hidden*kh*kw)`` + two linears then
        # collapses it to ``(N, mm_hidden)``.
        proj_out = self.mm_projector(vt_out)
        return proj_out

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        grid_thws = mm_kwargs["grid_thws"]
        if isinstance(grid_thws, list):
            grid_thws_tensor = torch.tensor(
                grid_thws, dtype=torch.int32, device=pixel_values.device
            )
        else:
            grid_thws_tensor = grid_thws
        target_dtype = next(self.vision_tower.parameters()).dtype
        if pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)
        features = vision_tower_forward(
            self.vision_tower,
            pixel_values,
            grid_thws_tensor,
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        # Concatenate per-item outputs into a single packed tensor; the
        # manager ``_scatter_output_slices`` splits them again using
        # ``per_item_output_tokens``.
        return torch.cat(features, dim=0)

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        # NOTE(moyan): This forward will automatically batch the forward pass internally
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_media_input(media_input)
        return vision_embeddings

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
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
