# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Implementation for vLLM.

Kimi-K2.5 extends Kimi-K2 with vision support.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

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

if TYPE_CHECKING:
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
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
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)
from vllm.utils.import_utils import is_numba_available
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
        processor_cls = KimiK25FusedVisionProcessor if is_numba_available() else None
        logger.info_once(
            "Using %s image preprocessing for Kimi-K2.5/K2.6 vision chunks.",
            "fused CPU" if processor_cls is not None else "remote HF",
        )
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls_overrides=processor_cls,
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
            grid_thws=MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override to use the text path instead of token path because vision chunk
        # is not considered
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs)

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
    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

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

        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
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
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
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

    # ------------------------------------------------------------------ #
    # SupportsEncoderCudaGraph protocol                                   #
    # Image-only (t == 1). Video chunks (t > 1) fall back to eager.      #
    # ------------------------------------------------------------------ #

    @property
    def _encoder_cudagraph_pad_totals(self) -> dict[int, int]:
        """Row count of each captured buffer set, keyed by cu_seqlens ptr."""
        totals = self.__dict__.get("_encoder_cg_pad_totals")
        if totals is None:
            totals = {}
            self.__dict__["_encoder_cg_pad_totals"] = totals
        return totals

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        pad_totals = self._encoder_cudagraph_pad_totals

        def pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
            # Varlen attention requires cu_seqlens[-1] to equal the number of
            # rows actually passed in. The captured buffers are sized for the
            # full token budget, so a smaller real batch has to be completed
            # with one trailing padding sequence; declaring fewer rows than the
            # buffer holds is undefined behaviour and returns NaN on FlashAttn.
            total = pad_totals.get(dst.data_ptr())
            n = min(src.shape[0], dst.shape[0])
            dst[:n].copy_(src[:n])
            dst[n:] = total if total is not None else src[-1]

        return EncoderCudaGraphConfig(
            modalities=["vision_chunk"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "merge_gather_idx",
            ],
            out_hidden_size=self.config.text_config.hidden_size,
            padding_logics={"cu_seqlens": pad_cu_seqlens},
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        # Min: 64 output tokens (e.g. ~128×128 image with patch 14 + merge 2×2).
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def _get_grid_thw_list(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[list[int]]:
        grid_thws = mm_kwargs["grid_thws"]
        if isinstance(grid_thws, torch.Tensor):
            return [[int(x) for x in row] for row in grid_thws.tolist()]
        return [[int(x) for x in row] for row in grid_thws]

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list["EncoderItemSpec"]:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        kh, kw = self.vision_tower.merge_kernel_size
        specs = []
        for t, h, w in self._get_grid_thw_list(mm_kwargs):
            if t != 1:
                # Video chunks not supported in encoder CUDA graph;
                # sentinel forces eager fallback via the manager.
                specs.append(EncoderItemSpec(input_size=t * h * w, output_tokens=2**30))
            else:
                specs.append(
                    EncoderItemSpec(
                        input_size=h * w,
                        output_tokens=(h // kh) * (w // kw),
                    )
                )
        return specs

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        grid_thw_list = self._get_grid_thw_list(mm_kwargs)
        pixel_values = mm_kwargs["pixel_values"]

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "grid_thws": pixel_values.new_zeros((0, 3), dtype=torch.long),
            }

        patch_counts = [t * h * w for t, h, w in grid_thw_list]
        cum = [0]
        for pc in patch_counts:
            cum.append(cum[-1] + pc)

        selected_pv = torch.cat(
            [pixel_values[cum[i] : cum[i + 1]] for i in indices], dim=0
        )
        selected_grid = torch.tensor(
            [grid_thw_list[i] for i in indices],
            dtype=torch.long,
            device=pixel_values.device,
        )
        return {"pixel_values": selected_pv, "grid_thws": selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> "EncoderCudaGraphCaptureInputs":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphCaptureInputs

        kh, kw = self.vision_tower.merge_kernel_size
        # Output tokens per item in the dummy grid (ceiling so total >= budget).
        per_item_out = (token_budget + max_batch_size - 1) // max_batch_size

        # Fit within RoPE precomputed max dimensions.
        rope = self.vision_tower.encoder.rope_2d
        max_wo = rope.max_width // kw
        wo = min(per_item_out, max_wo)
        ho = (per_item_out + wo - 1) // wo
        assert ho * kh <= rope.max_height, (
            f"per_item_out={per_item_out} exceeds RoPE grid capacity "
            f"(max {(rope.max_height // kh) * (rope.max_width // kw)} tokens)"
        )

        grid_thw_list = [[1, ho * kh, wo * kw] for _ in range(max_batch_size)]

        ps = self.vision_tower.patch_size
        if isinstance(ps, int):
            ps = (ps, ps)
        total_patches = max_batch_size * ho * kh * wo * kw
        dummy_pixel_values = torch.zeros(
            total_patches, 3, ps[0], ps[1], device=device, dtype=dtype
        )

        # max_seqlen must cover the worst case: one item consuming the full
        # budget, i.e. token_budget * kh * kw patches.
        # max_batch_size + 1 leaves a spare cu_seqlens slot so replay can append
        # a padding sequence covering rows the real batch does not fill.
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thw_list,
            max_batch_size=max_batch_size + 1,
            max_seqlen_override=token_budget * kh * kw,
            device=device,
        )

        values: dict[str, torch.Tensor] = {"pixel_values": dummy_pixel_values}
        values.update({k: v for k, v in metadata.items() if v is not None})

        cu_seqlens = values.get("cu_seqlens")
        if cu_seqlens is not None:
            self._encoder_cudagraph_pad_totals[cu_seqlens.data_ptr()] = total_patches

        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> "EncoderCudaGraphReplayBuffers":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

        grid_thw_list = self._get_grid_thw_list(mm_kwargs)
        pixel_values = mm_kwargs["pixel_values"]

        # Unpadded: pad_cu_seqlens completes the tail with the padding sequence
        # so cu_seqlens[-1] matches the captured buffer's row count.
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thw_list,
            max_batch_size=None,
            device=pixel_values.device,
        )

        values: dict[str, torch.Tensor | None] = {"pixel_values": pixel_values}
        values.update(metadata)
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = inputs.pop("pixel_values")
        # Remaining keys (pos_embeds, rope_freqs_cis, cu_seqlens, max_seqlen,
        # merge_gather_idx, sequence_lengths) are consumed as encoder_metadata.
        encoder_metadata = inputs

        # Fast path: uses precomputed pos_embeds + merge_gather_idx.
        vt_output = self.vision_tower(
            pixel_values, grid_thws=None, encoder_metadata=encoder_metadata
        )
        # vt_output: (total_output_tokens, kh*kw, vit_hidden_dim)

        proj_dtype = next(self.mm_projector.parameters()).dtype
        if vt_output.dtype != proj_dtype:
            vt_output = vt_output.to(proj_dtype)

        projected = self.mm_projector(vt_output)
        return projected.view(-1, self.config.text_config.hidden_size)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        media_input = self._parse_and_validate_media_input(**mm_kwargs)
        if media_input is None:
            proj_dtype = next(self.mm_projector.parameters()).dtype
            return torch.zeros(
                0,
                self.config.text_config.hidden_size,
                device=self.device,
                dtype=proj_dtype,
            )
        embeddings = self._process_media_input(media_input)
        return torch.cat(embeddings, dim=0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
