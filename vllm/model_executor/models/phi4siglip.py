# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM support for microsoft/Phi-4-reasoning-vision-15B.

Architecture: Siglip2 vision tower + MLP projector + Phi3 language model.
"""

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    Phi3Config,
    PreTrainedConfig,
    Siglip2VisionConfig,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    cached_encode,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
)
from .lfm2_siglip2 import Siglip2Model
from .llava import LlavaMultiModalProjector
from .utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    get_fp8_padded_hidden_size,
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)

if TYPE_CHECKING:
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )

logger = init_logger(__name__)

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# The HF processor replaces "<image>" with IMAGE_TOKEN_INDEX (-200) in input_ids.
# Negative token IDs cause OverflowError during decoding, so we remap to a real
# in-vocabulary token.  The Phi-4-reasoning-vision tokenizer ships with reserved
# dummy tokens (<|dummy_0|> … <|dummy_83|>); we reuse the first one as the
# image placeholder.  This mirrors how Phi-3-vision uses its dedicated <|image|>
# token (ID 32044).
_IMAGE_TOKEN_ID = 100256  # <|dummy_0|> in the Phi-4 tokenizer


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


class Phi4SiglipProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def _get_vision_config(self) -> dict:
        return self.get_hf_config().vision_config  # type: ignore[attr-defined]

    def _get_patch_size(self) -> int:
        vc = self._get_vision_config()
        if isinstance(vc, dict):
            return vc.get("patch_size", 16)
        return getattr(vc, "patch_size", 16)

    def _get_max_num_patches(self) -> int:
        return getattr(self.get_hf_config(), "max_num_patches", 3600)

    def _get_min_num_patches(self) -> int:
        return getattr(self.get_hf_config(), "min_num_patches", 256)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        patch_size = self._get_patch_size()
        min_patches = self._get_min_num_patches()
        max_patches = self._get_max_num_patches()

        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size
        num_patches = max(num_patches_h * num_patches_w, 1)
        num_patches = max(min(num_patches, max_patches), min_patches)
        return num_patches

    def get_image_size_with_most_features(self) -> ImageSize:
        patch_size = self._get_patch_size()
        max_patches = self._get_max_num_patches()
        side = int(math.sqrt(max_patches)) * patch_size
        return ImageSize(width=side, height=side)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"image": self._get_max_num_patches()}


class Phi4SiglipDummyInputsBuilder(
    BaseDummyInputsBuilder[Phi4SiglipProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return DEFAULT_IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
        }


class Phi4SiglipMultiModalProcessor(
    BaseMultiModalProcessor[Phi4SiglipProcessingInfo],
):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_attention_mask=MultiModalFieldConfig.batched("image"),
            spatial_shapes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int):
            # Read the actual patch grid from the NaFlex processor's
            # spatial_shapes output (same pattern as LFM2-VL).  This avoids
            # predicting from raw image dimensions, which can diverge from
            # the NaFlex resize/tile logic.
            out_item = out_mm_kwargs["image"][item_idx]
            spatial_shapes = out_item["spatial_shapes"].data
            assert isinstance(spatial_shapes, torch.Tensor)
            num_tokens = int(spatial_shapes.prod().item())
            return [_IMAGE_TOKEN_ID] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    tokenizer, DEFAULT_IMAGE_TOKEN, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class Phi4SiglipImagePixelInputs(TensorSchema):
    """Dimensions:
    - bn: Batch size * number of images
    - d: Max number of patches (padded per image or across images in the batch)
    - fd: Features per patch (patch_size * patch_size * channels)
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "d", "fd", dynamic_dims={"d"}),
    ]
    pixel_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "d", dynamic_dims={"d"}),
    ]
    spatial_shapes: Annotated[torch.Tensor, TensorShape("bn", 2)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Phi4SiglipMultiModalProcessor,
    info=Phi4SiglipProcessingInfo,
    dummy_inputs=Phi4SiglipDummyInputsBuilder,
)
class Phi4ForCausalLMV(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsPP,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.vision_tower.vision_model.head.": None,
            "model.vision_tower.vision_tower.": "vision_tower.",
            "model.mm_projector.0.": "multi_modal_projector.linear_1.",
            "model.mm_projector.2.": "multi_modal_projector.linear_2.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return DEFAULT_IMAGE_TOKEN
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config: PreTrainedConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        vision_config_dict: dict = getattr(config, "vision_config", {})
        if isinstance(vision_config_dict, dict):
            if "patch_size" not in vision_config_dict:
                vision_config_dict["patch_size"] = 16
            siglip2_config = Siglip2VisionConfig(**vision_config_dict)
        else:
            siglip2_config = vision_config_dict

        vision_hidden_size: int = config.mm_hidden_size  # type: ignore[attr-defined]
        text_hidden_size: int = config.hidden_size  # type: ignore[attr-defined]

        with self._mark_tower_model(vllm_config, "image"):
            layer_idx = -2
            num_hidden_layers = siglip2_config.num_hidden_layers + layer_idx + 1

            self.vision_tower = Siglip2Model(
                siglip2_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LlavaMultiModalProjector(
                vision_hidden_size=vision_hidden_size,
                text_hidden_size=text_hidden_size,
                projector_hidden_act="gelu",
                multimodal_projector_bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        head_dim = siglip2_config.hidden_size // siglip2_config.num_attention_heads
        self._vision_tp_size = (
            1 if is_vit_use_data_parallel() else get_tensor_model_parallel_world_size()
        )
        self._vision_fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            siglip2_config.num_attention_heads, head_dim
        )
        if (
            not isinstance(self.vision_tower, StageMissingLayer)
            and len(self.vision_tower.vision_model.encoder.layers) > 0
        ):
            self._vision_attn_backend = self.vision_tower.vision_model.encoder.layers[
                0
            ].self_attn.attn.attn_backend
        else:
            self._vision_attn_backend = get_vit_attn_backend(
                head_size=head_dim,
                dtype=torch.get_default_dtype(),
            )

        # The Transformers modeling backend builds the text model from its config
        text_kwargs = config.to_dict()
        for key in ("model_type", "architectures", "auto_map", "vision_config"):
            text_kwargs.pop(key, None)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=Phi3Config(**text_kwargs),
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Phi3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.configure_mm_token_handling(
            vocab_size=config.vocab_size,  # type: ignore[attr-defined]
            mm_token_ids=[_IMAGE_TOKEN_ID],
        )

    @staticmethod
    def _spatial_shapes_to_cpu(spatial_shapes: torch.Tensor) -> torch.Tensor:
        return (
            spatial_shapes
            if spatial_shapes.device.type == "cpu"
            else spatial_shapes.cpu()
        )

    @classmethod
    def _spatial_shapes_and_lengths(
        cls, spatial_shapes: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        """Normalize CPU spatial_shapes and compute per-image patch counts."""
        spatial_shapes_cpu = cls._spatial_shapes_to_cpu(spatial_shapes)
        lengths_list = [int(h) * int(w) for h, w in spatial_shapes_cpu.tolist()]
        return spatial_shapes_cpu, lengths_list

    def _pack_pixel_values(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        lengths_list: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Pack NaFlex prefix patches without reading `pixel_attention_mask`."""
        total_tokens = sum(lengths_list)
        if total_tokens == 0:
            patch_embedding = self.vision_tower.vision_model.embeddings.patch_embedding
            return torch.empty(
                (0, patch_embedding.input_size),
                device=device,
                dtype=patch_embedding.params_dtype,
            )

        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.device != device:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            if len(lengths_list) == 1:
                return pixel_values[0, : lengths_list[0]]
            if all(length == pixel_values.shape[1] for length in lengths_list):
                return pixel_values.reshape(-1, pixel_values.shape[-1])
            parts = [
                pixel_values[i, :length]
                for i, length in enumerate(lengths_list)
                if length > 0
            ]
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        parts = []
        for pv, length in zip(pixel_values, lengths_list):
            if length <= 0:
                continue
            chunk = pv[:length]
            if chunk.device != device:
                chunk = chunk.to(device=device, non_blocking=True)
            parts.append(chunk)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Phi4SiglipImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        if pixel_values is None:
            return None

        return Phi4SiglipImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    def _get_execution_device(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
    ) -> torch.device:
        if isinstance(pixel_values, torch.Tensor):
            pv_device: torch.device | None = pixel_values.device
        elif pixel_values:
            pv_device = pixel_values[0].device
        else:
            pv_device = None
        if pv_device is not None and pv_device.type != "cpu":
            return pv_device
        for module in (self.vision_tower, self.multi_modal_projector):
            for p in module.parameters():
                if p.device.type != "cpu":
                    return p.device
        return pv_device if pv_device is not None else torch.device("cpu")

    def _prepare_vision_attn_metadata(
        self,
        seq_lens: Sequence[int],
        *,
        device: torch.device,
        pad_to_num_seqs: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Prepare backend-specific attention metadata for packed sequences."""
        cu_seqlens_np = np.cumsum([0, *seq_lens], dtype=np.int32)
        if pad_to_num_seqs is not None and len(seq_lens) < pad_to_num_seqs:
            cu_seqlens_np = np.concatenate(
                [
                    cu_seqlens_np,
                    np.full(
                        pad_to_num_seqs - len(seq_lens),
                        cu_seqlens_np[-1],
                        dtype=np.int32,
                    ),
                ]
            )

        attn_backend = self._vision_attn_backend
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            attn_backend, cu_seqlens_np, device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(attn_backend, cu_seqlens_np),
            dtype=torch.int32,
        )

        if attn_backend == AttentionBackendEnum.TORCH_SDPA:
            # `torch_sdpa_wrapper` calls `.tolist()` on `cu_seqlens`; keeping it
            # on CPU avoids both H2D and D2H syncs on the eager SDPA path.
            cu_seqlens = torch.from_numpy(cu_seqlens_np)
        else:
            cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
                attn_backend,
                cu_seqlens_np,
                hidden_size=self.vision_tower.vision_model.config.hidden_size,
                tp_size=self._vision_tp_size,
                device=device,
                fp8_padded_hidden_size=self._vision_fp8_padded_hidden_size,
            )

        return {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "sequence_lengths": sequence_lengths,
        }

    def _encode_images(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        spatial_shapes_cpu, lengths_list = self._spatial_shapes_and_lengths(
            spatial_shapes
        )
        device = self._get_execution_device(pixel_values)

        pixel_values_packed = self._pack_pixel_values(
            pixel_values, lengths_list, device
        )
        if pixel_values_packed.numel() == 0:
            empty_out = torch.empty(
                (0, int(self.config.hidden_size)),  # type: ignore[attr-defined]
                device=device,
                dtype=self.multi_modal_projector.linear_2.params_dtype,
            )
            return empty_out, lengths_list

        attn_metadata = self._prepare_vision_attn_metadata(lengths_list, device=device)
        cu_seqlens = attn_metadata["cu_seqlens"]
        max_seqlen = attn_metadata["max_seqlen"]
        assert cu_seqlens is not None and max_seqlen is not None
        vision_features = self.vision_tower(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes_cpu,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=[-2],
            sequence_lengths=attn_metadata["sequence_lengths"],
        )

        if vision_features.dim() == 3:
            vision_features = vision_features.squeeze(0)

        image_features = self.multi_modal_projector(vision_features)
        return image_features, lengths_list

    def _process_image_input(
        self, image_input: Phi4SiglipImagePixelInputs
    ) -> MultiModalEmbeddings:
        image_features, lengths_list = self._encode_images(
            image_input["pixel_values"],
            image_input["spatial_shapes"],
        )
        return torch.split(image_features, lengths_list)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        out_hidden_size = int(self.config.hidden_size)  # type: ignore[attr-defined]
        if isinstance(self.vision_tower, StageMissingLayer) or isinstance(
            self.multi_modal_projector, StageMissingLayer
        ):
            return EncoderCudaGraphConfig(
                modalities=[],
                buffer_keys=[],
                out_hidden_size=out_hidden_size,
            )

        vision_model = self.vision_tower.vision_model
        attn_backend = self._vision_attn_backend
        has_dynamic_fp8 = any(
            bool(getattr(layer.self_attn.attn, "_fp8_dynamic_scale", False))
            for layer in vision_model.encoder.layers
        )
        # Non-UVA CPU offloading leaves offloaded parameters on CPU outside
        # forward and stages temporary H2D copies per call; check all parameters
        # across both captured modules (including selective cpu_offload_params).
        weights_on_accelerator = all(
            p.device.type != "cpu"
            for module in (self.vision_tower, self.multi_modal_projector)
            for p in module.parameters()
        )
        # TORCH_SDPA splits sequences using host .tolist(); dynamic FP8 scales
        # mutate host state and reduce amax across padded tail tokens.
        graph_safe = (
            weights_on_accelerator
            and attn_backend != AttentionBackendEnum.TORCH_SDPA
            and not has_dynamic_fp8
        )
        buffer_keys = [
            "pixel_values_packed",
            "pos_embeds",
            "cu_seqlens",
            "max_seqlen",
        ]
        if attn_backend == AttentionBackendEnum.FLASHINFER:
            buffer_keys.append("sequence_lengths")
        padding_logics = MMEncoderAttention.get_cu_seqlens_padding_logics(
            attn_backend, "cu_seqlens"
        )
        return EncoderCudaGraphConfig(
            modalities=["image"] if graph_safe else [],
            buffer_keys=buffer_keys,
            padding_logics=padding_logics,
            out_hidden_size=out_hidden_size,
        )

    def get_max_frames_per_video(self) -> int:
        return 0

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        model_config = vllm_config.model_config
        multimodal_config = model_config.multimodal_config
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            model_config.max_model_len,
        )
        mm_processor_kwargs = (
            multimodal_config.mm_processor_kwargs
            if multimodal_config is not None
            else None
        ) or {}
        min_patches = mm_processor_kwargs.get(
            "min_num_patches",
            getattr(self.config, "min_num_patches", 256),
        )
        max_patches = mm_processor_kwargs.get(
            "max_num_patches",
            getattr(self.config, "max_num_patches", 3600),
        )
        assert isinstance(min_patches, (int, str))
        min_patches_int = int(min_patches)
        if isinstance(max_patches, (int, str)):
            min_patches_int = min(min_patches_int, int(max_patches))
        # Aspect-ratio grids with < min_num_patches tokens (e.g. 11x22=242)
        # still round up into this first captured bucket via round_up_to.
        min_budget = min(max(1, min_patches_int), max_budget)
        return min_budget, max_budget

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list["EncoderItemSpec"]:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        _, lengths_list = self._spatial_shapes_and_lengths(mm_kwargs["spatial_shapes"])
        return [
            EncoderItemSpec(input_size=num_patches, output_tokens=num_patches)
            for num_patches in lengths_list
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        pixel_values = mm_kwargs["pixel_values"]
        spatial_shapes_cpu = self._spatial_shapes_to_cpu(mm_kwargs["spatial_shapes"])

        if not indices:
            empty_pv: torch.Tensor | list[torch.Tensor] = (
                [] if isinstance(pixel_values, list) else pixel_values[:0]
            )
            return {
                "pixel_values": empty_pv,
                "spatial_shapes": spatial_shapes_cpu[:0],
            }

        num_items = (
            len(pixel_values)
            if isinstance(pixel_values, list)
            else pixel_values.shape[0]
        )
        if indices == list(range(num_items)):
            return {
                "pixel_values": pixel_values,
                "spatial_shapes": spatial_shapes_cpu,
            }

        # Return per-image 2D views instead of stacking full padded rows;
        # _pack_pixel_values slices each view's valid prefix in one cat.
        return {
            "pixel_values": [pixel_values[i] for i in indices],
            "spatial_shapes": spatial_shapes_cpu[indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> "EncoderCudaGraphCaptureInputs":
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        del max_frames_per_batch, dtype, path, axis_keys
        embeddings = self.vision_tower.vision_model.embeddings
        patch_dim = embeddings.patch_embedding.input_size
        patch_dtype = embeddings.patch_embedding.params_dtype
        embed_dim = embeddings.embed_dim
        pos_dtype = embeddings.position_embedding.weight.dtype

        attn_metadata = self._prepare_vision_attn_metadata(
            [token_budget],
            device=device,
            pad_to_num_seqs=max_batch_size,
        )
        values: dict[str, torch.Tensor] = {
            "pixel_values_packed": torch.zeros(
                (token_budget, patch_dim), device=device, dtype=patch_dtype
            ),
            "pos_embeds": torch.zeros(
                (token_budget, embed_dim), device=device, dtype=pos_dtype
            ),
        }
        for key, val in attn_metadata.items():
            if val is not None:
                values[key] = val
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> "EncoderCudaGraphReplayBuffers":
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        del max_batch_size, max_frames_per_batch, path
        spatial_shapes_cpu, lengths_list = self._spatial_shapes_and_lengths(
            mm_kwargs["spatial_shapes"]
        )
        embeddings = self.vision_tower.vision_model.embeddings
        device = self._get_execution_device(mm_kwargs["pixel_values"])

        pixel_values_packed = self._pack_pixel_values(
            mm_kwargs["pixel_values"], lengths_list, device
        )
        pos_embeds = embeddings.get_packed_pos_embeds(
            spatial_shapes_cpu, lengths_list=lengths_list, device=device
        )

        attn_metadata = self._prepare_vision_attn_metadata(lengths_list, device=device)
        # Keep the captured max_seqlen upper bound intact across replays.
        attn_metadata.pop("max_seqlen", None)

        values: dict[str, torch.Tensor | None] = {
            "pixel_values_packed": pixel_values_packed,
            "pos_embeds": pos_embeds,
        }
        for key, val in attn_metadata.items():
            if val is not None:
                values[key] = val
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        del path
        vision_features = self.vision_tower(
            pixel_values_packed=values["pixel_values_packed"],
            spatial_shapes=None,
            cu_seqlens=values["cu_seqlens"],
            max_seqlen=values["max_seqlen"],
            select_layers=[-2],
            pos_embeds=values["pos_embeds"],
            sequence_lengths=values.get("sequence_lengths"),
        )
        if vision_features.dim() == 3:
            vision_features = vision_features.squeeze(0)
        return self.multi_modal_projector(vision_features)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        del path
        image_features, _ = self._encode_images(
            mm_kwargs["pixel_values"],
            mm_kwargs["spatial_shapes"],
        )
        return image_features

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
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
