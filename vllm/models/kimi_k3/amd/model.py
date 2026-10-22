# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multimodal model implementation for vLLM."""

import math
from collections.abc import Hashable, Iterable
from typing import Any, ClassVar, Literal, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    KimiVisionPatchMergerProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

from ..common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from .linear import KimiLinearForCausalLM


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    HasInnerState,
    IsHybrid,
):
    """Kimi-K3 model with Kimi-K2.5 vision and KimiLinear text."""

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
    # Graph replay gives no speedup above ~1k encoder tokens (compute bound) and
    # larger graphs cost KV cache, so larger batches run the eager path.
    _MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET: ClassVar[int] = 1040
    supports_encoder_tp_data = True
    supports_mm_device_do_normalize = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        self.model_config = model_config
        self.config = config
        quant_config = vllm_config.quant_config

        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                input_norm=build_mm_input_norm(vllm_config.model_config),
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
                out_hidden_size=config.text_config.hidden_size,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )
            self.merger_projector = KimiVisionPatchMergerProjector(self.mm_projector)

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
                "merge_gather_idx",
            ],
            out_hidden_size=self.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
            self._MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET,
        )
        return min_budget, max_budget

    @staticmethod
    def _get_grid_thws(mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid_thws = mm_kwargs["grid_thws"]
        if not isinstance(grid_thws, list):
            grid_thws = grid_thws.tolist()
        return grid_thws

    @staticmethod
    def _get_pixel_values(mm_kwargs: dict[str, Any]) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values)
        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:],
            )
        return pixel_values

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        kh, kw = self.config.vision_config.merge_kernel_size
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=(h // kh) * (w // kw),
            )
            for t, h, w in self._get_grid_thws(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thws = self._get_grid_thws(mm_kwargs)
        pixel_values = self._get_pixel_values(mm_kwargs)
        source_grid = mm_kwargs["grid_thws"]

        if not indices:
            empty_grid = (
                source_grid[:0] if isinstance(source_grid, torch.Tensor) else []
            )
            return {"pixel_values": pixel_values[:0], "grid_thws": empty_grid}

        patch_counts = [t * h * w for t, h, w in grid_thws]
        offsets = [0]
        for count in patch_counts:
            offsets.append(offsets[-1] + count)
        selected_pixel_values = torch.cat(
            [pixel_values[offsets[i] : offsets[i + 1]] for i in indices]
        )
        grid_device = (
            source_grid.device if isinstance(source_grid, torch.Tensor) else None
        )
        selected_grid = torch.tensor(
            [grid_thws[i] for i in indices],
            dtype=torch.long,
            device=grid_device,
        )
        return {"pixel_values": selected_pixel_values, "grid_thws": selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        kh, kw = self.config.vision_config.merge_kernel_size
        per_item_output = (token_budget + max_batch_size - 1) // max_batch_size
        rope = self.vision_tower.encoder.rope_2d
        max_output_width = rope.max_width // kw
        max_output_height = rope.max_height // kh
        output_width = min(math.ceil(math.sqrt(per_item_output)), max_output_width)
        output_height = (per_item_output + output_width - 1) // output_width
        if output_height > max_output_height:
            output_height = max_output_height
            output_width = (per_item_output + output_height - 1) // output_height
        if output_width > max_output_width:
            raise ValueError(
                f"Encoder CUDA graph budget {token_budget} exceeds K3 RoPE "
                f"capacity for max_batch_size={max_batch_size}"
            )
        grid_thws = [
            [1, output_height * kh, output_width * kw] for _ in range(max_batch_size)
        ]

        patch_size: int | tuple[int, int] = self.config.vision_config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        total_patches = sum(t * h * w for t, h, w in grid_thws)
        pixel_values = torch.zeros(
            total_patches,
            3,
            patch_size[0],
            patch_size[1],
            device=device,
            dtype=self.vision_tower.patch_embed.input_norm.input_dtype or dtype,
        )
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thws,
            max_batch_size=max_batch_size,
            max_seqlen_override=max(
                token_budget * kh * kw,
                max(t * h * w for t, h, w in grid_thws),
            ),
            device=device,
        )
        return EncoderCudaGraphCaptureInputs(
            values=metadata | {"pixel_values": pixel_values}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        pixel_values = self._get_pixel_values(mm_kwargs)
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            self._get_grid_thws(mm_kwargs),
            max_batch_size=max_batch_size,
            device=pixel_values.device,
        )
        return EncoderCudaGraphReplayBuffers(
            values=metadata | {"pixel_values": pixel_values}
        )

    def _project_encoder_features(self, image_features: torch.Tensor) -> torch.Tensor:
        projector_dtype = next(self.mm_projector.parameters()).dtype
        if image_features.dtype != projector_dtype:
            image_features = image_features.to(projector_dtype)
        output = self.mm_projector(image_features)
        return output.reshape(-1, output.shape[-1])

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values.pop("pixel_values")
        merge_gather_idx = values.pop("merge_gather_idx", None)
        if merge_gather_idx is not None and hasattr(self, "merger_projector"):
            hidden_states = self.vision_tower(
                pixel_values, None, encoder_metadata=values
            )
            output = self.merger_projector(hidden_states, merge_gather_idx)
            return output.reshape(-1, output.shape[-1])
        if merge_gather_idx is not None:
            values["merge_gather_idx"] = merge_gather_idx
        image_features = self.vision_tower(pixel_values, None, encoder_metadata=values)
        return self._project_encoder_features(image_features)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = self._get_pixel_values(mm_kwargs)
        grid_thws = self._get_grid_thws(mm_kwargs)
        if hasattr(self, "merger_projector"):
            metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
                grid_thws,
                max_batch_size=len(grid_thws),
                device=pixel_values.device,
            )
            merge_gather_idx = metadata.pop("merge_gather_idx")
            hidden_states = self.vision_tower(
                pixel_values, None, encoder_metadata=metadata
            )
            output = self.merger_projector(hidden_states, merge_gather_idx)
            return output.reshape(-1, output.shape[-1])
        image_features = self.vision_tower(pixel_values, grid_thws)
        return self._project_encoder_features(torch.cat(image_features))

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
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
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or a list of tensors, "
                f"got {type(pixel_values)}"
            )

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
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
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return self._process_media_input(media_input)

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
        return self.language_model.compute_logits(hidden_states)

    def get_model_state_cls(self) -> type:
        return self.language_model.get_model_state_cls()

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
