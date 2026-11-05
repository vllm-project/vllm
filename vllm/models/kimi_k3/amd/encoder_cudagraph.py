# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph support for the Kimi-K3 vision tower.

Mixin implementing the SupportsEncoderCudaGraph protocol for any host class
that holds the vision tower as ``self.vision_tower`` (a
``MoonViT3dPretrainedModel``) and projector as ``self.mm_projector``.
"""

import math
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

if TYPE_CHECKING:
    from torch import nn

    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel
    from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )


class KimiK3EncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    """SupportsEncoderCudaGraph for hosts with ``self.vision_tower``."""

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
    # Replay removes launch overhead, which only pays off up to ~512 output
    # tokens; padded larger graphs are not faster, so larger batches run eagerly.
    _MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET: ClassVar[int] = 512

    vision_tower: "MoonViT3dPretrainedModel"
    mm_projector: "nn.Module"
    config: "KimiK3Config"
    model_config: "ModelConfig"
    hidden_size: int

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
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
        self, vllm_config: "VllmConfig"
    ) -> tuple[int, int]:
        # Smallest realistic input: a 224x224 image is 16x16 patches, 8x8 = 64
        # tokens after the 2x2 merge. Smaller items replay the 64-token graph.
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

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list["EncoderItemSpec"]:
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
    ) -> "EncoderCudaGraphCaptureInputs":
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
    ) -> "EncoderCudaGraphReplayBuffers":
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

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        metadata = {k: v for k, v in values.items() if k != "pixel_values"}
        hidden_states = self.vision_tower(
            values["pixel_values"], None, encoder_metadata=metadata
        )
        dtype = next(self.mm_projector.parameters()).dtype
        output = self.mm_projector(hidden_states.to(dtype))
        return output.reshape(-1, output.shape[-1])

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = self._get_pixel_values(mm_kwargs)
        values = self.vision_tower.prepare_encoder_cudagraph_metadata(
            self._get_grid_thws(mm_kwargs),
            max_batch_size=None,
            device=pixel_values.device,
        )
        return self.encoder_cudagraph_forward(values | {"pixel_values": pixel_values})
