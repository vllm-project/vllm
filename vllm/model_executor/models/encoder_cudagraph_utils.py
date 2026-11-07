# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable implementations of the ``SupportsEncoderCudaGraph`` protocol."""

from collections.abc import Hashable
from typing import Any

import torch

from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)


class GridThwEncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    """Encoder CUDA graph support for Qwen2-VL-style vision towers.

    Inputs are concatenated patches (``pixel_values`` or
    ``pixel_values_videos``) with one ``(t, h, w)`` row per item in
    ``image_grid_thw`` or ``video_grid_thw``. ``self.visual`` must provide
    ``spatial_merge_size``, ``patch_embed``, ``input_norm``,
    ``prepare_encoder_metadata()`` and accept ``encoder_metadata`` in its
    forward. Hosts still implement ``get_encoder_cudagraph_config()`` and
    ``get_max_frames_per_video()``.

    This class holds only the glue to the encoder CUDA graph manager;
    model-specific metadata belongs in the vision tower. Do not add
    per-model branches here: override a hook, or implement the protocol
    directly.
    """

    visual: Any

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" in mm_kwargs:
            return "image"
        elif "video_grid_thw" in mm_kwargs:
            return "video"
        raise AssertionError("This line should be unreachable.")

    def _get_pixel_values_by_modality(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        if self.get_input_modality(mm_kwargs) == "image":
            return mm_kwargs["pixel_values"]
        return mm_kwargs["pixel_values_videos"]

    def _get_grid_thw_by_modality(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid_thw = mm_kwargs[f"{self.get_input_modality(mm_kwargs)}_grid_thw"]
        if not isinstance(grid_thw, list):
            grid_thw = grid_thw.tolist()
        return grid_thw

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        m = self.visual.spatial_merge_size
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * (h // m) * (w // m),
            )
            for t, h, w in self._get_grid_thw_by_modality(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        modality = self.get_input_modality(mm_kwargs)
        pixel_values_key = (
            "pixel_values" if modality == "image" else "pixel_values_videos"
        )
        grid_thw_key = f"{modality}_grid_thw"

        if len(indices) == 0:
            return {pixel_values_key: pixel_values[:0], grid_thw_key: []}

        cum_patches = [0]
        for t, h, w in grid_thw:
            cum_patches.append(cum_patches[-1] + t * h * w)

        selected_pv = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices]
        )
        return {
            pixel_values_key: selected_pv,
            grid_thw_key: [grid_thw[i] for i in indices],
        }

    def _get_encoder_cudagraph_per_item_output(
        self,
        token_budget: int,
        max_batch_size: int,
    ) -> int:
        """Return the output tokens of each dummy item in the capture grid."""
        # Ceil so one item using the full budget fits the captured buffers.
        return (token_budget + max_batch_size - 1) // max_batch_size

    def _get_encoder_cudagraph_capture_grid(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
    ) -> list[list[int]]:
        """Return the dummy ``grid_thw`` the graph is captured with."""
        m = self.visual.spatial_merge_size
        per_item_output = self._get_encoder_cudagraph_per_item_output(
            token_budget, max_batch_size
        )
        frames_per_item = max_frames_per_batch // max_batch_size
        if frames_per_item > 1:
            # Video layout (T > 1) so cu_seqlens, with one entry per frame, is
            # sized for video replays. Ceil keeps frames_per_item *
            # tokens_per_frame >= per_item_output.
            tokens_per_frame = (
                per_item_output + frames_per_item - 1
            ) // frames_per_item
            return [
                [frames_per_item, m, tokens_per_frame * m]
                for _ in range(max_batch_size)
            ]
        return [[1, m, per_item_output * m] for _ in range(max_batch_size)]

    def _get_encoder_cudagraph_patch_dim(self) -> int:
        patch_embed = self.visual.patch_embed
        return (
            patch_embed.proj.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size
            * patch_embed.patch_size
        )

    def _get_encoder_cudagraph_dummy_pixel_values(
        self,
        num_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the capture-time ``pixel_values``; overwritten on replay."""
        # uint8 when normalization is fused on-device.
        return torch.zeros(
            num_patches,
            self._get_encoder_cudagraph_patch_dim(),
            device=device,
            dtype=self.visual.input_norm.input_dtype or dtype,
        )

    def _get_encoder_cudagraph_capture_metadata_kwargs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
    ) -> dict[str, Any]:
        """Return extra capture kwargs for ``visual.prepare_encoder_metadata``."""
        return {}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        grid_config = self._get_encoder_cudagraph_capture_grid(
            token_budget, max_batch_size, max_frames_per_batch
        )
        dummy_pixel_values = self._get_encoder_cudagraph_dummy_pixel_values(
            sum(t * h * w for t, h, w in grid_config), device, dtype
        )
        # max_seqlen is baked into the graph at capture, so it must cover one
        # item consuming the full budget.
        metadata = self.visual.prepare_encoder_metadata(
            grid_config,
            max_batch_size=max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            max_seqlen_override=token_budget * self.visual.spatial_merge_size**2,
            device=device,
            **self._get_encoder_cudagraph_capture_metadata_kwargs(
                token_budget, max_batch_size, max_frames_per_batch
            ),
        )
        # Image and video pixel_values share the [num_patches, C*T*P*P] layout,
        # so one capture buffer serves both modalities.
        return EncoderCudaGraphCaptureInputs(
            values=metadata | {"pixel_values": dummy_pixel_values}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        grid_thw_list = self._get_grid_thw_by_modality(mm_kwargs)
        # Metadata is sized to the actual batch; the manager pads it into the
        # larger captured buffers.
        if self.get_input_modality(mm_kwargs) == "image":
            metadata = self.visual.prepare_encoder_metadata(
                grid_thw_list,
                max_batch_size=max_batch_size,
            )
        else:
            metadata = self.visual.prepare_encoder_metadata(
                grid_thw_list,
                max_frames_per_batch=max_frames_per_batch,
            )
        return EncoderCudaGraphReplayBuffers(
            values=metadata
            | {"pixel_values": self._get_pixel_values_by_modality(mm_kwargs)}
        )

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = inputs.pop("pixel_values")
        return self.visual(pixel_values, None, encoder_metadata=inputs)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        return self.visual(pixel_values, grid_thw)
