# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph support for the MiniMax M3 vision tower.

Mixin implementing the SupportsEncoderCudaGraph protocol for any host class
that holds the vision tower as ``self.vision_tower`` (a
``MiniMaxVLVisionModel``). The manager routes image items through this mixin;
video items use the eager encoder path.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.multimodal import MultiModalConfig
    from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )


class MiniMaxM3EncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    """SupportsEncoderCudaGraph for hosts with ``self.vision_tower``."""

    vision_tower: "MiniMaxVLVisionModel"
    multimodal_config: "MultiModalConfig | None"
    _encoder_cg_pad_totals: dict[int, int]

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        if (
            self.multimodal_config is not None
            and self.multimodal_config.get_limit_per_prompt("image") == 0
        ):
            raise ValueError(
                "cudagraph_mm_encoder requires the MiniMax M3 vision tower, "
                "but the image limit is 0. Disable cudagraph_mm_encoder."
            )

        mm_cfg = self.multimodal_config
        if (
            mm_cfg is not None
            and mm_cfg.mm_encoder_attn_dtype == "fp8"
            and mm_cfg.mm_encoder_fp8_scale_path is None
        ):
            raise ValueError(
                "cudagraph_mm_encoder requires static FP8 scales for the "
                "MiniMax M3 vision encoder. Set --mm-encoder-fp8-scale-path "
                "or disable cudagraph_mm_encoder."
            )

        # FlashInfer reads max_seqlen on the host and TORCH_SDPA calls
        # .tolist() on CUDA cu_seqlens; neither can be captured.
        backend = self.vision_tower.vision_model.attn_backend
        if backend not in (
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        ):
            raise ValueError(
                f"cudagraph_mm_encoder is not supported with the "
                f"{backend.name} ViT attention backend for MiniMax M3. Set "
                "--mm-encoder-attn-backend FLASH_ATTN or ROCM_AITER_FA "
                "(ROCm), or disable cudagraph_mm_encoder."
            )

        pad_totals = self._encoder_cg_pad_totals

        def pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
            # Varlen attention requires cu_seqlens[-1] to equal the number of
            # rows actually passed in. The captured buffers are sized for the
            # full token budget, so a smaller real batch has to be completed
            # with one trailing padding sequence; declaring fewer rows than the
            # buffer holds is undefined behaviour and returns NaN on FlashAttn.
            total = pad_totals.get(dst.data_ptr())
            if total is None:
                raise RuntimeError("cu_seqlens replay buffer was not registered")
            n = src.shape[0]
            dst[:n].copy_(src[:n])
            dst[n:] = total

        def pad_rope_planes(dst: torch.Tensor, src: torch.Tensor) -> None:
            # cos/sin planes are (3, N, half_rot_dim): the row axis is dim 1,
            # not dim 0 as the manager's default slice-copy assumes.
            dst.zero_()
            dst[:, : src.shape[1]].copy_(src)

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "rotary_cos",
                "rotary_sin",
                "cu_seqlens",
                "max_seqlen",
            ],
            out_hidden_size=self.vision_tower.out_hidden_size,
            padding_logics={
                "cu_seqlens": pad_cu_seqlens,
                "rotary_cos": pad_rope_planes,
                "rotary_sin": pad_rope_planes,
            },
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: "VllmConfig",
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    @staticmethod
    def _get_grid_thw_list(mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid_thw = mm_kwargs["image_grid_thw"]
        if isinstance(grid_thw, torch.Tensor):
            return [[int(x) for x in row] for row in grid_thw.tolist()]
        return [[int(x) for x in row] for row in grid_thw]

    @staticmethod
    def _get_pixel_values(mm_kwargs: dict[str, Any]) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values)
        return pixel_values

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list["EncoderItemSpec"]:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        merge = self.vision_tower.spatial_merge_size
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * h * w // (merge * merge),
            )
            for t, h, w in self._get_grid_thw_list(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thw_list = self._get_grid_thw_list(mm_kwargs)
        pixel_values = self._get_pixel_values(mm_kwargs)
        source_grid = mm_kwargs["image_grid_thw"]

        if len(indices) == 0:
            empty_grid = (
                source_grid[:0]
                if isinstance(source_grid, torch.Tensor)
                else pixel_values.new_zeros((0, 3), dtype=torch.long)
            )
            return {"pixel_values": pixel_values[:0], "image_grid_thw": empty_grid}

        patch_counts = [t * h * w for t, h, w in grid_thw_list]
        cum = [0]
        for pc in patch_counts:
            cum.append(cum[-1] + pc)
        selected_pv = torch.cat(
            [pixel_values[cum[i] : cum[i + 1]] for i in indices], dim=0
        )
        grid_device = (
            source_grid.device if isinstance(source_grid, torch.Tensor) else None
        )
        selected_grid = torch.tensor(
            [grid_thw_list[i] for i in indices],
            dtype=torch.long,
            device=grid_device,
        )
        return {"pixel_values": selected_pv, "image_grid_thw": selected_grid}

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

        merge = self.vision_tower.spatial_merge_size
        # Output tokens per item in the dummy grid (ceiling so total >= budget).
        per_item_out = (token_budget + max_batch_size - 1) // max_batch_size
        grid_thw_list = [
            [1, merge, per_item_out * merge] for _ in range(max_batch_size)
        ]

        embeddings = self.vision_tower.vision_model.embeddings
        patch_dim = (
            embeddings.num_channels
            * embeddings.temporal_patch_size
            * embeddings.patch_size**2
        )
        total_patches = sum(t * h * w for t, h, w in grid_thw_list)
        # Match the tower dtype exactly: MiniMaxVLPatchEmbed.forward mutates
        # the module on dtype mismatch, which must not happen during capture.
        dummy_pixel_values = torch.zeros(
            total_patches,
            patch_dim,
            device=device,
            dtype=self.vision_tower.dtype,
        )

        # The padding sequence can contain nearly all captured rows when a
        # small image replays this graph. Leave one spare cu_seqlens slot.
        metadata = self.vision_tower.vision_model.prepare_encoder_metadata(
            grid_thw_list,
            device=device,
            max_batch_size=max_batch_size + 1,
            max_seqlen_override=total_patches,
        )

        values: dict[str, torch.Tensor] = {"pixel_values": dummy_pixel_values}
        values.update({k: v for k, v in metadata.items() if v is not None})
        self._encoder_cg_pad_totals[values["cu_seqlens"].data_ptr()] = total_patches
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

        pixel_values = self._get_pixel_values(mm_kwargs)
        # Unpadded: pad_cu_seqlens completes the tail with the padding sequence
        # so cu_seqlens[-1] matches the captured buffer's row count.
        metadata = self.vision_tower.vision_model.prepare_encoder_metadata(
            self._get_grid_thw_list(mm_kwargs),
            device=pixel_values.device,
        )
        # FlashAttention reads this CPU scalar during capture, so the captured
        # launch must retain the bound for the padded buffer.
        metadata.pop("max_seqlen")

        values: dict[str, torch.Tensor | None] = {"pixel_values": pixel_values}
        values.update(metadata)
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = inputs.pop("pixel_values")
        # Remaining keys are consumed as encoder_metadata.
        return self.vision_tower(pixel_values, grid_thw=None, encoder_metadata=inputs)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        return self.vision_tower(
            pixel_values=self._get_pixel_values(mm_kwargs).type(
                self.vision_tower.dtype
            ),
            grid_thw=self._get_grid_thw_list(mm_kwargs),
        )
