# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

"""Encoder CUDA graph support for the HYV4 VL vision tower.

Mixin implementing the `SupportsEncoderCudaGraph` protocol
(`vllm/model_executor/models/interfaces.py`) on top of
`HYV4VisionTransformer`. The captured graph packs a whole batch of images
and videos into one varlen ViT run (per-frame attention via `cu_seqlens`,
FLASH_ATTN only) over fixed-size buffers, and the per-item merger
(`HYV4VisionPatchMerger`) runs eagerly on the graph's hidden-state output
in `postprocess_encoder_output`.

Bitwise parity: cudnn's Conv2d (patch embed and merger) is not
invariant to input shape, and cuBLAS GEMMs are only row-count invariant, so
the two convs run eagerly on the real unpadded shapes (patch conv during
replay-buffer preparation, the per-item merger in postprocess) while the
graph covers everything in between. Unused sequence slots repeat the final
real cumulative offset, leaving padded rows outside the attention sequences.

Budgets are in units of *patches* (rows of the packed ViT input). Video item
specs charge `_FRAME_CHARGE` merge cells per frame against the budget so
the packer splits frame-heavy batches: FlashAttn varlen pays a fixed
per-sequence scheduling cost even for zero-length padding sequences
(measured ~0.14us/seq/layer on GB200/FA4), so captured graphs keep their
cu_seqlens slot count minimal (`budget // _FRAME_CHARGE` for video graphs,
`max_batch_size` for image graphs via the modality capture axis).

Enabled via `--compilation-config '{"cudagraph_mm_encoder": true}'`; the
manager (`vllm/v1/worker/encoder_cudagraph.py`) handles budget packing, DP
sharding and eager fallback.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processors.hy_v4 import vision_token_count
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import (
    ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .mm_preprocess import HYV4VLProcessingInfo

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.multimodal import MultiModalConfig
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

    from .vision import HYV4VisionTransformer

# Merge cells (not patches) charged per video frame during packing. Bounds
# the cu_seqlens slots a captured video graph needs (budget /
# (_FRAME_CHARGE * merge**2)): FlashAttn varlen pays a fixed per-sequence
# scheduling cost per layer even for zero-length padding sequences, so a
# max_frames-sized slot array would tax every replay by tens of ms.
_FRAME_CHARGE = 16


class HYV4VLEncoderCudaGraphMixin:
    """`SupportsEncoderCudaGraph` for `HYV4VLForConditionalGeneration`.

    Expects `self.visual`/`self.config`/`self.model_config` from the
    host model.
    """

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    # Provided by the host model; declared here so mixin methods type-check.
    visual: "HYV4VisionTransformer | None"
    config: "HYV4VLConfig"
    model_config: "ModelConfig"
    multimodal_config: "MultiModalConfig | None"

    # -- helpers --

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" in mm_kwargs:
            return "image"
        if "video_grid_thw" in mm_kwargs:
            return "video"
        raise ValueError("HYV4 encoder CUDA graph inputs need a grid_thw key")

    @staticmethod
    def _get_grid_list(mm_kwargs: dict[str, Any], key: str) -> list[list[int]]:
        grid = mm_kwargs[key]
        if isinstance(grid, torch.Tensor):
            grid = grid.tolist()
        return [[int(v) for v in row] for row in grid]

    def _get_pixel_values(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        return mm_kwargs[self._pixel_key(self.get_input_modality(mm_kwargs))]

    @staticmethod
    def _pixel_key(modality: str) -> str:
        return "pixel_values" if modality == "image" else "pixel_values_videos"

    # -- SupportsEncoderCudaGraph protocol --

    def get_max_frames_per_video(self) -> int:
        info = MULTIMODAL_REGISTRY.get_processing_info(self.model_config)
        assert isinstance(info, HYV4VLProcessingInfo)
        return info.get_max_frames_per_video()

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        if self.visual is None:
            raise ValueError(
                "cudagraph_mm_encoder requires the HYV4 vision tower, but it "
                "was stubbed out because both image and video limits are 0. "
                "Disable cudagraph_mm_encoder."
            )
        backend = self.visual.attn_backend
        # FlashInfer reads max_seqlen on the host and TORCH_SDPA calls
        # .tolist() on CUDA cu_seqlens; only FLASH_ATTN is capture-safe here.
        if backend != AttentionBackendEnum.FLASH_ATTN:
            raise ValueError(
                f"cudagraph_mm_encoder is not supported with the "
                f"{backend.name} ViT attention backend for HYV4 VL. Set "
                "--mm-encoder-attn-backend FLASH_ATTN or disable "
                "cudagraph_mm_encoder."
            )

        buffer_keys = ["patches", "pos_embeds"]
        if self.visual.rotary_pos_emb is not None:
            buffer_keys += ["rope_cos", "rope_sin"]
        buffer_keys += ["cu_seqlens", "max_seqlen"]

        # FlashAttn varlen pays a fixed per-sequence scheduling cost even for
        # zero-length padding sequences (~0.14us/seq/layer measured on
        # GB200/FA4), so the frame-slot count baked into a captured graph
        # taxes every replay. Capture separate image and video graphs: image
        # graphs need only one slot per item, video graphs cover multi-frame
        # batches (bounded by the frame charge in the item specs).
        video_enabled = (
            self.multimodal_config is None
            or self.multimodal_config.get_limit_per_prompt("video") > 0
        )
        capture_axes = (("image", "video"),) if video_enabled else ()

        return EncoderCudaGraphConfig(
            modalities=["image", "video"],
            buffer_keys=buffer_keys,
            out_hidden_size=self.config.text_config.hidden_size,
            max_frames_per_video=self.get_max_frames_per_video(),
            capture_axes=capture_axes,
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: "VllmConfig",
    ) -> tuple[int, int]:
        # Budgets are in patches. Min: comfortably above one smallest image
        # (and >= the auto-inferred max_batch_size = 64 via the manager's
        # max_batch_size <= min_budget invariant). Max: one batched-prefill
        # worth of vision tokens, converted to patches.
        vision = self.config.vision_config
        patches_per_token = (
            vision.temporal_patch_size
            * vision.spatial_merge_size**2
            * vision.spatial_patch_size**2
        )
        min_budget = 64
        max_budget = (
            min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            * patches_per_token
        )
        return (min_budget, max_budget)

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        modality = self.get_input_modality(mm_kwargs)
        vision = self.config.vision_config
        eff_merge = vision.spatial_merge_size * vision.spatial_patch_size
        specs = []
        for grid in self._get_grid_list(mm_kwargs, f"{modality}_grid_thw"):
            t, h, w = grid
            patches = t * h * w
            path_tokens = patches
            if modality == "video":
                # Charge frames against the budget so the packer splits
                # frame-heavy batches; captured video graphs carry
                # budget // (_FRAME_CHARGE * eff_merge**2) cu_seqlens slots.
                path_tokens = max(patches, _FRAME_CHARGE * eff_merge * eff_merge * t)
            specs.append(
                EncoderItemSpec(
                    input_size=patches,
                    output_tokens=vision_token_count(grid, vision, modality),
                    path_output_tokens={"default": path_tokens},
                )
            )
        return specs

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        modality = self.get_input_modality(mm_kwargs)
        grid_key = f"{modality}_grid_thw"
        grids = self._get_grid_list(mm_kwargs, grid_key)
        pixel_values = self._get_pixel_values(mm_kwargs)

        if len(indices) == 0:
            return {
                self._pixel_key(modality): pixel_values[:0],
                grid_key: torch.zeros((0, 3), dtype=torch.int64),
                ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG: (modality,),
            }

        cum_patches = [0]
        for t, h, w in grids:
            cum_patches.append(cum_patches[-1] + t * h * w)

        return {
            self._pixel_key(modality): torch.cat(
                [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices]
            ),
            grid_key: torch.tensor([grids[i] for i in indices], dtype=torch.int64),
            ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG: (modality,),
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
    ) -> EncoderCudaGraphCaptureInputs:
        if self.visual is None:
            raise RuntimeError(
                "cudagraph_mm_encoder requires the HYV4 vision tower, but it "
                "was stubbed out because the image/video limits are 0. "
                "Disable cudagraph_mm_encoder."
            )

        visual = self.visual
        vision = self.config.vision_config
        eff_merge = vision.spatial_merge_size * vision.spatial_patch_size
        patches_per_cell = eff_merge**2

        # Replay buffers are global and flat: the dummy grids only fix buffer
        # *sizes*, their per-item layout is irrelevant at replay. Size them
        # tightly against the worst-case replay batch of `token_budget`
        # patches and at most `max_batch_size` items, with every dummy frame
        # grid small enough for the position-embedding / 2D-RoPE tables
        # (both sized position_edge).
        side = visual.embeddings.position_edge // eff_merge
        if side < 1:
            raise ValueError(
                f"HYV4 vision position table (edge "
                f"{visual.embeddings.position_edge}) is smaller than one "
                f"merge cell ({eff_merge}); cannot capture encoder CUDA graphs."
            )
        patches_per_frame_cap = side * side * patches_per_cell

        axis = axis_keys[0] if axis_keys else None
        if axis == "image" or (axis is None and max_frames_per_batch == 0):
            # One frame per dummy item; add dummy items until each stays
            # within the position-table capacity.
            n_items = max(max_batch_size, -(-token_budget // patches_per_frame_cap))
            t_per_item = [1] * n_items
            cells = -(-token_budget // (n_items * patches_per_cell))
        else:
            # Video-capable graph (the default when no capture axes are set).
            # The frame charge in the item specs bounds replay frames to
            # budget // (_FRAME_CHARGE * patches_per_cell).
            frames = max(
                max_batch_size,
                -(-token_budget // (_FRAME_CHARGE * patches_per_cell)),
                -(-token_budget // patches_per_frame_cap),
            )
            cells = -(-token_budget // (frames * patches_per_cell))
            base, rem = divmod(frames, max_batch_size)
            t_per_item = [base + (1 if i < rem else 0) for i in range(max_batch_size)]

        # Split cells into rows/cols that fit the position/RoPE tables.
        if cells <= side:
            rows, cols = 1, cells
        else:
            cols = side
            rows = -(-cells // side)
            assert rows <= side, "cells is capped by side**2 by construction"

        grids = [[t, rows * eff_merge, cols * eff_merge] for t in t_per_item]
        total_patches = sum(t_per_item) * rows * cols * patches_per_cell

        # max_seqlen is baked into the graph at capture time; cover the worst
        # replay case (one item's frame filling the whole budget) as well as
        # the dummy frame length.
        max_seqlen_override = max(token_budget, cells * patches_per_cell)
        metadata = visual.prepare_encoder_cudagraph_metadata(
            grids,
            device=device,
            max_seqlen_override=max_seqlen_override,
        )

        # The buffer holds *embedded* patches (input norm + patch conv are
        # applied eagerly at replay time, because cudnn conv results are not
        # invariant to input shape and would break bitwise parity).
        patches = torch.zeros(
            total_patches,
            vision.hidden_size,
            device=device,
            dtype=dtype,
        )

        values: dict[str, torch.Tensor] = {"patches": patches}
        values.update({k: v for k, v in metadata.items() if v is not None})
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        modality = self.get_input_modality(mm_kwargs)
        grids = self._get_grid_list(mm_kwargs, f"{modality}_grid_thw")
        pixel_values = self._get_pixel_values(mm_kwargs)

        assert self.visual is not None
        # Run the fused input norm and patch conv eagerly, replicating the
        # eager call exactly (cudnn conv is not batch-shape invariant, so the
        # conv must see the real unpadded batch to stay bitwise-identical).
        patches = self.visual.embeddings._patch_conv(
            self.visual.input_norm(pixel_values, self.visual.dtype)
        ).squeeze(0)
        # Repeating the final cumulative offset leaves padding rows outside
        # every sequence, as in the other packed vision encoders.
        metadata = self.visual.prepare_encoder_cudagraph_metadata(
            grids,
            device=pixel_values.device,
        )
        metadata.pop("max_seqlen")
        values: dict[str, torch.Tensor | None] = {"patches": patches}
        values.update(metadata)
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        assert self.visual is not None
        return self.visual.forward_packed(
            inputs["patches"],
            inputs["pos_embeds"],
            inputs.get("rope_cos"),
            inputs.get("rope_sin"),
            inputs["cu_seqlens"],
            inputs["max_seqlen"],
        ).squeeze(0)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        modality = self.get_input_modality(mm_kwargs)
        grids = self._get_grid_list(mm_kwargs, f"{modality}_grid_thw")
        pixel_values = self._get_pixel_values(mm_kwargs)
        assert self.visual is not None
        if not grids:
            return torch.zeros(
                (1, 0, self.config.vision_config.hidden_size),
                device=self.visual.device,
                dtype=self.visual.dtype,
            )
        return self.visual._encode(pixel_values, grids)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not indices:
            return
        assert batch_mm_kwargs is not None
        assert self.visual is not None
        hidden = outputs["default"]
        modality = self.get_input_modality(batch_mm_kwargs)
        grids = self._get_grid_list(batch_mm_kwargs, f"{modality}_grid_thw")
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # The merger runs eagerly on the original per-item shapes (cudnn's
        # merge Conv2d is not shape-invariant, so this is what keeps the
        # outputs bitwise identical to the eager path). The reads of the
        # graph's output buffer happen here, synchronously, before the
        # manager may replay the graph again; every output row is freshly
        # allocated by the merger.
        for idx, states in zip(indices, self.visual._iter_item_states(hidden, grids)):
            dest[idx] = self.visual.perceive(states)
