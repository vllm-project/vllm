# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph support for the DeepSeek-V4.1 vision tower.

Mixin implementing the ``SupportsEncoderCudaGraph`` protocol
(``vllm/model_executor/models/interfaces.py``) on top of the shared
``DeepseekV4ViT``/``DeepseekV4Aligner``. The captured graph packs a whole
batch of images into one varlen run: ViT blocks attend per image via
``cu_seqlens``, and the aligner's spatial merge becomes a gather+mask
against precomputed indices. Span assembly (IMAGE_START/IMAGE_NEW_LINE/
IMAGE_END delimiters) stays eager in ``postprocess_encoder_output`` but is
batched over all items of the packed batch.

Enabled via ``-O.cudagraph_mm_encoder=true``; the manager
(``vllm/v1/worker/encoder_cudagraph.py``) handles budget packing, DP
sharding and eager fallback.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.deepseek_v4.common.vision import (
    build_packed_merge_metadata,
    build_packed_vit_metadata,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .mm_preprocess import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.multimodal import MultiModalConfig
    from vllm.models.deepseek_v4.common.vision import (
        DeepseekV4Aligner,
        DeepseekV4ViT,
    )


class DeepseekV4VLEncoderCudaGraphMixin:
    """``SupportsEncoderCudaGraph`` for ``DeepseekV41ForCausalLM``.

    Expects ``self.vision``/``self.aligner``/``self.config``/
    ``self.multimodal_config`` and the ``image_start``/``image_newline``/
    ``image_end`` parameters from the host model.
    """

    if TYPE_CHECKING:
        vision: DeepseekV4ViT
        aligner: DeepseekV4Aligner
        config: Any
        multimodal_config: "MultiModalConfig | None"
        image_start: torch.nn.Parameter
        image_end: torch.nn.Parameter
        image_newline: torch.nn.Parameter

    # -- helpers shared with the eager multimodal path --

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        # Aligner rows in reading order, one per IMAGE slot.
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def _build_image_span(
        self, image_embeds: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """Full image span: aligner rows at IMAGE slots, the learned
        delimiter vectors at IMAGE_START/IMAGE_NEW_LINE/IMAGE_END."""
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = image_embeds
        return span

    @staticmethod
    def _get_grid_list(mm_kwargs: dict[str, Any], key: str) -> list[list[int]]:
        grid = mm_kwargs[key]
        if isinstance(grid, torch.Tensor):
            grid = grid.tolist()
        return [[int(v) for v in row] for row in grid]

    @property
    def _encoder_cg_pad_totals(self) -> dict[int, int]:
        """Row count of each captured buffer set, keyed by cu_seqlens ptr."""
        totals = self.__dict__.get("_encoder_cg_pad_totals")
        if totals is None:
            totals = {}
            self.__dict__["_encoder_cg_pad_totals"] = totals
        return totals

    # -- SupportsEncoderCudaGraph protocol --

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        mm_cfg = self.multimodal_config
        if (
            mm_cfg is not None
            and mm_cfg.mm_encoder_attn_dtype == "fp8"
            and mm_cfg.mm_encoder_fp8_scale_path is None
        ):
            raise ValueError(
                "cudagraph_mm_encoder is incompatible with dynamic FP8 "
                "scaling for the vision encoder (the amax history is "
                "host-side state that cannot be captured). Provide a static "
                "scale file via --mm-encoder-fp8-scale-path or disable "
                "cudagraph_mm_encoder."
            )

        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        backend = self.vision.blocks[0].attn.attn.attn_backend
        if backend == AttentionBackendEnum.FLASHINFER:
            raise ValueError(
                "cudagraph_mm_encoder is not supported with the FlashInfer "
                "ViT attention backend for DeepSeek-V4.1. Set "
                "VLLM_VIT_ATTN_BACKEND=FLASH_ATTN or disable "
                "cudagraph_mm_encoder."
            )

        pad_totals = self._encoder_cg_pad_totals

        def pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
            # Varlen attention requires cu_seqlens[-1] to equal the number of
            # rows actually passed in. The captured buffers are sized for the
            # full token budget, so a smaller real batch is completed with one
            # trailing padding sequence; declaring fewer rows than the buffer
            # holds is undefined behaviour and returns NaN on FlashAttn.
            total = pad_totals.get(dst.data_ptr())
            n = min(src.shape[0], dst.shape[0])
            dst[:n].copy_(src[:n])
            dst[n:] = total if total is not None else src[-1]

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "patches",
                "vit_cos",
                "vit_sin",
                "cu_seqlens",
                "max_seqlen",
                "merge_idx",
                "merge_mask",
            ],
            out_hidden_size=self.config.hidden_size,
            padding_logics={"cu_seqlens": pad_cu_seqlens},
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: "VllmConfig",
    ) -> tuple[int, int]:
        # Min: comfortably above the smallest possible image span (4 tokens
        # for a 1x1 grid); also sets max_batch_size = 64 via the manager's
        # auto-inference (max_batch_size <= min_budget invariant).
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        vit_grid = self._get_grid_list(mm_kwargs, "vit_grid")
        llm_grid = self._get_grid_list(mm_kwargs, "llm_grid")
        return [
            EncoderItemSpec(
                input_size=h * w,
                output_tokens=lh * (lw + 1) + 2,
            )
            for (h, w), (lh, lw) in zip(vit_grid, llm_grid, strict=True)
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        vit_grid = self._get_grid_list(mm_kwargs, "vit_grid")
        llm_grid = self._get_grid_list(mm_kwargs, "llm_grid")
        patches = mm_kwargs["patches"]
        types = mm_kwargs["types"]

        if len(indices) == 0:
            return {
                "patches": patches[:0],
                "vit_grid": torch.zeros((0, 2), dtype=torch.int64),
                "llm_grid": torch.zeros((0, 2), dtype=torch.int64),
                "types": types[:0],
            }

        cum_patches = [0]
        for h, w in vit_grid:
            cum_patches.append(cum_patches[-1] + h * w)
        cum_spans = [0]
        for lh, lw in llm_grid:
            cum_spans.append(cum_spans[-1] + lh * (lw + 1) + 2)

        return {
            "patches": torch.cat(
                [patches[cum_patches[i] : cum_patches[i + 1]] for i in indices]
            ),
            "vit_grid": torch.tensor([vit_grid[i] for i in indices], dtype=torch.int64),
            "llm_grid": torch.tensor([llm_grid[i] for i in indices], dtype=torch.int64),
            "types": torch.cat(
                [types[cum_spans[i] : cum_spans[i + 1]] for i in indices]
            ),
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
        if (
            self.multimodal_config is not None
            and self.multimodal_config.get_limit_per_prompt("image") == 0
        ):
            raise RuntimeError(
                "cudagraph_mm_encoder requires the vision tower, but it was "
                "stubbed out because the image limit is 0. Disable "
                "cudagraph_mm_encoder."
            )

        config = self.config
        r = config.vision_downsample_ratio
        # Span rows per dummy item (ceiling so the buffers tile the budget).
        # Each dummy image is a single aligner row of per_item_out blocks:
        # r x (r * per_item_out) patches. One real image using the whole
        # budget needs at most r^2 * token_budget patches, which this fits.
        per_item_out = -(-token_budget // max_batch_size)
        grids = [[r, r * per_item_out]] * max_batch_size
        total_patches = r * r * per_item_out * max_batch_size

        p = config.vision_patch_size
        patches = torch.zeros(total_patches, 3, p, p, device=device, dtype=dtype)

        metadata = build_packed_vit_metadata(
            grids,
            rope_dim=self.vision.rope_dim,
            rope_theta=self.vision.rope_theta,
            device=device,
            max_seqlen_override=total_patches,
            cached=False,
        )
        # Spare cu_seqlens slots let replay append a padding sequence that
        # covers the rows a smaller real batch does not fill.
        real_cu = metadata.pop("cu_seqlens")
        cu_seqlens = torch.full(
            (max_batch_size + 2,), total_patches, dtype=torch.int32, device=device
        )
        cu_seqlens[: real_cu.numel()] = real_cu

        merge = build_packed_merge_metadata(grids, r, device=device, dtype=dtype)

        self._encoder_cg_pad_totals[cu_seqlens.data_ptr()] = total_patches

        return EncoderCudaGraphCaptureInputs(
            values={
                "patches": patches,
                **metadata,
                "cu_seqlens": cu_seqlens,
                **merge,
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        vit_grid = self._get_grid_list(mm_kwargs, "vit_grid")
        patches = mm_kwargs["patches"]
        dtype = self.aligner.w1.weight.dtype
        if patches.dtype != dtype:
            patches = patches.to(dtype)

        # Unpadded: the manager zero-pads patches/cos/sin/merge buffers, and
        # pad_cu_seqlens appends the padding sequence covering the tail rows.
        metadata = build_packed_vit_metadata(
            vit_grid,
            rope_dim=self.vision.rope_dim,
            rope_theta=self.vision.rope_theta,
            device=patches.device,
        )
        merge = build_packed_merge_metadata(
            vit_grid,
            self.config.vision_downsample_ratio,
            device=patches.device,
            dtype=dtype,
        )
        return EncoderCudaGraphReplayBuffers(
            values={"patches": patches, **metadata, **merge}
        )

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        vit_out = self.vision.forward_packed(
            inputs["patches"],
            inputs["vit_cos"],
            inputs["vit_sin"],
            inputs["cu_seqlens"],
            inputs["max_seqlen"],
        )
        return self.aligner.forward_packed(
            vit_out, inputs["merge_idx"], inputs["merge_mask"]
        )

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        patches = mm_kwargs["patches"].to(self.aligner.w1.weight.dtype)
        vit_grid = self._get_grid_list(mm_kwargs, "vit_grid")
        outs: list[torch.Tensor] = []
        offset = 0
        for n_vit_h, n_vit_w in vit_grid:
            n_vit = n_vit_h * n_vit_w
            outs.append(
                self._encode_image(patches[offset : offset + n_vit], n_vit_h, n_vit_w)
            )
            offset += n_vit
        if not outs:
            return patches.new_zeros((0, self.config.hidden_size))
        return torch.cat(outs, dim=0)

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
        aligner_out = outputs["default"]
        r = self.config.vision_downsample_ratio
        vit_grid = self._get_grid_list(batch_mm_kwargs, "vit_grid")
        llm_grid = self._get_grid_list(batch_mm_kwargs, "llm_grid")
        types = batch_mm_kwargs["types"].to(aligner_out.device)

        n_rows = sum(-(-h // r) * (-(-w // r)) for h, w in vit_grid)
        span_lens = [lh * (lw + 1) + 2 for lh, lw in llm_grid]

        # Batched span assembly: one masked fill per role across all items.
        dtype = aligner_out.dtype
        span = aligner_out.new_empty(sum(span_lens), self.config.hidden_size)
        span[types == IMAGE] = aligner_out[:n_rows]
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)

        # Freshly allocated, so later replays cannot clobber the results.
        offset = 0
        for idx, span_len in zip(indices, span_lens):
            dest[idx] = span[offset : offset + span_len]
            offset += span_len
