# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph support for the MiniMax M3 vision tower.

Mixin implementing the SupportsEncoderCudaGraph protocol for any host class
that holds the vision tower as `self.vision_tower` (a
`MiniMaxVLVisionModel`). The manager routes image items through this mixin;
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
    """SupportsEncoderCudaGraph for hosts with `self.vision_tower`."""

    vision_tower: "MiniMaxVLVisionModel"
    multimodal_config: "MultiModalConfig | None"

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

        def pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
            # Zero-length padding (same as Qwen): tail slots repeat the last
            # real offset, so FlashAttn schedules no work for padding rows.
            n = src.shape[0]
            dst[:n].copy_(src)
            dst[n:] = src[-1]

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

        metadata = self.vision_tower.vision_model.prepare_encoder_metadata(
            grid_thw_list,
            device=device,
            max_batch_size=max_batch_size,
            max_seqlen_override=total_patches,
        )

        values: dict[str, torch.Tensor] = {"pixel_values": dummy_pixel_values}
        values.update({k: v for k, v in metadata.items() if v is not None})
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
        # Unpadded: the manager zero-pads the row buffers and pad_cu_seqlens
        # leaves the padding rows outside every sequence.
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

    # ── Sequence-parallel ViT (opt-in, with --mm-encoder-tp-mode data) ────
    #
    # Data-parallel mode encodes each image whole on one rank, so a step with
    # one image keeps the other TP ranks waiting for it. With
    # VLLM_MINIMAX_M3_VIT_SP=1, a batch of at most VLLM_MINIMAX_M3_VIT_SP_MAX_ITEMS
    # images with at least VLLM_MINIMAX_M3_VIT_SP_MIN_PATCHES patches is instead
    # split by patch rows across the TP ranks: every rank runs its rows through
    # the whole tower and all-gathers the attention input once per layer (bf16,
    # lossless). One CUDA graph per per-rank row bucket
    # (VLLM_MINIMAX_M3_VIT_SP_ROWS); batches that fit no bucket keep the
    # data-parallel path.

    @staticmethod
    def _encoder_sp_settings() -> dict[str, Any] | None:
        import vllm.envs as envs

        if not envs.VLLM_MINIMAX_M3_VIT_SP:
            return None
        rows = envs.VLLM_MINIMAX_M3_VIT_SP_ROWS
        return {
            "rows": [int(r) for r in rows.split(",") if r.strip()],
            "max_items": envs.VLLM_MINIMAX_M3_VIT_SP_MAX_ITEMS,
            "min_patches": envs.VLLM_MINIMAX_M3_VIT_SP_MIN_PATCHES,
        }

    def get_encoder_sp_rows(self) -> list[int]:
        from vllm.distributed import get_tensor_model_parallel_world_size

        settings = self._encoder_sp_settings()
        if settings is None or get_tensor_model_parallel_world_size() == 1:
            return []
        group = self.vision_tower.spatial_merge_size**2
        return sorted({r for r in settings["rows"] if r > 0 and r % group == 0})

    def prepare_encoder_sp_buffers(
        self, rows: int, device: torch.device
    ) -> dict[str, Any]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLSPMeta

        settings = self._encoder_sp_settings()
        assert settings is not None
        tp = get_tensor_model_parallel_world_size()
        vm = self.vision_tower.vision_model
        emb = vm.embeddings
        patch_dim = emb.num_channels * emb.temporal_patch_size * emb.patch_size**2
        merge = vm.spatial_merge_size
        cos0, _ = vm.mrope_setup(
            vm.inv_freq_t, vm.inv_freq_h, vm.inv_freq_w, [[1, merge, merge]], merge
        )
        planes = (cos0.shape[0], tp * rows, cos0.shape[2])
        # A rank's rows can touch one more image than the batch holds whole.
        segments = max(settings["max_items"], 1) + 1
        sp = MiniMaxVLSPMeta(
            rows=rows,
            q_start=get_tensor_model_parallel_rank() * rows,
            cu_q=torch.zeros(segments + 1, dtype=torch.int32, device=device),
            cu_k=torch.zeros(segments + 1, dtype=torch.int32, device=device),
            max_q=rows,
            max_k=tp * rows,
        )
        return {
            "pixel_values": torch.zeros(
                rows, patch_dim, device=device, dtype=self.vision_tower.dtype
            ),
            "rotary_cos": torch.zeros(planes, dtype=cos0.dtype, device=device),
            "rotary_sin": torch.zeros(planes, dtype=cos0.dtype, device=device),
            "sp": sp,
        }

    def encoder_sp_forward(self, buffers: dict[str, Any]) -> torch.Tensor:
        return self.vision_tower.forward_sp(
            buffers["pixel_values"],
            buffers["rotary_cos"],
            buffers["rotary_sin"],
            buffers["sp"],
        )

    def encoder_sp_execute(
        self, mm_kwargs: dict[str, Any], sp_graphs: dict[int, Any]
    ) -> list[torch.Tensor] | None:
        """Replay the sequence-parallel graph for this batch, or return None."""
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )

        settings = self._encoder_sp_settings()
        if settings is None or not sp_graphs:
            return None
        grids = self._get_grid_thw_list(mm_kwargs)
        if not grids or len(grids) > settings["max_items"]:
            return None
        vm = self.vision_tower.vision_model
        limited = vm._apply_max_frames_limit(grids)
        lens = [t * h * w for t, h, w in limited]
        total = sum(lens)
        if total < settings["min_patches"]:
            return None
        tp = get_tensor_model_parallel_world_size()
        rows = min((r for r in sp_graphs if r * tp >= total), default=None)
        if rows is None:
            return None
        graph, buffers, output = sp_graphs[rows]
        sp = buffers["sp"]

        # Every decision above uses only the batch's grids, so all ranks agree.
        rank = get_tensor_model_parallel_rank()
        start = min(rank * rows, total)
        end = min(start + rows, total)
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        cu_q, cu_k = [0], []
        for i in range(len(lens)):
            lo, hi = max(cu[i], start), min(cu[i + 1], end)
            if hi > lo:
                if not cu_k:
                    cu_k.append(cu[i])
                cu_q.append(cu_q[-1] + hi - lo)
                cu_k.append(cu[i + 1])
        if not cu_k:
            cu_k = [total]
        slots = sp.cu_q.shape[0]
        cu_q += [cu_q[-1]] * (slots - len(cu_q))
        cu_k += [cu_k[-1]] * (slots - len(cu_k))
        device = sp.cu_q.device
        sp.cu_q.copy_(torch.tensor(cu_q, dtype=torch.int32), non_blocking=True)
        sp.cu_k.copy_(torch.tensor(cu_k, dtype=torch.int32), non_blocking=True)

        pixel_values = self._get_pixel_values(mm_kwargs)
        local = end - start
        if local > 0:
            buffers["pixel_values"][:local].copy_(pixel_values[start:end])
        if local < rows:
            buffers["pixel_values"][local:].zero_()
        cos, sin = vm.mrope_setup(
            vm.inv_freq_t, vm.inv_freq_h, vm.inv_freq_w, limited, vm.spatial_merge_size
        )
        buffers["rotary_cos"][:, :total].copy_(cos.to(device))
        buffers["rotary_sin"][:, :total].copy_(sin.to(device))

        graph.replay()

        merged = tensor_model_parallel_all_gather(output, dim=0)
        group = vm.spatial_merge_size**2
        sizes = [t * h * w // group for t, h, w in grids]
        return list(merged[: total // group].split(sizes))

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
