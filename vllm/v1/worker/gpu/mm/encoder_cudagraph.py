"""CUDA graph manager for vision encoder budget-batch execution."""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Any

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _count_input_patches(grid_thw_list: list[list[int]]) -> int:
    """Count total input patches (T*H*W per image). Used for pixel_values slicing."""
    return sum(t * h * w for t, h, w in grid_thw_list)


def _count_output_tokens(
    grid_thw_list: list[list[int]], spatial_merge_size: int
) -> int:
    """Count total output tokens after spatial merging. Used for budget selection."""
    m = spatial_merge_size
    return sum(t * (h // m) * (w // m) for t, h, w in grid_thw_list)


@dataclass
class BudgetGraphMetadata:
    """Metadata for a single budget graph.

    CUDA graph replay pattern:
    1. Copy new batch data into input_buffers (pixel_values)
    2. Compute batch-specific tensors and copy into metadata_buffers:
       - Sequence metadata: cu_seqlens, sequence_lengths, max_seqlen
       - Position embeddings: pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin
    3. Replay graph
    4. Read encoder outputs from output_buffer
    """
    token_budget: int
    max_batch_size: int  # Max number of images/videos per batch
    graph: torch.cuda.CUDAGraph
    # Raw inputs updated before replay
    input_buffers: dict[str, Any]
    # Batch-specific tensors (sequence metadata + position embeddings) updated before replay
    metadata_buffers: dict[str, torch.Tensor]
    # Output written by graph, read after replay
    output_buffer: torch.Tensor


class EncoderCudaGraphManager:
    """Budget-based CUDA graph capture/replay for vision encoders."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        vision_model: torch.nn.Module,
    ):
        """Initialize CUDA graph manager with provided token budgets and max batch size."""
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype
        self.vision_model = vision_model

        comp_config = vllm_config.compilation_config
        self.token_budgets = sorted(comp_config.encoder_cudagraph_token_budgets)
        self.max_batch_size = comp_config.encoder_cudagraph_max_images_per_batch

        self.use_dp = (
            vllm_config.model_config.multimodal_config.mm_encoder_tp_mode == "data"
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )

        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}
        self.graph_hits = 0
        self.graph_misses = 0
        self.log_stats_interval = 100

        logger.info(
            f"EncoderCudaGraphManager initialized with budgets={self.token_budgets}, "
            f"max_batch_size={self.max_batch_size}, use_dp={self.use_dp}"
        )

    def capture(self):
        """Capture CUDA graphs for all token budgets."""
        for token_budget in self.token_budgets:
            self._capture_budget_graph(token_budget)

        logger.info(
            f"Encoder CUDA graph capture complete. "
            f"Captured {len(self.budget_graphs)} budget graphs."
        )

    def _capture_budget_graph(self, token_budget: int):
        """Capture CUDA graph for a single token budget."""
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d",
            token_budget, self.max_batch_size
        )
        # Generate dummy grid config for capture only (not used for runtime batching).
        # This is just one arbitrary example configuration that produces token_budget tokens.
        # At runtime, actual images will be packed in any combination that fits the budget.
        dummy_grid_config = self._generate_grid_config_for_budget(
            token_budget, self.max_batch_size
        )

        dummy_pixel_values, dummy_grid_thw = self._prepare_dummy_inputs(dummy_grid_config)

        encoder_metadata = {}
        encoder_metadata['pos_embeds'] = self.vision_model.fast_pos_embed_interpolate(
            dummy_grid_thw
        )
        rotary_cos, rotary_sin = self.vision_model.rot_pos_emb(dummy_grid_thw)
        encoder_metadata['rotary_pos_emb_cos'] = rotary_cos
        encoder_metadata['rotary_pos_emb_sin'] = rotary_sin

        from vllm.model_executor.models.vision import compute_encoder_metadata
        seq_metadata = compute_encoder_metadata(
            dummy_grid_thw,
            device=self.device,
            pad_to_batch_size=None,
            per_frame=True,
        )
        encoder_metadata['cu_seqlens'] = seq_metadata['cu_seqlens']
        # Override max_seqlen with a safe upper bound for capture.
        # max_seqlen.item() gets baked into the CUDA graph (not replayed),
        # so the capture value must cover any replay scenario.
        # Worst case: 1 image consuming the full budget →
        # seq_len = token_budget * spatial_merge_size^2.
        spatial_merge_size = self.vision_model.spatial_merge_size
        max_seqlen_safe = token_budget * (spatial_merge_size ** 2)
        encoder_metadata['max_seqlen'] = torch.tensor(
            max_seqlen_safe, dtype=torch.int32)

        with torch.inference_mode():
            output = self.vision_model(
                dummy_pixel_values,
                dummy_grid_thw,
                encoder_metadata=encoder_metadata
            )
            output_buffer = torch.empty_like(output)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                output = self.vision_model(
                    dummy_pixel_values,
                    dummy_grid_thw,
                    encoder_metadata=encoder_metadata
                )
                output_buffer.copy_(output)

        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            graph=graph,
            input_buffers={
                'pixel_values': dummy_pixel_values,
                'grid_thw': dummy_grid_thw,
            },
            output_buffer=output_buffer,
            metadata_buffers=encoder_metadata,
        )

    def _generate_grid_config_for_budget(
        self, token_budget: int, max_batch_size: int
    ) -> list[list[int]]:
        """Generate dummy grid configuration for CUDA graph capture.

        This creates an arbitrary example that produces tokens matching the given budget.
        NOT used for runtime batching decisions - only for generating dummy inputs.

        Uses rectangular grids [1, merge, per_image_output * merge] for exact budget match.
        """
        spatial_merge_size = self.vision_model.spatial_merge_size
        per_image_output = token_budget // max_batch_size

        # Synthetic rectangular grid: [1, merge, per_image_output * merge]
        # This produces exactly per_image_output tokens per image:
        #   output_tokens = T * (H/merge) * (W/merge)
        #                 = 1 * (merge/merge) * (per_image_output*merge/merge)
        #                 = per_image_output
        # Total output = max_batch_size * per_image_output = token_budget
        grid_config = [[1, spatial_merge_size, per_image_output * spatial_merge_size]
                       for _ in range(max_batch_size)]

        return grid_config

    def _prepare_dummy_inputs(
        self, grid_config: list[list[int]]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Create dummy pixel_values and grid_thw for capture."""
        # Compute total patches from grid config
        total_patches = _count_input_patches(grid_config)

        # Get patch dimensions from vision model's patch_embed
        patch_embed = self.vision_model.patch_embed
        in_channels = patch_embed.proj.in_channels
        patch_size = patch_embed.patch_size
        temporal_patch_size = patch_embed.temporal_patch_size

        # PatchEmbed expects shape (total_patches, flattened_patch_size)
        flattened_patch_size = in_channels * temporal_patch_size * patch_size * patch_size
        dummy_pixel_values = torch.randn(
            total_patches, flattened_patch_size,
            device=self.device,
            dtype=self.dtype,
        )

        return dummy_pixel_values, grid_config

    def _find_smallest_fitting_budget_given_tokens(self, total_tokens: int) -> Optional[int]:
        """Find smallest budget >= total_tokens.

        Returns:
            Token budget if found, None if no fitting budget.
        """
        for budget in self.token_budgets:
            if budget >= total_tokens:
                return budget

    def _run_budget_graph(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list[list[int]],
        token_budget: int,
        encoder_metadata: dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Execute budget graph.

        Args:
            pixel_values: Concatenated pixel values
            grid_thw_list: Grid dimensions per image
            token_budget: Token budget to use
            encoder_metadata: Pre-computed metadata (cu_seqlens, pos_embeds, rotary)

        Returns:
            Encoder outputs, or None if graph not captured.
        """
        num_images = len(grid_thw_list)
        if token_budget not in self.budget_graphs:
            self.graph_misses += num_images
            return None

        graph_meta = self.budget_graphs[token_budget]

        # Buffers are sized for the full budget; actual inputs may be smaller.
        # Zero then slice-copy so padded positions are invisible to attention
        # (cu_seqlens masks them out).
        buf = graph_meta.input_buffers['pixel_values']
        n = pixel_values.shape[0]
        buf.zero_()
        buf[:n].copy_(pixel_values)

        for key in ['pos_embeds', 'rotary_pos_emb_cos', 'rotary_pos_emb_sin']:
            buf = graph_meta.metadata_buffers[key]
            src = encoder_metadata[key]
            n = src.shape[0]
            buf.zero_()
            buf[:n].copy_(src)

        # cu_seqlens: pad tail with last value → zero-length seqs for empty slots
        buf = graph_meta.metadata_buffers['cu_seqlens']
        src = encoder_metadata['cu_seqlens']
        n = src.shape[0]
        buf[:n].copy_(src)
        buf[n:].fill_(src[-1])

        graph_meta.metadata_buffers['max_seqlen'].copy_(
            encoder_metadata['max_seqlen'])

        graph_meta.graph.replay()

        self.graph_hits += num_images
        return graph_meta.output_buffer

    def _execute_local(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list[list[int]],
    ) -> list[torch.Tensor] | None:
        """Execute encoder on local inputs using greedy-packed CUDA graphs.

        Sort images by output token count (smallest first), then greedily pack
        as many images as possible into each batch while staying within
        max_budget tokens and max_batch_size. Once a batch is finalised (next
        image would overflow either constraint), find the smallest fitting
        budget once for that batch.

        By exchange argument, greedy smallest-first packing minimises eager
        fallbacks — any other ordering yields a higher token sum in some batch,
        making that batch more likely to exceed the budget.

        Stats note:
          graph_hits  — counted inside _run_budget_graph after successful replay.
          graph_misses — counted here for single-image batches where the image
                         exceeds max_budget. Batches split due to max_batch_size
                         always satisfy total_tokens <= max_budget and therefore
                         always find a valid budget (no miss).
        """
        from vllm.model_executor.models.vision import compute_encoder_metadata

        spatial_merge = self.vision_model.spatial_merge_size
        num_images = len(grid_thw_list)
        max_budget = self.token_budgets[-1]

        # Per-image output token counts (for sorting and output slicing)
        per_image_out_tokens = [
            _count_output_tokens([grid], spatial_merge)
            for grid in grid_thw_list
        ]

        # Cumulative patch offsets in original order (for pixel_values slicing)
        patch_offsets = [0] * (num_images + 1)
        for image_idx in range(num_images):
            patch_offsets[image_idx + 1] = (
                patch_offsets[image_idx]
                + _count_input_patches([grid_thw_list[image_idx]])
            )

        # Sort ascending by output token count (smallest first)
        sorted_indices = sorted(
            range(num_images), key=lambda i: per_image_out_tokens[i]
        )

        # Greedy pack against max_budget and max_batch_size.
        # _find_smallest_fitting_budget_given_tokens is called once per
        # finalised batch, not per image.
        batches: list[tuple[list[int], int | None]] = []
        current_batch: list[int] = []
        current_batch_tokens = 0

        for orig_idx in sorted_indices:
            image_tokens = per_image_out_tokens[orig_idx]
            if (current_batch_tokens + image_tokens <= max_budget
                    and len(current_batch) < self.max_batch_size):
                current_batch.append(orig_idx)
                current_batch_tokens += image_tokens
            else:
                if current_batch:
                    batches.append((
                        current_batch,
                        self._find_smallest_fitting_budget_given_tokens(
                            current_batch_tokens
                        ),
                    ))
                current_batch = [orig_idx]
                current_batch_tokens = image_tokens

        if current_batch:
            batches.append((
                current_batch,
                self._find_smallest_fitting_budget_given_tokens(
                    current_batch_tokens
                ),
            ))

        # outputs_by_orig_idx maps each original image index to its output
        # tensor. Needed because greedy packing reorders images; we restore
        # the original order before returning.
        outputs_by_orig_idx: dict[int, torch.Tensor] = {}

        for batch_orig_indices, token_budget in batches:
            batch_grid = [grid_thw_list[i] for i in batch_orig_indices]
            batch_out_tokens = _count_output_tokens(batch_grid, spatial_merge)

            batch_pixel_values = torch.cat([
                pixel_values[patch_offsets[i]:patch_offsets[i + 1]]
                for i in batch_orig_indices
            ])

            if token_budget is None:
                # Single oversized image: image_tokens > max_budget.
                # graph_misses counted here for this eager fallback.
                logger.debug(
                    "Encoder CUDA graph fallback to eager: no budget for "
                    "%d tokens from %d images",
                    batch_out_tokens, len(batch_orig_indices),
                )
                self.graph_misses += len(batch_orig_indices)
                with torch.inference_mode():
                    raw = self.vision_model(batch_pixel_values, batch_grid)
                output_offset = 0
                for orig_idx in batch_orig_indices:
                    n_tok = per_image_out_tokens[orig_idx]
                    outputs_by_orig_idx[orig_idx] = raw[
                        output_offset:output_offset + n_tok
                    ]
                    output_offset += n_tok
            else:
                logger.debug(
                    "Encoder CUDA graph: batch_size=%d, tokens=%d, "
                    "budget=%d, waste=%.1f%%",
                    len(batch_orig_indices), batch_out_tokens, token_budget,
                    (token_budget - batch_out_tokens) / token_budget * 100,
                )
                encoder_metadata: dict = {}
                encoder_metadata['pos_embeds'] = (
                    self.vision_model.fast_pos_embed_interpolate(batch_grid)
                )
                rotary_cos, rotary_sin = self.vision_model.rot_pos_emb(batch_grid)
                encoder_metadata['rotary_pos_emb_cos'] = rotary_cos
                encoder_metadata['rotary_pos_emb_sin'] = rotary_sin

                seq_metadata = compute_encoder_metadata(
                    batch_grid,
                    device=self.device,
                    pad_to_batch_size=None,
                    per_frame=True,
                )
                encoder_metadata['cu_seqlens'] = seq_metadata['cu_seqlens']
                # Keep max_seqlen on CPU to avoid GPU sync during graph replay
                encoder_metadata['max_seqlen'] = seq_metadata['max_seqlen']

                # graph_hits counted inside _run_budget_graph after replay
                output = self._run_budget_graph(
                    batch_pixel_values, batch_grid, token_budget, encoder_metadata,
                )
                output_offset = 0
                for orig_idx in batch_orig_indices:
                    n_tok = per_image_out_tokens[orig_idx]
                    outputs_by_orig_idx[orig_idx] = output[
                        output_offset:output_offset + n_tok
                    ]
                    output_offset += n_tok

        # Return in original batch order (caller maps outputs to token positions)
        return [outputs_by_orig_idx[i] for i in range(num_images)]

    def execute(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> list[torch.Tensor] | None:
        """Execute encoder using CUDA graph with optional DP.

        Args:
            pixel_values: Concatenated pixel values
            grid_thw: Grid dimensions per image

        Returns:
            List of encoder outputs (one per image), or None if no matching budget.
        """
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list = grid_thw.tolist()
        else:
            grid_thw_list = grid_thw

        if self.use_dp:
            from vllm.model_executor.models.vision import (
                dp_shard_vision_inputs,
                dp_gather_vision_outputs,
            )

            spatial_merge_size_squared = self.vision_model.spatial_merge_size ** 2
            local_pixel_values, local_grid_thw_list, dp_meta = dp_shard_vision_inputs(
                pixel_values, grid_thw_list, spatial_merge_size_squared
            )

            local_outputs = self._execute_local(local_pixel_values, local_grid_thw_list)
            if local_outputs is None:
                return None

            hidden_size = self.vision_model.out_hidden_size
            outputs = dp_gather_vision_outputs(
                local_outputs, dp_meta, self.device, self.dtype, hidden_size
            )
            result = list(outputs)
        else:
            result = self._execute_local(pixel_values, grid_thw_list)

        # Log cumulative stats periodically
        if result is not None:
            stats = self.get_cumulative_stats()
            total_requests = self.graph_hits + self.graph_misses
            if total_requests > 0 and total_requests % self.log_stats_interval == 0:
                logger.debug(
                    "Encoder CUDA graph cumulative stats: hits=%d, misses=%d, hit_rate=%.1f%%",
                    stats["graph_hits"], stats["graph_misses"], stats["hit_rate"] * 100
                )

        return result

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative CUDA graph statistics."""
        total_requests = self.graph_hits + self.graph_misses
        hit_rate = (
            self.graph_hits / total_requests if total_requests > 0 else 0.0
        )

        return {
            "graph_hits": self.graph_hits,
            "graph_misses": self.graph_misses,
            "hit_rate": hit_rate,
            "num_budgets": len(self.budget_graphs),
            "token_budgets": self.token_budgets,
        }
