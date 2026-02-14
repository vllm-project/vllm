"""CUDA graph manager for vision encoder budget-batch execution."""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Any

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


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
        """Initialize CUDA graph manager."""
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

        logger.info(
            f"EncoderCudaGraphManager initialized with budgets={self.token_budgets}, "
            f"max_batch_size={self.max_batch_size}, use_dp={self.use_dp}"
        )

    def capture(self):
        """Capture CUDA graphs for all token budgets."""
        logger.info("Starting encoder CUDA graph capture...")

        for token_budget in self.token_budgets:
            self._capture_budget_graph(token_budget)

        logger.info(
            f"Encoder CUDA graph capture complete. "
            f"Captured {len(self.budget_graphs)} budget graphs."
        )

    def _capture_budget_graph(self, token_budget: int):
        """Capture CUDA graph for a single token budget."""
        logger.info(f"Capturing CUDA graph for token_budget={token_budget}")

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
        spatial_merge_size = self.vision_model.spatial_merge_size
        seq_metadata = compute_encoder_metadata(
            dummy_grid_thw,
            device=self.device,
            spatial_merge_size=spatial_merge_size,
            pad_to_batch_size=None,
            per_frame=True,
        )
        encoder_metadata['cu_seqlens'] = seq_metadata['cu_seqlens']
        # Use CPU tensor from compute_encoder_metadata to avoid GPU sync
        # when .item() is called during CUDA graph capture
        encoder_metadata['max_seqlen'] = seq_metadata['max_seqlen']

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

        logger.info(
            f"Captured CUDA graph for budget {token_budget}: "
            f"output_shape={output_buffer.shape}"
        )

    def _generate_grid_config_for_budget(
        self, token_budget: int, max_batch_size: int
    ) -> list[list[int]]:
        """Generate dummy grid configuration for CUDA graph capture.

        This creates an arbitrary example that produces token_budget tokens.
        NOT used for runtime batching decisions - only for generating dummy inputs.

        Uses square grids for simplicity: [1, H, H] where H accounts for spatial merging.
        """
        # token_budget refers to output tokens after patch merger
        # Need to account for spatial downsampling if model has a merger
        spatial_merge_size = self.vision_model.spatial_merge_size

        # Target output tokens per image
        target_output_tokens_per_image = token_budget // max_batch_size

        # Input patches needed: output_tokens * (spatial_merge_size^2)
        target_input_patches_per_image = target_output_tokens_per_image * (spatial_merge_size ** 2)

        # Use square grids: H = W = sqrt(input_patches_per_image)
        H = int(math.sqrt(target_input_patches_per_image))

        # Create uniform grid config [T, H, W] for all images
        grid_config = [[1, H, H] for _ in range(max_batch_size)]

        return grid_config

    def _prepare_dummy_inputs(
        self, grid_config: list[list[int]]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Create dummy pixel_values and grid_thw for capture."""
        # Compute total patches from grid config
        total_patches = sum(t * h * w for t, h, w in grid_config)

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

    def find_budget_graph(self, total_tokens: int) -> Optional[int]:
        """Find smallest budget >= total_tokens.

        Returns:
            Token budget if found, None if no fitting budget.
        """
        for budget in self.token_budgets:
            if budget >= total_tokens:
                return budget
        return None

    def run_budget_graph(
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
        if token_budget not in self.budget_graphs:
            self.graph_misses += 1
            return None

        graph_meta = self.budget_graphs[token_budget]

        graph_meta.input_buffers['pixel_values'].copy_(pixel_values)

        for key in ['pos_embeds', 'rotary_pos_emb_cos', 'rotary_pos_emb_sin',
                    'cu_seqlens', 'max_seqlen']:
            graph_meta.metadata_buffers[key].copy_(encoder_metadata[key])

        graph_meta.graph.replay()

        self.graph_hits += 1
        return graph_meta.output_buffer

    def _execute_local(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list[list[int]],
    ) -> list[torch.Tensor] | None:
        """Execute encoder on local inputs using CUDA graph."""
        spatial_merge_size = self.vision_model.spatial_merge_size
        total_tokens = sum(
            (t * (h // spatial_merge_size) * (w // spatial_merge_size))
            for t, h, w in grid_thw_list
        )

        token_budget = self.find_budget_graph(total_tokens)
        if token_budget is None:
            return None

        encoder_metadata = {}
        encoder_metadata['pos_embeds'] = self.vision_model.fast_pos_embed_interpolate(
            grid_thw_list
        )
        rotary_cos, rotary_sin = self.vision_model.rot_pos_emb(grid_thw_list)
        encoder_metadata['rotary_pos_emb_cos'] = rotary_cos
        encoder_metadata['rotary_pos_emb_sin'] = rotary_sin

        from vllm.model_executor.models.vision import compute_encoder_metadata
        seq_metadata = compute_encoder_metadata(
            grid_thw_list,
            device=self.device,
            spatial_merge_size=spatial_merge_size,
            pad_to_batch_size=None,
            per_frame=True,
        )
        encoder_metadata['cu_seqlens'] = seq_metadata['cu_seqlens']
        # Use CPU tensor from compute_encoder_metadata to avoid GPU sync
        # when .item() is called during CUDA graph capture
        encoder_metadata['max_seqlen'] = seq_metadata['max_seqlen']

        output = self.run_budget_graph(
            pixel_values,
            grid_thw_list,
            token_budget,
            encoder_metadata,
        )

        outputs = []
        start_idx = 0
        for t, h, w in grid_thw_list:
            num_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
            outputs.append(output[start_idx:start_idx + num_tokens])
            start_idx += num_tokens

        return outputs

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
            return list(outputs)
        else:
            return self._execute_local(pixel_values, grid_thw_list)

    def get_stats(self) -> dict[str, Any]:
        """Get CUDA graph statistics."""
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
