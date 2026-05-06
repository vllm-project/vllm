# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph manager for vision encoder budget-batch execution."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.vision import get_load_balance_assignment
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphConfig,
)

logger = init_logger(__name__)


@dataclass
class BudgetGraphMetadata:
    """Metadata for a single budget graph.

    CUDA graph replay pattern:
    1. Copy new batch data into input_buffer (e.g. pixel_values)
    2. Copy precomputed values into metadata_buffers
    3. Replay graph
    4. Read encoder outputs from output_buffer
    """

    token_budget: int
    max_batch_size: int  # Max number of images/videos per batch
    max_frames_per_batch: int  # Max total frames per batch (for video)
    graph: torch.cuda.CUDAGraph
    # The input tensor updated before replay (e.g. pixel_values)
    input_buffer: torch.Tensor
    # Buffers recorded into the CUDA graph (e.g. embeddings, sequence metadata).
    # Before replay the manager zeros then slice-copies new data into these.
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
        model: SupportsEncoderCudaGraph,
    ):
        """Initialize CUDA graph manager with provided token budgets
        and max batch size."""
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype
        self.model = model
        self.config: EncoderCudaGraphConfig = model.get_encoder_cudagraph_config()

        comp_config = vllm_config.compilation_config
        user_budgets = comp_config.encoder_cudagraph_token_budgets
        user_max_vision_items = comp_config.encoder_cudagraph_max_vision_items_per_batch
        user_max_frames = comp_config.encoder_cudagraph_max_frames_per_batch

        multimodal_config = vllm_config.model_config.multimodal_config

        if user_budgets and user_max_vision_items > 0:
            # Fully user-specified
            self.token_budgets = sorted(user_budgets)
            self.max_batch_size = user_max_vision_items
        else:
            # Auto-infer missing values from model
            min_budget, max_budget = model.get_encoder_cudagraph_budget_range(
                vllm_config
            )
            self.token_budgets = (
                sorted(user_budgets)
                if user_budgets
                else self._generate_budgets(min_budget, max_budget)
            )
            self.max_batch_size = (
                user_max_vision_items
                if user_max_vision_items > 0
                else max_budget // min_budget
            )

        assert multimodal_config is not None
        if multimodal_config.get_limit_per_prompt("video") == 0:
            self.max_frames_per_batch = 0
        elif user_max_frames is not None:
            self.max_frames_per_batch = user_max_frames
        else:
            # Set it to the model-specific value according to its `processing_info`.
            max_frames_per_video = self.model.get_max_frames_per_video()
            self.max_frames_per_batch = self.max_batch_size * max_frames_per_video

        mm_config = vllm_config.model_config.multimodal_config
        self.use_dp = (
            mm_config is not None
            and mm_config.mm_encoder_tp_mode == "data"
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )

        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}
        self.graph_hits = 0
        self.graph_misses = 0
        self.log_stats_interval = 100

        logger.info(
            "EncoderCudaGraphManager initialized with "
            "budgets=%s, max_batch_size=%d, max_frames_per_batch=%s, use_dp=%s",
            self.token_budgets,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.use_dp,
        )

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int) -> list[int]:
        """Generate power-of-2 token budgets from min_budget to max_budget."""
        budgets: list[int] = []
        b = min_budget
        while b <= max_budget:
            budgets.append(b)
            b *= 2
        # Always include max_budget if it's not already a power-of-2 boundary
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets

    def supports_modality(self, modality: str) -> bool:
        """Check if a modality is supported by this manager."""
        return modality in self.config.modalities

    def capture(self):
        """Capture CUDA graphs for all token budgets."""
        for token_budget in self.token_budgets:
            self._capture_budget_graph(token_budget)

        logger.info(
            "Encoder CUDA graph capture complete. Captured %d budget graphs.",
            len(self.budget_graphs),
        )

    def _capture_budget_graph(self, token_budget: int):
        """Capture CUDA graph for a single token budget."""
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
        )

        mm_kwargs = capture_inputs.mm_kwargs
        buffers = capture_inputs.buffers

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
            output_buffer = torch.empty_like(output)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
            output_buffer.copy_(output)

        # Since the image and video modalities share the same per-patch shape,
        # so we can use the image dummy inputs to capture CUDA graph for both
        # image and video.
        input_key = self.config.input_key_by_modality["image"]
        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffer=mm_kwargs[input_key],
            metadata_buffers=buffers,
            output_buffer=output_buffer,
        )

    def _find_smallest_fitting_budget_given_tokens(
        self, total_tokens: int
    ) -> int | None:
        """Find smallest budget >= total_tokens.

        Returns:
            Token budget if found, None if no fitting budget.
        """
        for budget in self.token_budgets:
            if budget >= total_tokens:
                return budget
        return None

    def _get_per_item_out_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Get per-item output token counts as plain ints."""
        return [
            int(t)
            for t in self.model.get_encoder_cudagraph_per_item_output_tokens(mm_kwargs)
        ]

    @staticmethod
    def _scatter_output_slices(
        output: torch.Tensor,
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
    ) -> None:
        """Slice a concatenated output tensor and scatter into dest by index."""
        offset = 0
        for idx in indices:
            n_tok = per_item_out_tokens[idx]
            sliced = output[offset : offset + n_tok]
            dest[idx] = sliced.clone() if clone else sliced
            offset += n_tok

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        replay_buffers: dict[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        """Execute budget graph.

        Args:
            mm_kwargs: Multimodal inputs for the batch.
            token_budget: Token budget to use.
            replay_buffers: Buffer values to copy into captured buffers.
                None values leave the corresponding buffer unchanged.

        Returns:
            Encoder outputs, or None if graph not captured.
        """
        num_items = self.model.get_encoder_cudagraph_num_items(mm_kwargs)
        if token_budget not in self.budget_graphs:
            self.graph_misses += num_items
            return None

        graph_meta = self.budget_graphs[token_budget]

        # Copy the input tensor. Buffers are sized for the full budget;
        # actual inputs may be smaller. Zero then slice-copy so padded
        # positions are invisible to attention (cu_seqlens masks them out).
        input_key = self.config.input_key_by_modality[
            self.model.get_input_modality(mm_kwargs)
        ]
        src = mm_kwargs[input_key]
        n = src.shape[0]
        graph_meta.input_buffer[:n].copy_(src)

        # Copy metadata buffers using keys from config.buffer_keys.
        for key in self.config.buffer_keys:
            src = replay_buffers.get(key)
            if src is None:
                continue
            buf = graph_meta.metadata_buffers[key]
            if src.ndim == 0:
                buf.copy_(src)
            else:
                n = src.shape[0]
                buf.zero_()
                buf[:n].copy_(src)

        graph_meta.graph.replay()

        self.graph_hits += num_items
        return graph_meta.output_buffer

    def _execute_local(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute encoder on local inputs using greedy-packed CUDA graphs.

        Sort images by output token count (smallest first), then greedily pack
        as many images as possible into each batch while staying within
        max_budget tokens and max_batch_size. Once a batch is finalised (next
        image would overflow either constraint), find the smallest fitting
        budget once for that batch.

        By exchange argument, greedy smallest-first packing minimises eager
        fallbacks -- any other ordering yields a higher token sum in some batch,
        making that batch more likely to exceed the budget.

        Stats note:
          graph_hits  -- counted inside _run_budget_graph after successful replay.
          graph_misses -- counted here for single-image batches where the image
                         exceeds max_budget. Batches split due to max_batch_size
                         always satisfy total_tokens <= max_budget and therefore
                         always find a valid budget (no miss).
        """
        num_items = self.model.get_encoder_cudagraph_num_items(mm_kwargs)
        max_budget = self.token_budgets[-1]

        per_item_out_tokens = self._get_per_item_out_tokens(mm_kwargs)

        # Sort ascending by output token count (smallest first)
        sorted_indices = sorted(range(num_items), key=lambda i: per_item_out_tokens[i])

        # Greedy pack against max_budget and max_batch_size.
        # _find_smallest_fitting_budget_given_tokens is called once per
        # finalised batch, not per image.
        batches: list[tuple[list[int], int | None]] = []
        current_batch: list[int] = []
        current_batch_tokens = 0

        for orig_idx in sorted_indices:
            item_tokens = per_item_out_tokens[orig_idx]
            if (
                current_batch_tokens + item_tokens <= max_budget
                and len(current_batch) < self.max_batch_size
            ):
                current_batch.append(orig_idx)
                current_batch_tokens += item_tokens
            else:
                if current_batch:
                    batches.append(
                        (
                            current_batch,
                            self._find_smallest_fitting_budget_given_tokens(
                                current_batch_tokens
                            ),
                        )
                    )
                current_batch = [orig_idx]
                current_batch_tokens = item_tokens

        if current_batch:
            batches.append(
                (
                    current_batch,
                    self._find_smallest_fitting_budget_given_tokens(
                        current_batch_tokens
                    ),
                )
            )

        # outputs_by_orig_idx maps each original image index to its output
        # tensor. Needed because greedy packing reorders images; we restore
        # the original order before returning.
        outputs_by_orig_idx: dict[int, torch.Tensor] = {}

        for batch_orig_indices, token_budget in batches:
            batch_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs, batch_orig_indices
            )
            batch_out_tokens = sum(per_item_out_tokens[i] for i in batch_orig_indices)

            if token_budget is None:
                # Single oversized image: item_tokens > max_budget.
                # graph_misses counted here for this eager fallback.
                logger.debug(
                    "Encoder CUDA graph fallback to eager: no budget for "
                    "%d tokens from %d images",
                    batch_out_tokens,
                    len(batch_orig_indices),
                )
                self.graph_misses += len(batch_orig_indices)
                with torch.inference_mode():
                    raw = self.model.encoder_eager_forward(batch_mm_kwargs)
                self._scatter_output_slices(
                    raw,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                )
            else:
                logger.debug(
                    "Encoder CUDA graph: batch_size=%d, tokens=%d, "
                    "budget=%d, waste=%.1f%%",
                    len(batch_orig_indices),
                    batch_out_tokens,
                    token_budget,
                    (token_budget - batch_out_tokens) / token_budget * 100,
                )
                replay = self.model.prepare_encoder_cudagraph_replay_buffers(
                    batch_mm_kwargs,
                    self.max_batch_size,
                    self.max_frames_per_batch,
                )

                # graph_hits counted inside _run_budget_graph after replay.
                output = self._run_budget_graph(
                    batch_mm_kwargs, token_budget, replay.buffers
                )
                assert output is not None
                self._scatter_output_slices(
                    output,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                    clone=True,
                )

        # Return in original batch order (caller maps outputs to token positions)
        return [outputs_by_orig_idx[i] for i in range(num_items)]

    def _dp_shard(
        self,
        mm_kwargs: dict[str, Any],
        per_item_out_tokens: list[int],
    ) -> tuple[dict[str, Any], list[int], list[int], int]:
        """Distribute items across TP ranks for data-parallel execution.

        Uses get_load_balance_assignment() to balance load by input size,
        then select_encoder_cudagraph_items() to extract each rank's inputs.

        Returns:
            local_mm_kwargs: Inputs for this rank.
            image_rank_assignment: Flattened assignment order across all ranks.
            images_per_rank: Number of items per rank.
            max_output_tokens_per_rank: Max output tokens across all ranks
                (for padding during all_gather).
        """
        tp_size = get_tensor_model_parallel_world_size()
        current_rank = get_tensor_model_parallel_rank()

        per_item_input_sizes = self.model.get_encoder_cudagraph_per_item_input_sizes(
            mm_kwargs
        )

        (image_rank_assignment, images_per_rank, input_patches_per_rank) = (
            get_load_balance_assignment(per_item_input_sizes, tp_size)
        )

        # Extract local indices for this rank
        cum_images_per_rank = [0]
        for count in images_per_rank:
            cum_images_per_rank.append(cum_images_per_rank[-1] + count)

        local_indices = image_rank_assignment[
            cum_images_per_rank[current_rank] : cum_images_per_rank[current_rank + 1]
        ]

        if len(local_indices) > 0:
            local_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs, local_indices
            )
        else:
            local_mm_kwargs = self.model.select_encoder_cudagraph_items(mm_kwargs, [])

        max_output_tokens_per_rank = (
            max(
                sum(
                    per_item_out_tokens[i]
                    for i in image_rank_assignment[
                        cum_images_per_rank[r] : cum_images_per_rank[r + 1]
                    ]
                )
                for r in range(tp_size)
            )
            if len(per_item_out_tokens) > 0
            else 0
        )

        return (
            local_mm_kwargs,
            image_rank_assignment,
            images_per_rank,
            max_output_tokens_per_rank,
        )

    def _dp_gather(
        self,
        local_outputs: list[torch.Tensor],
        per_item_out_tokens: list[int],
        image_rank_assignment: list[int],
        images_per_rank: list[int],
        max_output_tokens_per_rank: int,
    ) -> list[torch.Tensor]:
        """Gather outputs from all TP ranks and reorder to original sequence.

        Assumes 2D output tensors [tokens, hidden]. Follows the same
        pad -> all_gather -> unpad -> reorder algorithm as
        run_dp_sharded_mrope_vision_model() in the eager path.
        """
        hidden_size = self.config.out_hidden_size
        tp_size = len(images_per_rank)

        if len(local_outputs) > 0:
            local_concat = torch.cat(local_outputs, dim=0)
        else:
            local_concat = torch.empty(
                (0, hidden_size), device=self.device, dtype=self.dtype
            )

        # Pad to max_output_tokens_per_rank for all_gather
        current_len = local_concat.shape[0]
        if current_len < max_output_tokens_per_rank:
            padding = torch.empty(
                (max_output_tokens_per_rank - current_len, hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            local_padded = torch.cat([local_concat, padding], dim=0)
        else:
            local_padded = local_concat

        gathered = tensor_model_parallel_all_gather(local_padded, dim=0)

        # Unpad each rank's contribution
        rank_outputs: list[torch.Tensor] = []
        current_idx = 0
        for rank in range(tp_size):
            start = rank * max_output_tokens_per_rank
            rank_count = images_per_rank[rank]
            rank_indices = image_rank_assignment[current_idx : current_idx + rank_count]
            rank_tokens = sum(per_item_out_tokens[i] for i in rank_indices)
            current_idx += rank_count
            rank_outputs.append(gathered[start : start + rank_tokens])

        # Reorder to original sequence
        total_items = len(per_item_out_tokens)
        result: list[torch.Tensor | None] = [None] * total_items
        current_idx = 0
        for rank in range(tp_size):
            count = images_per_rank[rank]
            if count > 0:
                rank_items = image_rank_assignment[current_idx : current_idx + count]
                self._scatter_output_slices(
                    rank_outputs[rank],
                    rank_items,
                    per_item_out_tokens,
                    result,
                )
                current_idx += count

        return [t for t in result if t is not None]

    def execute(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute encoder using CUDA graph with optional DP.

        Args:
            mm_kwargs: Multimodal keyword arguments containing the
                input tensor and grid dimensions.

        Returns:
            List of encoder outputs (one per item).
        """
        if self.use_dp:
            per_item_out_tokens = self._get_per_item_out_tokens(mm_kwargs)

            (
                local_mm_kwargs,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            ) = self._dp_shard(mm_kwargs, per_item_out_tokens)

            local_outputs = self._execute_local(local_mm_kwargs)

            result = self._dp_gather(
                local_outputs,
                per_item_out_tokens,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            )
        else:
            result = self._execute_local(mm_kwargs)

        # Log cumulative stats periodically
        stats = self.get_cumulative_stats()
        total_requests = self.graph_hits + self.graph_misses
        if total_requests > 0 and total_requests % self.log_stats_interval == 0:
            logger.debug(
                "Encoder CUDA graph cumulative stats: "
                "hits=%d, misses=%d, hit_rate=%.1f%%",
                stats["graph_hits"],
                stats["graph_misses"],
                stats["hit_rate"] * 100,
            )

        return result

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative CUDA graph statistics."""
        total_requests = self.graph_hits + self.graph_misses
        hit_rate = self.graph_hits / total_requests if total_requests > 0 else 0.0

        return {
            "graph_hits": self.graph_hits,
            "graph_misses": self.graph_misses,
            "hit_rate": hit_rate,
            "num_budgets": len(self.budget_graphs),
            "token_budgets": self.token_budgets,
        }
