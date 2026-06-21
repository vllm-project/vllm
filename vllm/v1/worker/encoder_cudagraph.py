# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph manager for vision encoder budget-batch execution."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
)
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.model_executor.models.vision import get_load_balance_assignment
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphConfig,
    EncoderItemSpec,
)

logger = init_logger(__name__)

BudgetGraphMapKey: TypeAlias = int | tuple[int, Hashable]


@dataclass
class BudgetGraphMetadata:
    """Metadata for a single budget graph.

    CUDA graph replay pattern:
    * Copy precomputed values into input_buffers
    * Replay graph
    * Read encoder outputs from output_buffer
    """

    token_budget: int
    max_batch_size: int  # Max number of images/videos per batch
    max_frames_per_batch: int  # Max total frames per batch (for video)
    graph: torch.cuda.CUDAGraph
    # Buffers recorded into the CUDA graph (e.g. embeddings, sequence metadata).
    # Before replay the manager updates these in-place. By default buffers are
    # zeroed before slice-copying the actual values; model-specific padding
    # behavior is provided by EncoderCudaGraphConfig.padding_logics.
    input_buffers: dict[str, torch.Tensor]
    # Output written by graph, read after replay
    output_buffer: torch.Tensor
    # Second CUDA-graph capture axis key, None if unused.
    secondary_capture_axis_key: Hashable | None = None


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

        if (
            self.config.enable_dual_path_graph
            and self.config.enable_secondary_capture_axis
        ):
            raise NotImplementedError(
                "Combining dual-path encoder CUDA graphs with secondary capture axis "
                "is not supported yet."
            )

        # Invariant: max_batch_size <= min_token_budget.
        # This ensures per_image_output = budget // max_batch_size >= 1
        # for every captured budget, preventing reshape crashes on empty
        # tensors during CUDA graph capture. Validated/enforced below for
        # each configuration path.
        if user_budgets and user_max_vision_items > 0:
            # Fully user-specified: validate the invariant.
            self.token_budgets = sorted(user_budgets)
            self.max_batch_size = user_max_vision_items
            min_tok = min(self.token_budgets)
            if self.max_batch_size > min_tok:
                raise ValueError(
                    f"encoder_cudagraph_max_vision_items_per_batch "
                    f"({self.max_batch_size}) must be <= smallest token "
                    f"budget ({min_tok}). With budgets="
                    f"{self.token_budgets}, per_image_output = "
                    f"{min_tok} // {self.max_batch_size} = "
                    f"{min_tok // self.max_batch_size}, which would cause "
                    f"a capture failure. Either increase the smallest "
                    f"budget or decrease max_vision_items_per_batch."
                )
        else:
            # Auto-infer missing values from model.
            min_budget, max_budget = model.get_encoder_cudagraph_budget_range(
                vllm_config
            )
            if min_budget <= 0 or max_budget <= 0:
                raise ValueError(
                    f"Invalid encoder cudagraph budget range: "
                    f"min_budget={min_budget}, max_budget={max_budget}. "
                    f"Both must be positive."
                )
            if min_budget > max_budget:
                raise ValueError(
                    f"Invalid encoder cudagraph budget range: "
                    f"min_budget={min_budget} > max_budget={max_budget}."
                )

            if user_max_vision_items > 0:
                # User provided max_vision_items only; adjust auto-inferred
                # budgets so min(budgets) >= max_batch_size.
                self.max_batch_size = user_max_vision_items
                effective_min = max(min_budget, user_max_vision_items)
                self.token_budgets = self._generate_budgets(effective_min, max_budget)
            elif user_budgets:
                # User provided budgets only; cap auto-inferred
                # max_batch_size to min(user_budgets).
                self.token_budgets = sorted(user_budgets)
                self.max_batch_size = min(
                    max_budget // min_budget,
                    min(self.token_budgets),
                )
            else:
                # Fully auto-inferred.
                self.token_budgets = self._generate_budgets(min_budget, max_budget)
                self.max_batch_size = min(
                    max_budget // min_budget,
                    min(self.token_budgets),
                )

        assert multimodal_config is not None
        if multimodal_config.get_limit_per_prompt("video") == 0:
            self.max_frames_per_batch = 0
        elif user_max_frames is not None:
            self.max_frames_per_batch = user_max_frames
        else:
            # Set it to the model-specific value from config.
            max_frames_per_video = self.config.max_frames_per_video
            self.max_frames_per_batch = self.max_batch_size * max_frames_per_video

        mm_config = vllm_config.model_config.multimodal_config
        self.use_dp = (
            mm_config is not None
            and mm_config.mm_encoder_tp_mode == "data"
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )

        self._ordered_secondary_capture_axis_keys: tuple[Hashable, ...] | None = None
        if self.config.enable_secondary_capture_axis:
            keys = tuple(self.model.get_encoder_cudagraph_secondary_capture_axis_keys())
            if not keys:
                raise ValueError(
                    "Secondary capture axis is enabled but no keys were returned."
                )
            self._ordered_secondary_capture_axis_keys = keys

        self.budget_graphs: dict[str, dict[BudgetGraphMapKey, BudgetGraphMetadata]] = {}
        self.graph_pool: Any | None = None
        self.graph_hits = 0
        self.graph_misses = 0
        self.log_stats_interval = 100

        if self.config.enable_dual_path_graph:
            max_budget = self.token_budgets[-1]
            self.global_token_budgets = self._generate_budgets(
                self.config.global_token_per_image,
                max_budget,
            )
            self.local_token_budgets = self._generate_budgets(
                self.config.local_token_per_patch,
                max_budget,
            )
            # When `image_width <= 640 and image_height <= 640`, the mm inputs
            # will only contain global image, without generating local patches.
            self.local_token_budgets.insert(0, 0)
            logger.info(
                "EncoderCudaGraphManager dual-path mode: "
                "global_budgets=%s, local_budgets=%s",
                self.global_token_budgets,
                self.local_token_budgets,
            )
        else:
            logger.info(
                "EncoderCudaGraphManager initialized with "
                "budgets=%s, max_batch_size=%d, max_frames_per_batch=%s, use_dp=%s, "
                "ordered_secondary_capture_axis_keys=%s",
                self.token_budgets,
                self.max_batch_size,
                self.max_frames_per_batch,
                self.use_dp,
                self._ordered_secondary_capture_axis_keys,
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

    def clear(self) -> None:
        """Release captured encoder CUDA graphs and the manager-local pool."""
        for graph_set in self.budget_graphs.values():
            graph_set.clear()
        self.graph_pool = None

    def capture(self, graph_pool: Any):
        """Capture CUDA graphs for all token budgets."""
        self.graph_pool = graph_pool

        if self.config.enable_dual_path_graph:
            self._capture_one_path(self.global_token_budgets, path="global")
            self._capture_one_path(self.local_token_budgets, path="local")
            logger.info(
                "Encoder CUDA graph capture complete. "
                "Captured %d global + %d local budget graphs.",
                len(self.budget_graphs["global"]),
                len(self.budget_graphs["local"]),
            )
            return

        self._capture_one_path(self.token_budgets, path="default")
        logger.info(
            "Encoder CUDA graph capture complete. Captured %d budget graphs.",
            len(self.budget_graphs["default"]),
        )

    def _capture_one_path(self, budgets: list[int], path: str = "default") -> None:
        for token_budget in sorted(budgets, reverse=True):
            if token_budget == 0:
                continue

            if not self.config.enable_secondary_capture_axis:
                self._capture_budget_graph(token_budget, path=path)
                continue

            for secondary_capture_axis_key in self._ordered_secondary_capture_axis_keys:
                self._capture_budget_graph(
                    token_budget,
                    path=path,
                    secondary_capture_axis_key=secondary_capture_axis_key,
                )

    def _num_graphs_for_budgets(self, budgets: list[int]) -> int:
        num_budgets = sum(1 for budget in budgets if budget != 0)
        if self.config.enable_secondary_capture_axis:
            assert self._ordered_secondary_capture_axis_keys is not None
            return num_budgets * len(self._ordered_secondary_capture_axis_keys)
        return num_budgets

    def get_num_graphs_to_capture(self) -> int:
        if self.config.enable_dual_path_graph:
            return self._num_graphs_for_budgets(
                self.global_token_budgets
            ) + self._num_graphs_for_budgets(self.local_token_budgets)
        return self._num_graphs_for_budgets(self.token_budgets)

    def _get_graph_set(
        self, path: str = "default"
    ) -> dict[BudgetGraphMapKey, BudgetGraphMetadata]:
        # Lazy init global/local graph sets for dual-path models, or default graph
        # set for single-path models.
        if path not in self.budget_graphs:
            self.budget_graphs[path] = {}
        return self.budget_graphs[path]

    def _capture_budget_graph(
        self,
        token_budget: int,
        path: str = "default",
        secondary_capture_axis_key: Hashable | None = None,
    ):
        """Capture CUDA graph for a single token budget."""
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d, secondary_capture_axis_key=%s",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            secondary_capture_axis_key,
        )

        graph_set = self._get_graph_set(path)

        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
            path,
            secondary_capture_axis_key,
        )

        values = capture_inputs.values

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            output_buffer = torch.empty_like(output)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            output_buffer.copy_(output)

        graph_map_key: BudgetGraphMapKey = (
            token_budget
            if secondary_capture_axis_key is None
            else (token_budget, secondary_capture_axis_key)
        )
        graph_set[graph_map_key] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffers=values,
            output_buffer=output_buffer,
            secondary_capture_axis_key=secondary_capture_axis_key,
        )

    def _find_smallest_fitting_budget_given_tokens(
        self, total_tokens: int, budgets: list[int] | None = None
    ) -> int | None:
        """Find smallest budget >= total_tokens.

        Returns:
            Token budget if found, None if no fitting budget.
        """
        budgets = budgets if budgets is not None else self.token_budgets
        for budget in budgets:
            if budget >= total_tokens:
                return budget
        return None

    def _get_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        """Get item specs from the model."""
        return self.model.get_encoder_cudagraph_item_specs(mm_kwargs)

    def _get_per_item_out_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Get per-item output token counts as plain ints."""
        return [spec.output_tokens for spec in self._get_item_specs(mm_kwargs)]

    @staticmethod
    def _copy_padded_buffer(
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> None:
        dst.zero_()
        dst[: src.shape[0]].copy_(src)

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
        secondary_capture_axis_key: Hashable | None = None,
    ) -> torch.Tensor | None:
        """Execute budget graph.

        Args:
            mm_kwargs: Multimodal inputs for the batch.
            token_budget: Token budget to use.
            path: Path for the graph. Should be one of ["default", "global", "local"].
        Returns:
            Encoder outputs, or None if graph not captured.
        """
        graph_set = self._get_graph_set(path)
        num_items = len(self._get_item_specs(mm_kwargs))

        graph_map_key: BudgetGraphMapKey = (
            token_budget
            if secondary_capture_axis_key is None
            else (token_budget, secondary_capture_axis_key)
        )
        if graph_map_key not in graph_set:
            self.graph_misses += num_items
            return None

        graph_meta = graph_set[graph_map_key]

        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            self.max_frames_per_batch,
            path,
        )

        # Copy replay buffers into graph input buffers. Iterate over the
        # graph's own buffer keys (which may differ per path for dual-path
        # models) rather than the global config.buffer_keys.
        for key, buf in graph_meta.input_buffers.items():
            src = replay.values.get(key)
            if src is None:
                continue
            if src.ndim == 0:
                buf.copy_(src)
            else:
                padding_logic = self.config.padding_logics.get(
                    key, self._copy_padded_buffer
                )
                padding_logic(buf, src)

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

        For dual-path models (``enable_dual_path_graph=True``), two independent
        graph sets are used: one for global images, one for local patches.
        Budgets are found independently per path; if only one path fits, the
        other falls back to eager via partial fallback.

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
        if self.config.enable_dual_path_graph:
            return self._execute_local_dual_path(mm_kwargs)
        return self._execute_local_single_path(mm_kwargs)

    def _execute_local_single_path(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Single-path greedy-packing execution (original behaviour)."""
        item_specs = self._get_item_specs(mm_kwargs)
        num_items = len(item_specs)
        max_budget = self.token_budgets[-1]

        per_item_out_tokens = [spec.output_tokens for spec in item_specs]

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
            batch_out_tokens = sum(per_item_out_tokens[i] for i in batch_orig_indices)

            if token_budget is None:
                # Single oversized image: item_tokens > max_budget.
                # graph_misses counted here for this eager fallback.
                batch_mm_kwargs = self.model.select_encoder_cudagraph_items(
                    mm_kwargs,
                    batch_orig_indices,
                )
                logger.debug(
                    "Encoder CUDA graph fallback to eager: no budget for "
                    "%d tokens from %d images",
                    batch_out_tokens,
                    len(batch_orig_indices),
                )
                self.graph_misses += len(batch_orig_indices)
                with torch.inference_mode():
                    raw = self.model.encoder_eager_forward(batch_mm_kwargs)
                scatter_output_slices(
                    raw,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                )
                continue

            secondary_capture_axis_key: Hashable | None = None
            if self.config.enable_secondary_capture_axis:
                secondary_capture_axis_key = (
                    self.model.resolve_encoder_cudagraph_secondary_capture_axis_key(
                        mm_kwargs,
                        batch_orig_indices,
                        self._ordered_secondary_capture_axis_keys,
                    )
                )

            batch_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs,
                batch_orig_indices,
                secondary_capture_axis_key,
            )
            logger.debug(
                "Encoder CUDA graph: batch_size=%d, tokens=%d, budget=%d, waste=%.1f%%",
                len(batch_orig_indices),
                batch_out_tokens,
                token_budget,
                (token_budget - batch_out_tokens) / token_budget * 100,
            )

            # graph_hits counted inside _run_budget_graph after replay.
            output = self._run_budget_graph(
                batch_mm_kwargs,
                token_budget,
                path="default",
                secondary_capture_axis_key=secondary_capture_axis_key,
            )
            assert output is not None
            self.model.postprocess_encoder_output(
                output,
                batch_orig_indices,
                per_item_out_tokens,
                outputs_by_orig_idx,
                clone=True,
                batch_mm_kwargs=batch_mm_kwargs,
            )

        # Return in original batch order (caller maps outputs to token positions)
        return [outputs_by_orig_idx[i] for i in range(num_items)]

    def _execute_local_dual_path(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Dual-path greedy-packing execution.

        Each image contributes both global tokens (constant per image)
        and local tokens (patches * patch_tokens). Greedy packing
        respects both budgets independently, then selects the smallest
        fitting budget per path with partial eager fallback.
        """
        item_specs = self._get_item_specs(mm_kwargs)
        num_items = len(item_specs)

        max_global_budget = self.global_token_budgets[-1]
        max_local_budget = self.local_token_budgets[-1]

        per_item_global_tokens = [spec.global_output_tokens for spec in item_specs]
        per_item_local_tokens = [spec.local_output_tokens for spec in item_specs]
        per_item_total_tokens = [spec.output_tokens for spec in item_specs]

        # Sort ascending by total output tokens
        sorted_indices = sorted(
            range(num_items), key=lambda i: per_item_total_tokens[i]
        )

        # Each batch is a tuple of (indices, global_budget, local_budget).
        batches: list[tuple[list[int], int | None, int | None]] = []
        current_batch: list[int] = []
        current_global_tokens = 0
        current_local_tokens = 0

        for orig_idx in sorted_indices:
            global_token = per_item_global_tokens[orig_idx]
            local_token = per_item_local_tokens[orig_idx]
            if (
                current_global_tokens + global_token <= max_global_budget
                and current_local_tokens + local_token <= max_local_budget
                and len(current_batch) < self.max_batch_size
            ):
                current_batch.append(orig_idx)
                current_global_tokens += global_token
                current_local_tokens += local_token
            else:
                if current_batch:
                    batches.append(
                        (
                            current_batch,
                            self._find_smallest_fitting_budget_given_tokens(
                                current_global_tokens, self.global_token_budgets
                            ),
                            self._find_smallest_fitting_budget_given_tokens(
                                current_local_tokens, self.local_token_budgets
                            ),
                        )
                    )
                current_batch = [orig_idx]
                current_global_tokens = global_token
                current_local_tokens = local_token

        if current_batch:
            batches.append(
                (
                    current_batch,
                    self._find_smallest_fitting_budget_given_tokens(
                        current_global_tokens, self.global_token_budgets
                    ),
                    self._find_smallest_fitting_budget_given_tokens(
                        current_local_tokens, self.local_token_budgets
                    ),
                )
            )

        outputs_by_orig_idx: dict[int, torch.Tensor] = {}

        for batch_orig_indices, global_budget, local_budget in batches:
            batch_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs, batch_orig_indices
            )
            batch_global_tokens = sum(
                per_item_global_tokens[i] for i in batch_orig_indices
            )
            batch_local_tokens = sum(
                per_item_local_tokens[i] for i in batch_orig_indices
            )

            both_eager = global_budget is None and local_budget is None

            if both_eager:
                logger.debug(
                    "Encoder CUDA graph dual-path full eager fallback: "
                    "%d global + %d local tokens from %d images",
                    batch_global_tokens,
                    batch_local_tokens,
                    len(batch_orig_indices),
                )
                self.graph_misses += len(batch_orig_indices)
                with torch.inference_mode():
                    raw = self.model.encoder_eager_forward(batch_mm_kwargs)
                per_item_total = [
                    per_item_global_tokens[i] + per_item_local_tokens[i] + 1
                    for i in batch_orig_indices
                ]
                scatter_output_slices(
                    raw, batch_orig_indices, per_item_total, outputs_by_orig_idx
                )
                continue

            logger.debug(
                "Encoder CUDA graph dual-path: batch_size=%d, "
                "global=%d (budget=%s), local=%d (budget=%s)",
                len(batch_orig_indices),
                batch_global_tokens,
                global_budget,
                batch_local_tokens,
                local_budget,
            )

            # Execute global path: graph or eager fallback
            if global_budget is not None:
                global_output = self._run_budget_graph(
                    batch_mm_kwargs,
                    global_budget,
                    path="global",
                )
                assert global_output is not None
            else:
                with torch.inference_mode():
                    global_output = self.model.encoder_eager_forward(
                        batch_mm_kwargs, path="global"
                    )

            # Execute local path: graph or eager fallback
            if local_budget is not None and batch_local_tokens > 0:
                local_output = self._run_budget_graph(
                    batch_mm_kwargs,
                    local_budget,
                    path="local",
                )
                assert local_output is not None
            elif batch_local_tokens > 0:
                with torch.inference_mode():
                    local_output = self.model.encoder_eager_forward(
                        batch_mm_kwargs, path="local"
                    )
            else:
                local_output = None

            self.model.postprocess_encoder_output(
                global_output,
                batch_orig_indices,
                per_item_global_tokens,
                outputs_by_orig_idx,
                clone=True,
                batch_mm_kwargs=batch_mm_kwargs,
                local_output=local_output,
            )

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

        item_specs = self._get_item_specs(mm_kwargs)
        per_item_input_sizes = [spec.input_size for spec in item_specs]

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
                scatter_output_slices(
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

        num_budgets = sum(len(g) for g in self.budget_graphs.values())

        return {
            "graph_hits": self.graph_hits,
            "graph_misses": self.graph_misses,
            "hit_rate": hit_rate,
            "num_budgets": num_budgets,
            "token_budgets": self.token_budgets,
        }
