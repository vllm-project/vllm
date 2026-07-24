# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph manager for vision encoder budget-batch execution."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

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
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.model_executor.models.vision import get_load_balance_assignment
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import select_mm_items
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphConfig,
    EncoderItemSpec,
)

logger = init_logger(__name__)


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

        self.budget_graphs: dict[str, dict[int, BudgetGraphMetadata]] = {}
        self.graph_pool: Any | None = None
        self.graph_hits = 0
        self.graph_misses = 0
        self.log_stats_interval = 100

        max_budget = self.token_budgets[-1]
        if not self.config.paths:
            raise ValueError("Encoder CUDA graph config must define at least one path")
        self.path_token_budgets: dict[str, list[int]] = {}
        for path, path_config in self.config.paths.items():
            min_path_budget = path_config.min_token_budget
            if min_path_budget is None:
                budgets = list(self.token_budgets)
            else:
                if min_path_budget <= 0 or min_path_budget > max_budget:
                    raise ValueError(
                        f"Invalid minimum budget {min_path_budget} for encoder "
                        f"CUDA graph path {path!r}; max budget is {max_budget}"
                    )
                budgets = self._generate_budgets(min_path_budget, max_budget)
            if path_config.allow_zero_tokens:
                budgets.insert(0, 0)
            self.path_token_budgets[path] = budgets

        logger.info(
            "EncoderCudaGraphManager initialized with paths=%s, "
            "max_batch_size=%d, max_frames_per_batch=%s, use_dp=%s",
            self.path_token_budgets,
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

    def clear(self) -> None:
        """Release captured encoder CUDA graphs and the manager-local pool."""
        for graph_set in self.budget_graphs.values():
            graph_set.clear()
        self.graph_pool = None

    def capture(self, graph_pool: Any):
        """Capture CUDA graphs for every configured path and token budget."""
        self.graph_pool = graph_pool

        for path, budgets in self.path_token_budgets.items():
            for token_budget in sorted(budgets, reverse=True):
                if token_budget > 0:
                    self._capture_budget_graph(token_budget, path=path)

        logger.info(
            "Encoder CUDA graph capture complete. Captured %d graphs across %d paths.",
            self.get_num_graphs_to_capture(),
            len(self.path_token_budgets),
        )

    def get_num_graphs_to_capture(self) -> int:
        return sum(
            token_budget > 0
            for budgets in self.path_token_budgets.values()
            for token_budget in budgets
        )

    def _get_graph_set(self, path: str = "default") -> dict[int, BudgetGraphMetadata]:
        """Return the captured graphs for one encoder path."""
        if path not in self.budget_graphs:
            self.budget_graphs[path] = {}
        return self.budget_graphs[path]

    def _capture_budget_graph(self, token_budget: int, path: str = "default"):
        """Capture CUDA graph for a single token budget."""
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        graph_set = self._get_graph_set(path)

        values = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
            path,
        )

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            output_buffer = torch.empty_like(output)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            output_buffer.copy_(output)

        graph_set[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffers=values,
            output_buffer=output_buffer,
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

    def _get_item_specs(
        self, modality: str, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        """Get item specs from the model."""
        return self.model.get_encoder_cudagraph_item_specs(mm_kwargs, modality)

    def _get_per_item_out_tokens(
        self, modality: str, mm_kwargs: dict[str, Any]
    ) -> list[int]:
        """Get per-item output token counts as plain ints."""
        return [
            spec.output_tokens for spec in self._get_item_specs(modality, mm_kwargs)
        ]

    @staticmethod
    def _copy_padded_buffer(
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> None:
        dst.zero_()
        dst[: src.shape[0]].copy_(src)

    def _run_budget_graph(
        self,
        modality: str,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor | None:
        """Execute budget graph.

        Args:
            mm_kwargs: Multimodal inputs for the batch.
            token_budget: Token budget to use.
            path: Configured encoder path.
        Returns:
            Encoder outputs, or None if graph not captured.
        """
        graph_set = self._get_graph_set(path)
        num_items = len(self._get_item_specs(modality, mm_kwargs))

        if token_budget not in graph_set:
            self.graph_misses += num_items
            return None

        graph_meta = graph_set[token_budget]

        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            modality,
            self.max_batch_size,
            self.max_frames_per_batch,
            path,
        )

        # Copy replay values into the buffers recorded for this path.
        for key, buf in graph_meta.input_buffers.items():
            src = replay.get(key)
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

    def _run_eager(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor],
    ) -> None:
        with torch.inference_mode():
            raw_outputs = cast(SupportsMultiModal, self.model).embed_multimodal(
                **mm_kwargs
            )

        if raw_outputs is None:
            raise ValueError("Multimodal encoder returned no outputs")

        if isinstance(raw_outputs, torch.Tensor):
            if raw_outputs.ndim == 3 and raw_outputs.shape[0] == len(indices):
                outputs = list(raw_outputs.unbind(0))
            elif len(indices) == 1:
                outputs = [raw_outputs]
            elif raw_outputs.ndim == 2:
                split_sizes = [per_item_out_tokens[i] for i in indices]
                outputs = list(torch.split(raw_outputs, split_sizes))
            else:
                raise ValueError(
                    "Cannot map multimodal encoder tensor output with shape "
                    f"{tuple(raw_outputs.shape)} to {len(indices)} items"
                )
        else:
            outputs = list(raw_outputs)

        if len(outputs) != len(indices):
            raise ValueError(
                f"Multimodal encoder returned {len(outputs)} outputs for "
                f"{len(indices)} items"
            )
        for index, output in zip(indices, outputs):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected encoder tensor output, got {type(output)}")
            dest[index] = output

    def _execute_local(
        self,
        modality: str,
        items: Sequence[MultiModalKwargsItem],
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute locally using greedy packing across all configured paths."""
        item_specs = self._get_item_specs(modality, mm_kwargs)
        num_items = len(item_specs)
        if len(items) != num_items:
            raise ValueError(
                f"Got {len(items)} multimodal items but {num_items} item specs"
            )

        paths = tuple(self.path_token_budgets)
        per_item_path_tokens = {
            path: [spec.get_path_output_tokens(path) for spec in item_specs]
            for path in paths
        }
        per_item_out_tokens = [spec.output_tokens for spec in item_specs]
        max_path_budgets = {path: max(self.path_token_budgets[path]) for path in paths}

        sorted_indices = sorted(range(num_items), key=lambda i: per_item_out_tokens[i])
        batches: list[tuple[list[int], dict[str, int | None]]] = []
        current_batch: list[int] = []
        current_tokens = dict.fromkeys(paths, 0)

        def append_current_batch() -> None:
            if not current_batch:
                return
            path_budgets = {
                path: (
                    0
                    if current_tokens[path] == 0
                    else self._find_smallest_fitting_budget_given_tokens(
                        current_tokens[path], self.path_token_budgets[path]
                    )
                )
                for path in paths
            }
            batches.append((list(current_batch), path_budgets))

        for orig_idx in sorted_indices:
            item_tokens = {path: per_item_path_tokens[path][orig_idx] for path in paths}
            fits = len(current_batch) < self.max_batch_size and all(
                current_tokens[path] + item_tokens[path] <= max_path_budgets[path]
                for path in paths
            )
            if current_batch and not fits:
                append_current_batch()
                current_batch = []
                current_tokens = dict.fromkeys(paths, 0)

            current_batch.append(orig_idx)
            for path in paths:
                current_tokens[path] += item_tokens[path]

        append_current_batch()

        outputs_by_orig_idx: dict[int, torch.Tensor] = {}
        for batch_indices, path_budgets in batches:
            batch_mm_kwargs = select_mm_items(items, mm_kwargs, batch_indices)
            batch_path_tokens = {
                path: sum(per_item_path_tokens[path][i] for i in batch_indices)
                for path in paths
            }
            needs_eager = any(
                batch_path_tokens[path] > 0 and path_budgets[path] is None
                for path in paths
            )

            if needs_eager:
                logger.debug(
                    "Encoder CUDA graph fallback to eager for %d items: %s",
                    len(batch_indices),
                    batch_path_tokens,
                )
                self.graph_misses += len(batch_indices)
                self._run_eager(
                    batch_mm_kwargs,
                    batch_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                )
                continue

            graph_outputs: dict[str, torch.Tensor] = {}
            for path in paths:
                token_budget = path_budgets[path]
                if batch_path_tokens[path] == 0:
                    continue
                assert token_budget is not None and token_budget > 0
                output = self._run_budget_graph(
                    modality, batch_mm_kwargs, token_budget, path=path
                )
                assert output is not None
                graph_outputs[path] = output

            self.model.postprocess_encoder_output(
                graph_outputs,
                batch_indices,
                per_item_out_tokens,
                outputs_by_orig_idx,
                clone=True,
                batch_mm_kwargs=batch_mm_kwargs,
            )

        return [outputs_by_orig_idx[i] for i in range(num_items)]

    def _dp_shard(
        self,
        modality: str,
        items: Sequence[MultiModalKwargsItem],
        mm_kwargs: dict[str, Any],
        per_item_out_tokens: list[int],
    ) -> tuple[
        list[MultiModalKwargsItem],
        dict[str, Any],
        list[int],
        list[int],
        int,
    ]:
        """Distribute items across TP ranks for data-parallel execution."""
        tp_size = get_tensor_model_parallel_world_size()
        current_rank = get_tensor_model_parallel_rank()

        item_specs = self._get_item_specs(modality, mm_kwargs)
        per_item_input_sizes = [spec.input_size for spec in item_specs]
        image_rank_assignment, images_per_rank, _ = get_load_balance_assignment(
            per_item_input_sizes, tp_size
        )

        cum_images_per_rank = [0]
        for count in images_per_rank:
            cum_images_per_rank.append(cum_images_per_rank[-1] + count)

        local_indices = image_rank_assignment[
            cum_images_per_rank[current_rank] : cum_images_per_rank[current_rank + 1]
        ]
        local_items = [items[i] for i in local_indices]
        local_mm_kwargs = select_mm_items(items, mm_kwargs, local_indices)

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
            if per_item_out_tokens
            else 0
        )

        return (
            local_items,
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
        modality: str,
        items: Sequence[MultiModalKwargsItem],
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute an encoder batch using CUDA graphs with optional DP."""
        if modality not in self.config.modalities:
            raise ValueError(f"Unsupported encoder CUDA graph modality: {modality}")

        if self.use_dp:
            per_item_out_tokens = self._get_per_item_out_tokens(modality, mm_kwargs)
            (
                local_items,
                local_mm_kwargs,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            ) = self._dp_shard(modality, items, mm_kwargs, per_item_out_tokens)

            local_outputs = self._execute_local(modality, local_items, local_mm_kwargs)
            result = self._dp_gather(
                local_outputs,
                per_item_out_tokens,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            )
        else:
            result = self._execute_local(modality, items, mm_kwargs)

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
