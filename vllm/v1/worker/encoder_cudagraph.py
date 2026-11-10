# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph manager for vision encoder budget-batch execution."""

import itertools
import math
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
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import round_up_to
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.encoder_cudagraph_defs import (
    ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
    EncoderCudaGraphConfig,
    EncoderItemSpec,
)

logger = init_logger(__name__)

CaptureAxisKeys: TypeAlias = tuple[Hashable, ...]

BudgetGraphMapKey: TypeAlias = int | tuple[int, CaptureAxisKeys]

# Auto-generated budgets step by 128 tokens up to this size, then by 1/8 octave.
_LINEAR_GRID_END = 2560

# Warn when capture axes multiply the captured graph count past this point.
_CAPTURE_GRAPH_COUNT_WARNING_THRESHOLD = 64


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
    # Per-axis capture keys, empty when no extra capture axis is used.
    axis_keys: CaptureAxisKeys = ()


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
        max_token_budget = comp_config.encoder_cudagraph_max_token_budget

        multimodal_config = vllm_config.model_config.multimodal_config

        self._capture_axes: tuple[CaptureAxisKeys, ...] = tuple(
            tuple(axis) for axis in self.config.capture_axes
        )
        if any(len(axis) == 0 for axis in self._capture_axes):
            raise ValueError("Encoder cudagraph capture axes must be non-empty.")

        if len(self.config.paths) > 1 and self._capture_axes:
            raise NotImplementedError(
                "Combining dual-path encoder CUDA graphs with capture axes "
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
                self.token_budgets = self._generate_budgets(
                    effective_min,
                    max_budget,
                    max_token_budget,
                )
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
                self.token_budgets = self._generate_budgets(
                    min_budget,
                    max_budget,
                    max_token_budget,
                )
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

        self.budget_graphs: dict[str, dict[BudgetGraphMapKey, BudgetGraphMetadata]] = {}
        self.graph_pool: Any | None = None
        # One set of max-size input buffers and one max-size output buffer
        # per (path, axis_keys) combination, shared by all budgets of that
        # combination as prefix views. Populated at the first
        # (largest-budget) capture of each combination.
        self._shared_input_buffers: dict[
            tuple[str, CaptureAxisKeys], dict[str, torch.Tensor]
        ] = {}
        self._shared_output_buffers: dict[
            tuple[str, CaptureAxisKeys], torch.Tensor
        ] = {}
        # Input keys whose shared buffer smaller budgets view. Other budgets'
        # captures overwrite it, so replay must refill these keys.
        self._shared_input_keys: dict[tuple[str, CaptureAxisKeys], set[str]] = {}
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
                # Paths with a declared min budget have fixed-size items
                # (e.g. DeepSeek-OCR's global path), so budget by exact
                # item multiples instead of the subdivided grid.
                budgets = self._item_multiple_budgets(min_path_budget, max_budget)
            if path_config.allow_zero_tokens:
                budgets.insert(0, 0)
            self.path_token_budgets[path] = budgets

        logger.info(
            "EncoderCudaGraphManager initialized with paths=%s, "
            "max_batch_size=%d, max_frames_per_batch=%s, use_dp=%s, "
            "capture_axes=%s",
            self.path_token_budgets,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.use_dp,
            self._capture_axes,
        )

    @staticmethod
    def _generate_budgets(
        min_budget: int,
        max_budget: int,
        cap: int | None = None,
    ) -> list[int]:
        """Generate token budgets from min_budget up to max_budget or cap.

        Quarter-octave steps up to 1024 tokens, steps of 128 up to 2560, then
        1/8-octave steps (rounded up to 128), so a replay pads at most ~25%
        below 1024, ~12% up to 2560 and ~9% above. Capture buffers are shared
        across budgets (see `_capture_budget_graph`), so extra graphs cost
        no buffer memory. Batches above the largest budget run eagerly.
        """
        if cap is not None:
            max_budget = min(max_budget, max(cap, min_budget))
        points = {min_budget, max_budget}
        base = 1
        while base < min(max_budget, 1024):
            for k in range(4):
                point = base + k * (base // 4)
                if min_budget <= point <= max_budget:
                    points.add(point)
            base *= 2
        linear_end = min(max_budget, _LINEAR_GRID_END)
        points.update(p for p in range(1024, linear_end + 1, 128) if p >= min_budget)
        point = _LINEAR_GRID_END
        while point < max_budget:
            point = -(-int(point * 2 ** (1 / 8)) // 128) * 128
            if min_budget <= point <= max_budget:
                points.add(point)
        return sorted(points)

    @staticmethod
    def _item_multiple_budgets(item_tokens: int, max_budget: int) -> list[int]:
        """Budgets for a path whose items all have item_tokens tokens.

        Every item count up to 16, then 1/8-octave steps, so batches of up to
        16 items fit their budget exactly.
        """
        max_items = max_budget // item_tokens
        counts = {c for c in range(1, 17) if c <= max_items} | {max_items}
        count = 16
        while count < max_items:
            count = math.ceil(count * 2 ** (1 / 8))
            counts.add(min(count, max_items))
        return [c * item_tokens for c in sorted(counts)]

    def supports_modality(self, modality: str) -> bool:
        """Check if a modality is supported by this manager."""
        return modality in self.config.modalities

    def is_captured(self) -> bool:
        """Return whether a CUDA graph pool is active."""
        return self.graph_pool is not None

    def clear(self) -> None:
        """Release captured encoder CUDA graphs and the manager-local pool."""
        for graph_set in self.budget_graphs.values():
            graph_set.clear()
        self.graph_pool = None
        self._shared_input_buffers.clear()
        self._shared_output_buffers.clear()
        self._shared_input_keys.clear()

    def capture(self, graph_pool: Any):
        """Capture CUDA graphs for every configured path and token budget."""
        if not self.config.modalities:
            logger.info("Encoder CUDA graph capture skipped: no supported modalities.")
            return

        self.graph_pool = graph_pool

        num_graphs = self.get_num_graphs_to_capture()
        if num_graphs > _CAPTURE_GRAPH_COUNT_WARNING_THRESHOLD:
            logger.warning(
                "Capturing %d encoder CUDA graphs; the subdivided default "
                "budget table and capture axes multiply the graph count. "
                "Consider explicit encoder_cudagraph_token_budgets or "
                "fewer capture axis keys.",
                num_graphs,
            )

        for path, budgets in self.path_token_budgets.items():
            for token_budget in sorted(budgets, reverse=True):
                if token_budget == 0:
                    continue
                for axis_keys in itertools.product(*self._capture_axes):
                    self._capture_budget_graph(
                        token_budget,
                        path=path,
                        axis_keys=axis_keys,
                    )

        logger.info(
            "Encoder CUDA graph capture complete. Captured %d graphs across %d paths.",
            num_graphs,
            len(self.path_token_budgets),
        )

    def _num_graphs_for_budgets(self, budgets: list[int]) -> int:
        num_budgets = sum(1 for budget in budgets if budget != 0)
        return num_budgets * math.prod(len(axis) for axis in self._capture_axes)

    def get_num_graphs_to_capture(self) -> int:
        if not self.config.modalities:
            return 0
        return sum(
            self._num_graphs_for_budgets(budgets)
            for budgets in self.path_token_budgets.values()
        )

    def _get_graph_set(
        self, path: str = "default"
    ) -> dict[BudgetGraphMapKey, BudgetGraphMetadata]:
        """Return the captured graphs for one encoder path."""
        if path not in self.budget_graphs:
            self.budget_graphs[path] = {}
        return self.budget_graphs[path]

    @staticmethod
    def _share_prefix_compatible(
        probe: dict[str, torch.Tensor],
        shared: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """View prefix-compatible probe buffers into the shared buffers.

        For each probe buffer that is a row-prefix of its shared counterpart
        (same trailing dims and dtype, no more rows), copy the probe contents
        into the shared buffer and return the prefix view instead. Anything
        else, including 0-dim buffers (which hold per-budget values and save
        no memory when shared), keeps its own per-budget allocation.
        """
        values = dict(probe)
        for key, buf in probe.items():
            full = shared.get(key)
            if full is None or buf.ndim == 0:
                continue
            if (
                full.ndim == buf.ndim
                and full.shape[1:] == buf.shape[1:]
                and full.dtype == buf.dtype
                and buf.shape[0] <= full.shape[0]
            ):
                view = full[: buf.shape[0]]
                view.copy_(buf)
                values[key] = view
        return values

    def _capture_budget_graph(
        self,
        token_budget: int,
        path: str = "default",
        axis_keys: CaptureAxisKeys = (),
    ):
        """Capture CUDA graph for a single token budget."""
        # Every frame yields at least one token, so a batch that fits the
        # budget has at most token_budget frames.
        max_frames_per_batch = min(self.max_frames_per_batch, token_budget)
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d, axis_keys=%s",
            token_budget,
            self.max_batch_size,
            max_frames_per_batch,
            axis_keys,
        )

        graph_set = self._get_graph_set(path)

        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            max_frames_per_batch,
            self.device,
            self.dtype,
            path,
            axis_keys,
        )

        # Budgets are captured largest-first, so the first capture of each
        # (path, axis_keys) group provides the shared buffers that later
        # captures of the group view into. Buffers whose shape or dtype is
        # not prefix-compatible keep their own per-budget allocation.
        share_key = (path, axis_keys)
        shared = self._shared_input_buffers.get(share_key)
        if shared is None:
            values = capture_inputs.values
            self._shared_input_buffers[share_key] = values
        else:
            values = self._share_prefix_compatible(capture_inputs.values, shared)
            self._shared_input_keys.setdefault(share_key, set()).update(
                key
                for key, buf in values.items()
                if buf is not capture_inputs.values[key]
            )

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            shared_out = self._shared_output_buffers.get(share_key)
            if shared_out is None or output.shape[0] > shared_out.shape[0]:
                shared_out = torch.empty_like(output)
                self._shared_output_buffers[share_key] = shared_out
            output_buffer = shared_out[: output.shape[0]]

        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(graph, pool=self.graph_pool, stream=current_stream()),
        ):
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
            output_buffer.copy_(output)

        graph_map_key: BudgetGraphMapKey = (
            token_budget if not axis_keys else (token_budget, axis_keys)
        )
        graph_set[graph_map_key] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            graph=graph,
            input_buffers=values,
            output_buffer=output_buffer,
            axis_keys=axis_keys,
        )

    def _find_smallest_fitting_budget_given_tokens(
        self, total_tokens: int, budgets: list[int] | None = None
    ) -> int | None:
        """Find smallest budget >= total_tokens, or None if none fits."""
        budgets = budgets if budgets is not None else self.token_budgets
        return round_up_to(budgets, total_tokens)

    def _path_fitting_budget(self, path: str, tokens: int) -> int | None:
        """Smallest captured budget on a path; 0 when idle, None if oversized."""
        if tokens == 0:
            return 0
        return self._find_smallest_fitting_budget_given_tokens(
            tokens, self.path_token_budgets[path]
        )

    def _get_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        """Get item specs from the model."""
        # Implementations read per-item grid/patch counts off device tensors
        # to size the cudagraph buffers, so the D2H is inherent here.
        with gpu_sync_allowed():
            return self.model.get_encoder_cudagraph_item_specs(mm_kwargs)

    def _select_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> tuple[dict[str, Any], CaptureAxisKeys]:
        """Select the mm kwargs for `indices` from the model.

        Returns the sliced mm_kwargs together with the resolved capture-axis
        keys (popped out of the returned dict; empty without capture axes).
        """
        # Same as `_get_item_specs`: implementations re-read the per-item
        # grid/patch counts to slice the batch, so the D2H is inherent.
        with gpu_sync_allowed():
            selected = self.model.select_encoder_cudagraph_items(mm_kwargs, indices)
        axis_keys = selected.pop(ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG, ())
        if len(axis_keys) != len(self._capture_axes):
            raise ValueError(
                f"Model returned {len(axis_keys)} capture axis keys, "
                f"expected {len(self._capture_axes)} "
                f"(one per axis of capture_axes)."
            )
        return selected, axis_keys

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
        axis_keys: CaptureAxisKeys = (),
    ) -> torch.Tensor | None:
        """Execute budget graph.

        Args:
            mm_kwargs: Multimodal inputs for the batch.
            token_budget: Token budget to use.
            path: Configured encoder path.
            axis_keys: Resolved capture-axis keys for this batch.

        Returns:
            Encoder outputs, or None if graph not captured.

        """
        graph_set = self._get_graph_set(path)
        num_items = len(self._get_item_specs(mm_kwargs))

        graph_map_key: BudgetGraphMapKey = (
            token_budget if not axis_keys else (token_budget, axis_keys)
        )
        if graph_map_key not in graph_set:
            self.graph_misses += num_items
            return None

        graph_meta = graph_set[graph_map_key]

        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            graph_meta.max_frames_per_batch,
            path,
        )

        # Copy replay values into the buffers recorded for this path.
        shared_keys = self._shared_input_keys.get((path, graph_meta.axis_keys), ())
        for key, buf in graph_meta.input_buffers.items():
            src = replay.values.get(key)
            if src is None:
                if key in shared_keys:
                    raise RuntimeError(
                        f"Encoder CUDA graph buffer {key!r} is shared across "
                        "token budgets, so its capture-time contents may come "
                        "from another budget; replay must provide its values."
                    )
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
        """Execute the local batch with at most one replay per path."""
        item_specs = self._get_item_specs(mm_kwargs)
        num_items = len(item_specs)
        paths = tuple(self.path_token_budgets)
        per_item_path_tokens = {
            path: [spec.get_path_output_tokens(path) for spec in item_specs]
            for path in paths
        }
        per_item_out_tokens = [spec.output_tokens for spec in item_specs]

        if num_items == 0:
            return []
        # Largest items first: varlen attention schedules long sequences first,
        # which balances its tiles better on mixed batches.
        by_size = sorted(
            range(num_items), key=lambda i: per_item_out_tokens[i], reverse=True
        )
        graph_indices = [i for i in by_size if not item_specs[i].eager_only]
        eager_indices = [i for i in by_size if item_specs[i].eager_only]
        # One replay per path: the smallest budget that holds the whole batch.
        # A batch beyond the captured range runs eagerly in one forward.
        fits = len(graph_indices) <= self.max_batch_size

        outputs_by_orig_idx: dict[int, torch.Tensor] = {}
        for batch_indices, use_graphs in (
            (graph_indices, fits),
            (eager_indices, False),
        ):
            if not batch_indices:
                continue
            batch_mm_kwargs, axis_keys = self._select_items(mm_kwargs, batch_indices)
            graph_outputs: dict[str, torch.Tensor] = {}
            all_eager = True

            for path in paths:
                path_tokens = sum(per_item_path_tokens[path][i] for i in batch_indices)
                if path_tokens == 0:
                    continue

                token_budget = (
                    self._path_fitting_budget(path, path_tokens) if use_graphs else None
                )
                if token_budget is None:
                    with torch.inference_mode():
                        output = self.model.encoder_eager_forward(
                            batch_mm_kwargs, path=path
                        )
                else:
                    all_eager = False
                    graph_output = self._run_budget_graph(
                        batch_mm_kwargs,
                        token_budget,
                        path=path,
                        axis_keys=axis_keys,
                    )
                    assert graph_output is not None
                    output = graph_output
                graph_outputs[path] = output

            if all_eager:
                self.graph_misses += len(batch_indices)

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
            local_mm_kwargs = self._select_items(mm_kwargs, local_indices)[0]
        else:
            local_mm_kwargs = self._select_items(mm_kwargs, [])[0]

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
