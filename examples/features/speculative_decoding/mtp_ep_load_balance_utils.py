# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = 2

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512, 1024)
DEFAULT_DRAFT_LENGTHS = (0, 2, 4, 6)
DEFAULT_LAYERS = (0, 9, 19, 29, 39)
DEFAULT_NUM_EXPERTS = 256
DEFAULT_MAX_TOKENS = 128
DEFAULT_MAX_MODEL_LEN = 4096


@dataclass(frozen=True)
class CapturedStep:
    step_kind: str
    routing_data: np.ndarray
    total_scheduled_tokens: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepTiming:
    total_ms: float
    attention_ms: float
    routing_ms: float
    prepare_ms: float
    finalize_ms: float
    ffn_ms: float

    @property
    def all2all_ms(self) -> float:
        return self.prepare_ms + self.finalize_ms

    @property
    def prepare_finalize_ms(self) -> float:
        return self.all2all_ms

    @property
    def unattributed_ms(self) -> float:
        return (
            self.total_ms
            - self.attention_ms
            - self.routing_ms
            - self.all2all_ms
            - self.ffn_ms
        )


def should_capture_baseline_decode_step(scheduler_output: Any) -> bool:
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
    scheduled_spec_tokens = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        {},
    )
    return (
        bool(num_scheduled_tokens)
        and not scheduled_spec_tokens
        and all(num_tokens == 1 for num_tokens in num_scheduled_tokens.values())
    )


def should_capture_mtp_verification_step(scheduler_output: Any) -> bool:
    scheduled_spec_tokens = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        {},
    )
    return bool(scheduled_spec_tokens)


def _request_offsets(
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for req_id in req_ids:
        if req_id not in num_scheduled_tokens:
            continue
        offsets[req_id] = offset
        offset += num_scheduled_tokens[req_id]
    return offsets


def select_step_routing_data(
    scheduler_output: Any,
    model_runner_output: Any,
    use_spec_decode: bool,
) -> CapturedStep | None:
    routed_experts = getattr(model_runner_output, "routed_experts", None)
    if routed_experts is None:
        return None

    routing_data = np.asarray(routed_experts.routing_data)
    req_ids = list(getattr(model_runner_output, "req_ids", []))
    num_scheduled_tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    expected_rows = sum(num_scheduled_tokens.get(req_id, 0) for req_id in req_ids)
    if routing_data.shape[0] != expected_rows:
        raise ValueError(
            "routing_data rows do not match the scheduled token layout: "
            f"{routing_data.shape[0]} vs {expected_rows}."
        )

    if use_spec_decode:
        if not should_capture_mtp_verification_step(scheduler_output):
            return None
        offsets = _request_offsets(req_ids, num_scheduled_tokens)
        selected_req_ids = tuple(
            req_id
            for req_id in req_ids
            if req_id in scheduler_output.scheduled_spec_decode_tokens
        )
        if not selected_req_ids:
            return None

        selected_segments = []
        total_scheduled_tokens = 0
        for req_id in selected_req_ids:
            segment_offset = offsets[req_id]
            segment_len = num_scheduled_tokens[req_id]
            selected_segments.append(
                routing_data[segment_offset : segment_offset + segment_len]
            )
            total_scheduled_tokens += segment_len

        return CapturedStep(
            step_kind="mtp_verification",
            routing_data=np.concatenate(selected_segments, axis=0),
            total_scheduled_tokens=total_scheduled_tokens,
            request_ids=selected_req_ids,
        )

    if not should_capture_baseline_decode_step(scheduler_output):
        return None

    return CapturedStep(
        step_kind="baseline_decode",
        routing_data=routing_data,
        total_scheduled_tokens=routing_data.shape[0],
        request_ids=tuple(req_ids),
    )


def count_layer_expert_histograms(
    routing_data: np.ndarray,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    num_experts: int = DEFAULT_NUM_EXPERTS,
) -> np.ndarray:
    if routing_data.ndim != 3:
        raise ValueError(
            "routing_data must be a rank-3 array shaped as "
            "(num_tokens, num_layers, topk)."
        )

    histograms = np.zeros((len(layers), num_experts), dtype=np.int64)
    for row_idx, layer_idx in enumerate(layers):
        layer_assignments = routing_data[:, layer_idx, :].reshape(-1)
        histograms[row_idx] = np.bincount(
            layer_assignments,
            minlength=num_experts,
        )[:num_experts]
    return histograms


def average_step_histograms(step_histograms: np.ndarray) -> np.ndarray:
    if step_histograms.size == 0:
        raise ValueError("step_histograms must contain at least one captured step.")
    return step_histograms.mean(axis=0)


def sort_experts_desc(avg_histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_expert_ids = np.argsort(-avg_histograms, axis=1, kind="stable")
    sorted_histograms = np.take_along_axis(
        avg_histograms,
        sorted_expert_ids,
        axis=1,
    )
    return sorted_histograms, sorted_expert_ids


def reorder_histograms_by_expert_order(
    avg_histograms: np.ndarray,
    expert_order: np.ndarray,
) -> np.ndarray:
    return np.take_along_axis(avg_histograms, expert_order, axis=1)


def aggregate_worker_step_timings(
    worker_timings: list[dict[str, float] | None],
) -> StepTiming:
    valid_timings = [timing for timing in worker_timings if timing is not None]
    if not valid_timings:
        raise ValueError("Expected at least one worker timing bundle.")
    return StepTiming(
        total_ms=max(timing["total_ms"] for timing in valid_timings),
        attention_ms=max(timing["attention_ms"] for timing in valid_timings),
        routing_ms=max(timing["routing_ms"] for timing in valid_timings),
        prepare_ms=max(timing["prepare_ms"] for timing in valid_timings),
        finalize_ms=max(timing["finalize_ms"] for timing in valid_timings),
        ffn_ms=max(timing["ffn_ms"] for timing in valid_timings),
    )


def compute_balancedness(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    max_count = counts.max(initial=0.0)
    if max_count <= 0:
        return 1.0
    return float(counts.mean() / max_count)


def compute_gini(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    total = counts.sum()
    if total <= 0:
        return 0.0

    n = counts.size
    index = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * index - n - 1) * counts)
    return float(numerator / (n * total))


def classify_imbalance_change(
    balancedness_delta: float,
    gini_delta: float,
    tol: float = 1e-12,
) -> str:
    if abs(balancedness_delta) <= tol and abs(gini_delta) <= tol:
        return "unchanged"
    if balancedness_delta < -tol and gini_delta >= -tol:
        return "worsened"
    if balancedness_delta > tol and gini_delta <= tol:
        return "improved"
    return "mixed"


def build_condition_metrics(
    *,
    batch_size: int,
    draft_length: int,
    num_steps: int,
    layers: tuple[int, ...],
    avg_histograms: np.ndarray,
    baseline_histograms: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_row, layer_idx in enumerate(layers):
        avg_counts = avg_histograms[layer_row]
        baseline_counts = baseline_histograms[layer_row]

        avg_total = float(avg_counts.sum())
        baseline_total = float(baseline_counts.sum())
        balancedness = compute_balancedness(avg_counts)
        baseline_balancedness = compute_balancedness(baseline_counts)
        gini = compute_gini(avg_counts)
        baseline_gini = compute_gini(baseline_counts)

        rows.append(
            {
                "batch_size": batch_size,
                "draft_length": draft_length,
                "layer": layer_idx,
                "num_steps": num_steps,
                "avg_total_routed_assignments_per_step": avg_total,
                "baseline_avg_total_routed_assignments_per_step": baseline_total,
                "avg_total_routed_assignments_delta": avg_total - baseline_total,
                "balancedness": balancedness,
                "baseline_balancedness": baseline_balancedness,
                "balancedness_delta": balancedness - baseline_balancedness,
                "balancedness_relative_change": (
                    balancedness / baseline_balancedness - 1.0
                    if baseline_balancedness > 0
                    else 0.0
                ),
                "gini": gini,
                "baseline_gini": baseline_gini,
                "gini_delta": gini - baseline_gini,
                "imbalance_change": classify_imbalance_change(
                    balancedness - baseline_balancedness,
                    gini - baseline_gini,
                ),
            }
        )
    return rows


def select_dataset_indices(batch_size: int, available_items: int) -> np.ndarray:
    if batch_size > available_items:
        raise ValueError(
            f"Requested batch_size={batch_size}, but only {available_items} "
            "dataset items are available."
        )
    return np.arange(batch_size, dtype=np.int64)


def build_speedup_rows(
    latency_by_condition: dict[tuple[int, int], float],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        baseline_latency = latency_by_condition[(batch_size, 0)]
        for draft_length in draft_lengths:
            latency_ms = latency_by_condition[(batch_size, draft_length)]
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "condition_latency_ms": latency_ms,
                    "baseline_latency_ms": baseline_latency,
                    "speedup": (
                        baseline_latency / latency_ms if latency_ms > 0 else 0.0
                    ),
                }
            )
    return rows


def summarize_step_time_components(
    step_total_ms: np.ndarray,
    step_attention_ms: np.ndarray,
    step_routing_ms: np.ndarray,
    step_prepare_ms: np.ndarray,
    step_finalize_ms: np.ndarray,
    step_ffn_ms: np.ndarray,
) -> dict[str, float]:
    avg_total_ms = float(np.mean(step_total_ms))
    avg_attention_ms = float(np.mean(step_attention_ms))
    avg_routing_ms = float(np.mean(step_routing_ms))
    avg_prepare_ms = float(np.mean(step_prepare_ms))
    avg_finalize_ms = float(np.mean(step_finalize_ms))
    avg_all2all_ms = avg_prepare_ms + avg_finalize_ms
    avg_ffn_ms = float(np.mean(step_ffn_ms))
    return {
        "avg_step_total_ms": avg_total_ms,
        "avg_attention_ms": avg_attention_ms,
        "avg_routing_ms": avg_routing_ms,
        "avg_prepare_ms": avg_prepare_ms,
        "avg_finalize_ms": avg_finalize_ms,
        "avg_all2all_ms": avg_all2all_ms,
        "avg_ffn_ms": avg_ffn_ms,
    }


def normalize_time_components(
    summary_row: dict[str, float | int],
    baseline_total_ms: float,
) -> dict[str, float]:
    if baseline_total_ms <= 0:
        raise ValueError("baseline_total_ms must be positive.")
    return {
        "normalized_attention_ms": (
            float(summary_row["avg_attention_ms"]) / baseline_total_ms
        ),
        "normalized_routing_ms": (
            float(summary_row["avg_routing_ms"]) / baseline_total_ms
        ),
        "normalized_all2all_ms": (
            float(summary_row["avg_all2all_ms"]) / baseline_total_ms
        ),
        "normalized_ffn_ms": (
            float(summary_row["avg_ffn_ms"]) / baseline_total_ms
        ),
        "ffn_share": (
            float(summary_row["avg_ffn_ms"])
            / float(summary_row["avg_step_total_ms"])
            if float(summary_row["avg_step_total_ms"]) > 0
            else 0.0
        ),
    }
