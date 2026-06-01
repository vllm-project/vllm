# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = 5

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "likaixin/InstructCoder"
DEFAULT_DATASET_CONFIG = None
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512)
DEFAULT_DRAFT_LENGTHS = (0, 2, 4, 6)
DEFAULT_LAYERS = (0, 9, 19, 29, 39)
DEFAULT_NUM_EXPERTS = 256
DEFAULT_MAX_TOKENS = 128
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_NUM_SAMPLES = 512
TPOT_DEFINITION = (
    "tpot_ms = built_in_decode_time_total_ms / "
    "sum(max(num_generation_tokens - 1, 0))"
)


@dataclass(frozen=True)
class CapturedStep:
    step_kind: str
    routing_data: np.ndarray
    total_scheduled_tokens: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepCaptureDecision:
    captured_step: CapturedStep | None
    local_step_kind: str
    drop_reason: str | None = None


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


@dataclass(frozen=True)
class GlobalStepTimingAggregation:
    global_step_indices: np.ndarray
    global_step_total_ms: np.ndarray
    global_step_ffn_ms: np.ndarray
    global_step_other_ms: np.ndarray
    global_step_kinds: np.ndarray
    global_step_histograms: np.ndarray
    global_step_total_tokens: np.ndarray
    num_global_candidate_steps: int
    num_global_captured_steps: int
    num_global_prefill_dropped_steps: int
    num_global_mixed_dropped_steps: int
    num_global_non_target_dropped_steps: int


@dataclass(frozen=True)
class FinishedRequestStatTotals:
    decode_time_total_ms: float
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int


def classify_step_capture(
    scheduler_output: Any,
    model_runner_output: Any,
    worker_step_metadata: dict[str, Any] | None,
    use_spec_decode: bool,
) -> StepCaptureDecision:
    routed_experts = getattr(model_runner_output, "routed_experts", None)
    if routed_experts is None:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="missing_routing",
            drop_reason="missing_routing",
        )

    routing_data = np.asarray(routed_experts.routing_data)
    req_ids = tuple(getattr(model_runner_output, "req_ids", ()))
    num_scheduled_tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    expected_rows = sum(num_scheduled_tokens.get(req_id, 0) for req_id in req_ids)
    if routing_data.shape[0] != expected_rows:
        raise ValueError(
            "routing_data rows do not match the scheduled token layout: "
            f"{routing_data.shape[0]} vs {expected_rows}."
        )

    scheduled_spec_tokens = dict(
        getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
    )
    metadata_req_ids = req_ids
    has_prefill = False
    if worker_step_metadata is not None:
        metadata_req_ids = tuple(worker_step_metadata.get("req_ids", ()))
        if metadata_req_ids and metadata_req_ids != req_ids:
            raise ValueError(
                "worker step metadata req_ids do not match model runner output: "
                f"{metadata_req_ids} vs {req_ids}."
            )
        has_prefill = bool(worker_step_metadata.get("has_prefill", False))

    if has_prefill:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="prefill",
            drop_reason="prefill",
        )

    if use_spec_decode:
        if not scheduled_spec_tokens:
            return StepCaptureDecision(
                captured_step=None,
                local_step_kind="non_target",
                drop_reason="non_target",
            )
        if set(req_ids) != set(scheduled_spec_tokens):
            return StepCaptureDecision(
                captured_step=None,
                local_step_kind="mixed",
                drop_reason="mixed",
            )
        return StepCaptureDecision(
            captured_step=CapturedStep(
                step_kind="verification_only",
                routing_data=routing_data,
                total_scheduled_tokens=expected_rows,
                request_ids=req_ids,
            ),
            local_step_kind="verification_only",
        )

    if scheduled_spec_tokens:
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="mixed",
            drop_reason="mixed",
        )
    if not num_scheduled_tokens or not all(
        num_tokens == 1 for num_tokens in num_scheduled_tokens.values()
    ):
        return StepCaptureDecision(
            captured_step=None,
            local_step_kind="non_target",
            drop_reason="non_target",
        )
    return StepCaptureDecision(
        captured_step=CapturedStep(
            step_kind="decode_only",
            routing_data=routing_data,
            total_scheduled_tokens=routing_data.shape[0],
            request_ids=req_ids,
        ),
        local_step_kind="decode_only",
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


def select_step_routing_data(
    scheduler_output: Any,
    model_runner_output: Any,
    use_spec_decode: bool,
) -> CapturedStep | None:
    return classify_step_capture(
        scheduler_output,
        model_runner_output,
        worker_step_metadata=None,
        use_spec_decode=use_spec_decode,
    ).captured_step


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
            f"Requested num_samples={batch_size}, but only {available_items} "
            "dataset items are available."
        )
    return np.arange(batch_size, dtype=np.int64)


def num_condition_rounds(num_samples: int, global_batch_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive.")
    return math.ceil(num_samples / global_batch_size)


def shard_global_batch_indices(
    *,
    num_samples: int,
    global_batch_size: int,
    round_idx: int,
    dp_size: int,
    dp_rank: int,
) -> np.ndarray:
    if dp_size <= 0:
        raise ValueError("dp_size must be positive.")
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} must be in [0, {dp_size}).")
    start = round_idx * global_batch_size
    stop = min(start + global_batch_size, num_samples)
    if start >= stop:
        return np.empty((0,), dtype=np.int64)
    round_indices = np.arange(start, stop, dtype=np.int64)
    floor = len(round_indices) // dp_size
    remainder = len(round_indices) % dp_size
    local_start = dp_rank * floor + min(dp_rank, remainder)
    local_len = floor + (1 if dp_rank < remainder else 0)
    return round_indices[local_start : local_start + local_len]


def build_speedup_rows(
    decode_time_ms_by_condition: dict[tuple[int, int], float],
    generation_tokens_by_condition: dict[tuple[int, int], int],
    output_tokens_excl_first_by_condition: dict[tuple[int, int], int],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        baseline_decode_time_ms = decode_time_ms_by_condition[(batch_size, 0)]
        baseline_generation_tokens = generation_tokens_by_condition[(batch_size, 0)]
        baseline_output_tokens_excl_first = output_tokens_excl_first_by_condition[
            (batch_size, 0)
        ]
        baseline_tpot = compute_tpot_ms_from_finished_stats(
            baseline_decode_time_ms,
            baseline_output_tokens_excl_first,
        )
        baseline_decode_throughput = compute_decode_throughput_tok_s(
            baseline_generation_tokens,
            baseline_decode_time_ms,
        )
        for draft_length in draft_lengths:
            decode_time_total_ms = decode_time_ms_by_condition[
                (batch_size, draft_length)
            ]
            num_generation_tokens_total = generation_tokens_by_condition[
                (batch_size, draft_length)
            ]
            num_output_tokens_excl_first_total = (
                output_tokens_excl_first_by_condition[(batch_size, draft_length)]
            )
            tpot_ms = compute_tpot_ms_from_finished_stats(
                decode_time_total_ms,
                num_output_tokens_excl_first_total,
            )
            decode_throughput = compute_decode_throughput_tok_s(
                num_generation_tokens_total,
                decode_time_total_ms,
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "decode_time_total_ms": decode_time_total_ms,
                    "num_generation_tokens_total": num_generation_tokens_total,
                    "num_output_tokens_excl_first_total": (
                        num_output_tokens_excl_first_total
                    ),
                    "tpot_ms": tpot_ms,
                    "baseline_tpot_ms": baseline_tpot,
                    "tpot_speedup": (
                        baseline_tpot / tpot_ms if tpot_ms > 0 else 0.0
                    ),
                    "decode_throughput_tok_s": decode_throughput,
                    "baseline_decode_throughput_tok_s": baseline_decode_throughput,
                    "decode_throughput_speedup": (
                        decode_throughput / baseline_decode_throughput
                        if baseline_decode_throughput > 0
                        else 0.0
                    ),
                }
            )
    return rows


def aggregate_global_step_time_components(
    rank_step_data: list[dict[str, np.ndarray]],
    *,
    data_parallel_size: int,
    expected_step_kind: str,
    layers: tuple[int, ...],
    num_experts: int,
    tol_ms: float = 1e-3,
) -> GlobalStepTimingAggregation:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive.")
    if len(rank_step_data) != data_parallel_size:
        raise ValueError(
            "rank_step_data length must match data_parallel_size: "
            f"{len(rank_step_data)} vs {data_parallel_size}."
        )

    per_step_records: dict[int, list[dict[str, Any]]] = {}
    for rank_idx, rank_data in enumerate(rank_step_data):
        step_indices = np.asarray(
            rank_data["candidate_first_ep_collective_seq_ids"], dtype=np.int64
        )
        step_kinds = np.asarray(rank_data["candidate_step_kinds"], dtype=np.str_)
        step_total_ms = np.asarray(
            rank_data["candidate_step_total_ms"], dtype=np.float64
        )
        step_ffn_ms = np.asarray(rank_data["candidate_step_ffn_ms"], dtype=np.float64)
        step_total_tokens = np.asarray(
            rank_data["candidate_step_total_tokens"], dtype=np.int64
        )
        step_histograms = np.asarray(rank_data["candidate_step_histograms"])
        size = step_indices.shape[0]
        for array_name, array in (
            ("candidate_step_kinds", step_kinds),
            ("candidate_step_total_ms", step_total_ms),
            ("candidate_step_ffn_ms", step_ffn_ms),
            ("candidate_step_total_tokens", step_total_tokens),
        ):
            if array.shape[0] != size:
                raise ValueError(
                    f"Rank {rank_idx} {array_name} has inconsistent length: "
                    f"{array.shape[0]} vs {size}."
                )
        if step_histograms.shape != (size, len(layers), num_experts):
            raise ValueError(
                f"Rank {rank_idx} candidate_step_histograms has shape "
                f"{step_histograms.shape}; expected "
                f"{(size, len(layers), num_experts)}."
            )

        seen_step_indices: set[int] = set()
        for idx in range(size):
            step_index = int(step_indices[idx])
            if step_index < 0:
                continue
            if step_index in seen_step_indices:
                raise ValueError(
                    f"Rank {rank_idx} produced duplicate first_ep_collective_seq_id="
                    f"{step_index}."
                )
            seen_step_indices.add(step_index)

            record = {
                "rank_idx": rank_idx,
                "step_kind": str(step_kinds[idx]),
                "step_total_ms": float(step_total_ms[idx]),
                "step_ffn_ms": float(step_ffn_ms[idx]),
                "step_total_tokens": int(step_total_tokens[idx]),
                "step_histograms": step_histograms[idx],
            }
            per_step_records.setdefault(step_index, []).append(record)

    global_step_indices: list[int] = []
    global_step_total_ms: list[float] = []
    global_step_ffn_ms: list[float] = []
    global_step_other_ms: list[float] = []
    global_step_kinds: list[str] = []
    global_step_histograms: list[np.ndarray] = []
    global_step_total_tokens: list[int] = []
    num_global_prefill_dropped_steps = 0
    num_global_mixed_dropped_steps = 0
    num_global_non_target_dropped_steps = 0

    for step_index in sorted(per_step_records):
        records = per_step_records[step_index]
        if len(records) != data_parallel_size:
            num_global_non_target_dropped_steps += 1
            continue

        step_kinds = {str(record["step_kind"]) for record in records}
        if "prefill" in step_kinds:
            num_global_prefill_dropped_steps += 1
            continue
        if "mixed" in step_kinds:
            num_global_mixed_dropped_steps += 1
            continue
        if step_kinds != {expected_step_kind}:
            num_global_non_target_dropped_steps += 1
            continue

        total_ms = max(float(record["step_total_ms"]) for record in records)
        ffn_ms = max(float(record["step_ffn_ms"]) for record in records)
        other_ms = total_ms - ffn_ms
        if ffn_ms < 0:
            raise ValueError(
                f"Captured first_ep_collective_seq_id={step_index} produced "
                f"negative FFN time: {ffn_ms:.6f} ms."
            )
        if other_ms < -tol_ms:
            raise ValueError(
                "Captured first_ep_collective_seq_id="
                f"{step_index} produced negative Other time: {other_ms:.6f} ms."
            )

        global_step_indices.append(step_index)
        global_step_total_ms.append(total_ms)
        global_step_ffn_ms.append(ffn_ms)
        global_step_other_ms.append(other_ms)
        global_step_kinds.append(expected_step_kind)
        global_step_histograms.append(
            np.sum([record["step_histograms"] for record in records], axis=0)
        )
        global_step_total_tokens.append(
            sum(int(record["step_total_tokens"]) for record in records)
        )

    if global_step_histograms:
        histogram_array = np.stack(global_step_histograms, axis=0).astype(np.int64)
    else:
        histogram_array = np.empty((0, len(layers), num_experts), dtype=np.int64)

    return GlobalStepTimingAggregation(
        global_step_indices=np.asarray(global_step_indices, dtype=np.int64),
        global_step_total_ms=np.asarray(global_step_total_ms, dtype=np.float64),
        global_step_ffn_ms=np.asarray(global_step_ffn_ms, dtype=np.float64),
        global_step_other_ms=np.asarray(global_step_other_ms, dtype=np.float64),
        global_step_kinds=np.asarray(global_step_kinds, dtype=np.str_),
        global_step_histograms=histogram_array,
        global_step_total_tokens=np.asarray(global_step_total_tokens, dtype=np.int64),
        num_global_candidate_steps=len(per_step_records),
        num_global_captured_steps=len(global_step_indices),
        num_global_prefill_dropped_steps=num_global_prefill_dropped_steps,
        num_global_mixed_dropped_steps=num_global_mixed_dropped_steps,
        num_global_non_target_dropped_steps=num_global_non_target_dropped_steps,
    )


def summarize_global_step_time_components(
    step_total_ms: np.ndarray,
    step_ffn_ms: np.ndarray,
    step_other_ms: np.ndarray,
) -> dict[str, float]:
    avg_total_ms = float(np.mean(step_total_ms))
    avg_ffn_ms = float(np.mean(step_ffn_ms))
    avg_other_ms = float(np.mean(step_other_ms))
    return {
        "avg_step_total_ms": avg_total_ms,
        "avg_ffn_ms": avg_ffn_ms,
        "avg_other_ms": avg_other_ms,
    }


def normalize_global_time_components(
    summary_row: dict[str, float | int],
    baseline_total_ms: float,
) -> dict[str, float]:
    if baseline_total_ms <= 0:
        raise ValueError("baseline_total_ms must be positive.")
    return {
        "normalized_ffn_ms": (
            float(summary_row["avg_ffn_ms"]) / baseline_total_ms
        ),
        "normalized_other_ms": (
            float(summary_row["avg_other_ms"]) / baseline_total_ms
        ),
        "ffn_share": (
            float(summary_row["avg_ffn_ms"])
            / float(summary_row["avg_step_total_ms"])
            if float(summary_row["avg_step_total_ms"]) > 0
            else 0.0
        ),
        "other_share": (
            float(summary_row["avg_other_ms"])
            / float(summary_row["avg_step_total_ms"])
            if float(summary_row["avg_step_total_ms"]) > 0
            else 0.0
        ),
    }


def compute_num_output_tokens_excluding_first(
    output_lengths: np.ndarray,
) -> int:
    output_lengths = np.asarray(output_lengths, dtype=np.int64)
    return int(np.maximum(output_lengths - 1, 0).sum())


def compute_tpot_ms(
    decode_only_total_ms: float,
    output_lengths: np.ndarray,
) -> float:
    num_output_tokens_excl_first = compute_num_output_tokens_excluding_first(
        output_lengths
    )
    if num_output_tokens_excl_first <= 0:
        return 0.0
    return decode_only_total_ms / num_output_tokens_excl_first


def compute_tpot_ms_from_finished_stats(
    decode_time_total_ms: float,
    num_output_tokens_excl_first_total: int,
) -> float:
    if num_output_tokens_excl_first_total <= 0:
        return 0.0
    return decode_time_total_ms / num_output_tokens_excl_first_total


def compute_decode_throughput_tok_s(
    num_generation_tokens_total: int,
    decode_time_total_ms: float,
) -> float:
    if decode_time_total_ms <= 0:
        return 0.0
    return num_generation_tokens_total / (decode_time_total_ms / 1000.0)


def build_expert_to_ep_rank(
    *,
    num_experts: int,
    ep_size: int,
    placement_strategy: str = "linear",
) -> np.ndarray:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    expert_to_rank = np.full((num_experts,), -1, dtype=np.int64)
    if placement_strategy == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        for ep_rank in range(ep_size):
            local_count = base + (1 if ep_rank < remainder else 0)
            start = ep_rank * base + min(ep_rank, remainder)
            expert_to_rank[start : start + local_count] = ep_rank
    elif placement_strategy == "round_robin":
        for expert_id in range(num_experts):
            expert_to_rank[expert_id] = expert_id % ep_size
    else:
        raise ValueError(f"Unsupported expert placement strategy: {placement_strategy}")
    if np.any(expert_to_rank < 0):
        raise ValueError("Expert to EP rank mapping is incomplete.")
    return expert_to_rank


def merge_expert_to_ep_rank_maps(
    rank_maps: list[np.ndarray],
    *,
    num_experts: int,
    ep_size: int,
) -> np.ndarray:
    if not rank_maps:
        return build_expert_to_ep_rank(num_experts=num_experts, ep_size=ep_size)
    merged = np.full((num_experts,), -1, dtype=np.int64)
    for rank_map in rank_maps:
        rank_map = np.asarray(rank_map, dtype=np.int64)
        if rank_map.shape != (num_experts,):
            raise ValueError(
                f"expert_to_ep_rank map has shape {rank_map.shape}; "
                f"expected {(num_experts,)}."
            )
        owned = rank_map >= 0
        conflicts = owned & (merged >= 0) & (merged != rank_map)
        if np.any(conflicts):
            conflict_ids = np.flatnonzero(conflicts)[:8].tolist()
            raise ValueError(
                "Conflicting expert to EP rank ownership for experts "
                f"{conflict_ids}."
            )
        merged[owned] = rank_map[owned]
    if np.any(merged < 0):
        fallback = build_expert_to_ep_rank(num_experts=num_experts, ep_size=ep_size)
        merged[merged < 0] = fallback[merged < 0]
    return merged


def build_rank_load_from_histograms(
    avg_histograms: np.ndarray,
    expert_to_ep_rank: np.ndarray,
    ep_size: int,
) -> np.ndarray:
    avg_histograms = np.asarray(avg_histograms, dtype=np.float64)
    expert_to_ep_rank = np.asarray(expert_to_ep_rank, dtype=np.int64)
    if avg_histograms.ndim != 2:
        raise ValueError("avg_histograms must be shaped as (layers, experts).")
    if expert_to_ep_rank.shape != (avg_histograms.shape[1],):
        raise ValueError(
            "expert_to_ep_rank length must match avg_histograms expert dimension."
        )
    rank_load = np.zeros((avg_histograms.shape[0], ep_size), dtype=np.float64)
    for expert_id, ep_rank in enumerate(expert_to_ep_rank):
        if not 0 <= int(ep_rank) < ep_size:
            raise ValueError(
                f"expert_id={expert_id} has invalid ep_rank={int(ep_rank)}."
            )
        rank_load[:, int(ep_rank)] += avg_histograms[:, expert_id]
    return rank_load
