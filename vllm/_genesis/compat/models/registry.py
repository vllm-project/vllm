# SPDX-License-Identifier: Apache-2.0
"""Genesis models registry — curated list of models we officially support.

Each entry declares:
  - HuggingFace id
  - Approximate size on disk
  - Quantization format
  - Compute requirements (min VRAM at TP=N)
  - Tested hardware classes
  - Tested launch configs (vllm pin, env flags, command-line args)
  - Expected metrics on each tested hardware (TPS, TTFT, VRAM)
  - Known quirks (operator-relevant gotchas)
  - Lifecycle status (PROD / SUPPORTED / EXPERIMENTAL / PLANNED)

The registry is the **single source of truth** for:

  - `genesis list-models` — show available models
  - `genesis pull <key>` — download with verified config
  - `genesis init` wizard — pick model based on detected hardware
  - `genesis bench --model <key>` — compare measured vs expected metrics

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpectedMetrics:
    """Reference metrics measured on a specific hardware class."""
    hardware_class: str        # e.g. "rtx_a5000_x2"
    wall_tps_median: float
    decode_tpot_ms: float
    ttft_ms: float
    vram_gb_per_rank: float
    tool_call_pass_rate: float  # 0..1
    captured_at: str            # ISO date


@dataclass(frozen=True)
class TestedConfig:
    """One blessed launch configuration for a model."""
    name: str                   # human-readable workload label
    vllm_pin: str               # exact vllm version + git commit
    tensor_parallel_size: int
    kv_cache_dtype: str         # "auto" | "fp8_e5m2" | "turboquant_k8v4" | etc.
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    speculative_config: dict[str, Any] | None
    enable_prefix_caching: bool
    additional_args: tuple[str, ...] = field(default_factory=tuple)
    recommended_genesis_patches: tuple[str, ...] = field(default_factory=tuple)
    expected: ExpectedMetrics | None = None


@dataclass(frozen=True)
class ModelEntry:
    """One entry in the SUPPORTED_MODELS registry."""
    key: str                    # short stable id (used in CLI)
    hf_id: str                  # huggingface.co/{hf_id}
    hf_revision: str | None     # commit SHA pin (None = latest main)
    title: str                  # human-readable name
    size_gb: float              # approx total weight size
    quant_format: str           # "fp8" | "autoround_int4" | "awq_int4" | ...
    model_class: str            # "qwen3" | "qwen3_5" | "qwen3_moe" | ...
    is_hybrid: bool
    is_moe: bool
    num_experts: int | None
    min_vram_gb_per_rank: dict[int, float]  # {tp: min_vram_per_rank_gb}
    tested_hardware: tuple[str, ...]  # ["rtx_a5000", "rtx_4090", ...]
    tested_configs: tuple[TestedConfig, ...]
    recommended_workloads: tuple[str, ...]  # ["long_ctx_tool_call", ...]
    license: str
    gated: bool                 # requires HF token + license acceptance
    credits: str
    quirks: tuple[str, ...] = field(default_factory=tuple)
    status: str = "SUPPORTED"   # "PROD" | "SUPPORTED" | "EXPERIMENTAL" | "PLANNED"
    notes: str = ""


# ─── The registry ────────────────────────────────────────────────────────


SUPPORTED_MODELS: dict[str, ModelEntry] = {
    "qwen3_6_27b_int4_autoround": ModelEntry(
        key="qwen3_6_27b_int4_autoround",
        hf_id="Intel/Qwen3.6-27B-A3B-int4-AutoRound",
        # Alternate community-maintained id: "Lorbus/Qwen3.6-27B-int4-AutoRound"
        # — same architecture + AutoRound recipe, different quant pass; both
        # validated by Genesis cross-rig (Sander on Lorbus, noonghunna on Lorbus,
        # webcodes-cz on Intel A3B variant). Defaulting to Intel A3B as the
        # canonical entry; users wanting Lorbus can pass --hf-id-override.
        hf_revision=None,  # community quant — track latest
        title="Qwen3.6-27B-A3B AutoRound INT4 (Lorbus)",
        size_gb=14.2,
        quant_format="autoround_int4",
        model_class="qwen3_5",
        is_hybrid=True,
        is_moe=True,
        num_experts=128,
        min_vram_gb_per_rank={1: 24.0, 2: 14.0},
        tested_hardware=("rtx_a5000", "rtx_4090", "rtx_5090"),
        tested_configs=(
            TestedConfig(
                name="long-ctx tool-call (PROD baseline)",
                vllm_pin="0.20.1rc1.dev16+g7a1eb8ac2",
                tensor_parallel_size=2,
                kv_cache_dtype="turboquant_k8v4",
                max_model_len=280000,
                max_num_seqs=2,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.90,
                speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                enable_prefix_caching=False,
                recommended_genesis_patches=(
                    "P58", "P60", "P60b", "P61", "P61b", "P62", "P64",
                    "P66", "P67", "P68", "P69", "P72", "P74",
                    "P85", "P87", "P91", "P98", "P99", "P100", "P101",
                    "PN8", "PN11", "PN12", "PN13", "PN14",
                ),
                expected=ExpectedMetrics(
                    hardware_class="rtx_a5000_x2",
                    wall_tps_median=103.3,
                    decode_tpot_ms=9.36,
                    ttft_ms=127.2,
                    vram_gb_per_rank=22.0,
                    tool_call_pass_rate=1.0,
                    captured_at="2026-04-29",  # v794 PROD validation
                ),
            ),
        ),
        recommended_workloads=("long_ctx_tool_call", "interactive"),
        license="apache-2.0",
        gated=False,
        credits="Intel quant team (AutoRound), base by Qwen team",
        quirks=(
            "Disable --enable-prefix-caching: triggers DS conv state layout "
            "error when MTP accept>1 (separate report; see "
            "feedback_27b_prefix_cache_fix.md).",
            "Use turboquant_k8v4 KV cache for +17% TPS over fp8_e5m2 "
            "(packed-slot cache locality; see feedback_turboquant_speed_bonus.md).",
        ),
        status="PROD",
        notes=(
            "Current Genesis PROD baseline (v794, validated 2026-04-29). "
            "Long-context (256K+) tool-call path. 5 PN-family patches "
            "engaged for memory + correctness."
        ),
    ),

    "qwen3_6_35b_a3b_fp8": ModelEntry(
        key="qwen3_6_35b_a3b_fp8",
        hf_id="Qwen/Qwen3.6-35B-A3B-FP8",
        hf_revision=None,
        title="Qwen3.6-35B-A3B FP8 (Minachist)",
        size_gb=38.0,
        quant_format="fp8",
        model_class="qwen3_5",
        is_hybrid=True,
        is_moe=True,
        num_experts=128,
        min_vram_gb_per_rank={1: 48.0, 2: 24.0},
        tested_hardware=("rtx_a5000", "h100"),
        tested_configs=(
            TestedConfig(
                name="35B PROD throughput (TP=2)",
                vllm_pin="0.20.0",
                tensor_parallel_size=2,
                kv_cache_dtype="fp8_e5m2",
                max_model_len=128000,
                max_num_seqs=2,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.92,
                speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                enable_prefix_caching=False,
                recommended_genesis_patches=(
                    "P3", "P5", "P6", "P15", "P22", "P26",
                    "P58", "P60", "P60b", "P61", "P61b", "P62", "P64",
                    "P66", "P67", "P68", "P69", "P72", "P74",
                    "P81", "P87", "P91", "P100", "P101",
                    "PN8", "PN11", "PN12", "PN13",
                ),
                expected=ExpectedMetrics(
                    hardware_class="rtx_a5000_x2",
                    wall_tps_median=183.27,
                    decode_tpot_ms=5.45,
                    ttft_ms=185.0,
                    vram_gb_per_rank=22.5,
                    tool_call_pass_rate=0.95,
                    captured_at="2026-04-28",  # v789 PROD
                ),
            ),
        ),
        recommended_workloads=("interactive", "throughput"),
        license="apache-2.0",
        gated=False,
        credits="Qwen team (Alibaba)",
        quirks=(
            "Pure FP8 — P81 (block-scaled MM low-M decode) gives +23% "
            "median decode on small batch sizes.",
        ),
        status="SUPPORTED",
        notes=(
            "Alternative PROD baseline (v759 era). Higher TPS than 27B "
            "but ~2× VRAM cost. Good for throughput-bound workloads."
        ),
    ),

    "qwen3_6_35b_a3b_int4_autoround": ModelEntry(
        key="qwen3_6_35b_a3b_int4_autoround",
        hf_id="Intel/Qwen3.6-35B-A3B-int4-AutoRound",
        hf_revision=None,
        title="Qwen3.6-35B-A3B AutoRound INT4",
        size_gb=18.5,
        quant_format="autoround_int4",
        model_class="qwen3_5",
        is_hybrid=True,
        is_moe=True,
        num_experts=128,
        min_vram_gb_per_rank={1: 32.0, 2: 18.0},
        tested_hardware=("rtx_a5000", "rtx_3090", "rtx_pro_6000_blackwell"),
        tested_configs=(
            TestedConfig(
                name="35B INT4 long-ctx (community-validated)",
                vllm_pin="0.20.1rc1.dev16+g7a1eb8ac2",
                tensor_parallel_size=2,
                kv_cache_dtype="turboquant_k8v4",
                max_model_len=128000,
                max_num_seqs=2,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.92,
                speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                enable_prefix_caching=False,
                recommended_genesis_patches=(
                    "P58", "P60", "P60b", "P67", "P87", "P91",
                    "PN8", "PN11", "PN12", "PN13", "PN14",
                ),
                expected=None,  # awaiting blessed measurement
            ),
        ),
        recommended_workloads=("long_ctx_tool_call",),
        license="apache-2.0",
        gated=False,
        credits="Intel quant team, base by Qwen team",
        quirks=(
            "P87 Marlin pad-on-load required: in_proj_ba shard at TP=2 has "
            "n=32 which fails MarlinLinearKernel.can_implement without P87.",
        ),
        status="SUPPORTED",
        notes=(
            "Larger sibling of qwen3_6_27b_int4_autoround. Same AutoRound "
            "quant scheme, 30% more parameters, ~50% larger VRAM footprint."
        ),
    ),

    "qwen3_6_27b_fp8_lmhead_fp8": ModelEntry(
        key="qwen3_6_27b_fp8_lmhead_fp8",
        hf_id="inferRouter/Qwen3.6-27B-FP8-lmhead-fp8",
        hf_revision=None,
        title="Qwen3.6-27B FP8 + lm_head FP8 (webcodes-cz)",
        size_gb=20.0,  # rough estimate — FP8 27B with FP8 lm_head
        quant_format="fp8",
        model_class="qwen3_5",
        is_hybrid=True,
        is_moe=True,
        num_experts=128,
        min_vram_gb_per_rank={1: 32.0},
        tested_hardware=("rtx_5090",),
        tested_configs=(
            TestedConfig(
                name="32GB single-card serving (community, webcodes-cz)",
                vllm_pin="0.20.1+",
                tensor_parallel_size=1,
                kv_cache_dtype="turboquant_k8v4",
                max_model_len=5120,
                max_num_seqs=3,
                max_num_batched_tokens=15360,
                gpu_memory_utilization=0.96,
                speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                enable_prefix_caching=False,
                additional_args=(),
                recommended_genesis_patches=(
                    "P58", "P60", "P60b", "P67", "P81",
                    "PN8", "PN11", "PN12", "PN13", "PN14",
                ),
                expected=None,
            ),
        ),
        recommended_workloads=("interactive",),
        license="apache-2.0",
        gated=False,
        credits="webcodes-cz (FP8 lm_head conversion), Qwen team (base)",
        quirks=(
            "Requires vllm with FP8 lm_head support (PR #41000). Older "
            "vllm pins will fail with 'unknown quant for lm_head'.",
        ),
        status="EXPERIMENTAL",
        notes=(
            "Tight-fit 32GB card config validated by webcodes-cz on 5090. "
            "Demonstrates that hybrid-TQ + FP8 lm_head fits where pure FP8 "
            "doesn't. Genesis-side untested — community recipe."
        ),
    ),

    "qwen3_next_80b_awq": ModelEntry(
        key="qwen3_next_80b_awq",
        hf_id="Qwen/Qwen3-Next-80B-AWQ",
        hf_revision=None,
        title="Qwen3-Next-80B AWQ INT4 (planned upgrade target)",
        size_gb=40.0,
        quant_format="awq_int4",
        model_class="qwen3_next",
        is_hybrid=True,
        is_moe=True,
        num_experts=None,  # not yet documented
        min_vram_gb_per_rank={1: 48.0, 2: 28.0, 4: 16.0},
        tested_hardware=("rtx_pro_6000_blackwell",),  # planned
        tested_configs=(),  # awaiting hardware availability
        recommended_workloads=("interactive", "throughput"),
        license="apache-2.0",
        gated=False,
        credits="Qwen team (Alibaba)",
        quirks=(),
        status="PLANNED",
        notes=(
            "Future target after RTX PRO 6000 Blackwell upgrade (Q3 2026). "
            "Genesis-side untested. Listed for visibility."
        ),
    ),
}


# ─── Lookup helpers ──────────────────────────────────────────────────────


def get_model(key: str) -> ModelEntry | None:
    """Return the ModelEntry for `key`, or None if not in registry."""
    return SUPPORTED_MODELS.get(key)


def list_models(status_filter: str | None = None) -> list[ModelEntry]:
    """List all registered models, optionally filtered by status."""
    models = list(SUPPORTED_MODELS.values())
    if status_filter:
        models = [m for m in models if m.status == status_filter]
    # Sort by status priority then size: PROD first, then SUPPORTED, then ...
    status_order = ["PROD", "SUPPORTED", "EXPERIMENTAL", "PLANNED"]
    return sorted(
        models,
        key=lambda m: (status_order.index(m.status)
                       if m.status in status_order else 999,
                       m.size_gb),
    )


def list_recommended_for_hardware(
    *, vram_gb_total: float, num_gpus: int, hardware_class: str | None = None,
) -> list[ModelEntry]:
    """Models that fit the given hardware envelope. Used by `genesis init`."""
    suitable: list[ModelEntry] = []
    for m in SUPPORTED_MODELS.values():
        if m.status == "PLANNED":
            continue
        # Find the smallest TP that fits per-rank VRAM budget
        per_rank_budget = vram_gb_total / num_gpus
        candidate_tps = [
            tp for tp, need in m.min_vram_gb_per_rank.items()
            if tp <= num_gpus and need <= per_rank_budget
        ]
        if not candidate_tps:
            continue
        if hardware_class and m.tested_hardware and \
                hardware_class not in m.tested_hardware:
            # Allow but mark as not-tested-on-this-hw
            pass
        suitable.append(m)
    # Stable sort: PROD first, then SUPPORTED, then by size
    status_order = ["PROD", "SUPPORTED", "EXPERIMENTAL", "PLANNED"]
    return sorted(
        suitable,
        key=lambda m: (status_order.index(m.status)
                       if m.status in status_order else 999,
                       m.size_gb),
    )
