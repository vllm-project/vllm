# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup memory planner / tier plan logger for hierarchical offload."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.config.offload import HierarchicalOffloadConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TierPlan:
    """Predicted placement for hierarchical expert staging."""

    device_expert_gb: float
    ram_expert_gb: float
    disk_expert_gb: float
    num_moe_layers: int
    num_local_experts: int
    slots_per_layer: int
    expert_row_bytes: int
    bottleneck: str
    policy: str
    disk_path: str | None

    def summary(self) -> str:
        lines = [
            "Hierarchical expert tier plan:",
            f"  policy={self.policy}",
            f"  moe_layers={self.num_moe_layers} "
            f"local_experts={self.num_local_experts} "
            f"slots/layer={self.slots_per_layer}",
            f"  expert_row={self.expert_row_bytes / 1e6:.2f} MB",
            f"  device_slots={self.device_expert_gb:.3f} GiB",
            f"  ram_cache={self.ram_expert_gb:.3f} GiB",
            f"  disk_backing={self.disk_expert_gb:.3f} GiB "
            f"path={self.disk_path!r}",
            f"  predicted_bottleneck={self.bottleneck}",
        ]
        return "\n".join(lines)


def _mem_available_bytes() -> int:
    """Best-effort MemAvailable (Linux) else a conservative fallback."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # Fallback: 25% of reported total RAM via os.sysconf when available.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size * 0.25)
    except (ValueError, OSError, AttributeError):
        return 8 * 1024**3


def resolve_ram_budget_bytes(cfg: HierarchicalOffloadConfig) -> int:
    """Resolve pinned RAM budget in bytes from config."""
    if cfg.tier_ram_gb == 0:
        return 0
    if cfg.tier_ram_gb > 0:
        return int(cfg.tier_ram_gb * 1024**3)
    # auto: take up to 50% of MemAvailable with a 2 GiB OS reserve.
    available = max(0, _mem_available_bytes() - 2 * 1024**3)
    return int(available * 0.5)


def compute_slots_per_layer(
    cfg: HierarchicalOffloadConfig,
    *,
    num_moe_layers: int,
    num_local_experts: int,
    expert_row_bytes: int,
    top_k: int = 8,
) -> int:
    """Derive device slot count per MoE layer."""
    if cfg.tier_num_slots > 0:
        return min(cfg.tier_num_slots, num_local_experts)
    if num_moe_layers <= 0 or expert_row_bytes <= 0:
        return min(max(top_k * 4, 16), num_local_experts)
    if cfg.tier_device_expert_gb > 0:
        budget = int(cfg.tier_device_expert_gb * 1024**3)
        per_layer = max(1, budget // max(num_moe_layers, 1))
        slots = max(1, per_layer // expert_row_bytes)
    else:
        # Heuristic when backend forced on without explicit budget:
        # enough slots for a few batches of unique top-k experts.
        slots = max(top_k * 4, 32)
    return max(1, min(slots, num_local_experts))


def build_tier_plan(
    cfg: HierarchicalOffloadConfig,
    *,
    num_moe_layers: int,
    num_local_experts: int,
    expert_row_bytes: int,
    top_k: int = 8,
) -> TierPlan:
    """Build and return a tier placement plan."""
    slots = compute_slots_per_layer(
        cfg,
        num_moe_layers=num_moe_layers,
        num_local_experts=num_local_experts,
        expert_row_bytes=expert_row_bytes,
        top_k=top_k,
    )
    device_bytes = slots * expert_row_bytes * max(num_moe_layers, 1)
    total_expert_bytes = num_local_experts * expert_row_bytes * max(num_moe_layers, 1)
    ram_budget = resolve_ram_budget_bytes(cfg)
    ram_bytes = min(ram_budget, total_expert_bytes)
    disk_bytes = max(0, total_expert_bytes - ram_bytes)

    if disk_bytes > 0 and cfg.tier_disk_path is None:
        bottleneck = "disk_required_but_unset"
    elif disk_bytes > 0:
        bottleneck = "nvme"
    elif slots < num_local_experts:
        bottleneck = "pcie_or_ram_hits"
    else:
        bottleneck = "none_full_residency"

    return TierPlan(
        device_expert_gb=device_bytes / 1024**3,
        ram_expert_gb=ram_bytes / 1024**3,
        disk_expert_gb=disk_bytes / 1024**3,
        num_moe_layers=num_moe_layers,
        num_local_experts=num_local_experts,
        slots_per_layer=slots,
        expert_row_bytes=expert_row_bytes,
        bottleneck=bottleneck,
        policy=cfg.tier_policy,
        disk_path=cfg.tier_disk_path,
    )


def format_tier_plan(plan: TierPlan) -> str:
    """Format a TierPlan for logging."""
    return plan.summary()


def log_tier_plan(plan: TierPlan) -> None:
    """Log the tier plan once at startup."""
    logger.info_once("%s", plan.summary())
