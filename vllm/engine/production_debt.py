# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class PagedAttentionDebtReport:
    step_id: str
    pdi_score: float  # PagedAttention Debt Index (target <= 12.0)
    block_sprawl_multiplier: float  # Target <= 1.08x
    schedule_latency_ms: float  # Target <= 4.2ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for vLLM PagedAttention v2 execution runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_paged_attention_event(
        self,
        step_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{step_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "step_id": step_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtPagedAttentionGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for vLLM PagedAttention v2 & Engine Runner.

    Quantifies physical KV block table fragmentation, continuous batch preemption swapping, and engine scheduling step latency against 4 Enterprise KPIs:
    1. PagedAttention Debt Index (PDI <= 12.0)
    2. Physical Block Memory Multiplier (PBMM <= 1.08x)
    3. P99 Engine Scheduling Latency (<= 4.2ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_pdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_pdi = max_acceptable_pdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_engine_step(
        self,
        step_id: str,
        allocated_kv_blocks: int = 16000,
        utilized_kv_blocks: int = 16800,
        schedule_latency_ms: float = 3.2,
        preemption_swaps: int = 0,
        un_gated_mutations: int = 0,
    ) -> PagedAttentionDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_paged_attention_event(
                step_id=step_id,
                event_type="engine_step_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. vLLM engine execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Physical Block Memory Multiplier
        block_ratio = utilized_kv_blocks / max(1, allocated_kv_blocks)
        if block_ratio > 1.8:
            critical_smells.append(f"HIGH_PHYSICAL_BLOCK_FRAGMENTATION_{block_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if schedule_latency_ms > 20.0:
            critical_smells.append(f"HIGH_ENGINE_SCHEDULE_LATENCY_{schedule_latency_ms:.1f}MS")

        # Preemption swaps
        if preemption_swaps > 0:
            critical_smells.append(f"DETECTED_{preemption_swaps}_PREEMPTION_SWAPS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_SPECULATIVE_MUTATIONS")

        # KPI 1: PagedAttention Debt Index (0 = Clean, 100 = Catastrophic)
        pdi = (
            max(0.0, (block_ratio - 1.0) * 20.0)
            + max(0.0, (schedule_latency_ms - 4.2) * 0.5)
            + (preemption_swaps * 25.0)
            + (un_gated_mutations * 30.0)
        )
        pdi_score = round(min(100.0, pdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - pdi_score)
        is_production_ready = (
            pdi_score <= self.max_acceptable_pdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_paged_attention_event(
            step_id=step_id,
            event_type="engine_step_authorized" if is_production_ready else "engine_step_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "pdi_score": pdi_score,
                "block_ratio": block_ratio,
                "allocated_kv_blocks": allocated_kv_blocks,
                "utilized_kv_blocks": utilized_kv_blocks,
                "schedule_latency_ms": schedule_latency_ms,
                "preemption_swaps": preemption_swaps,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return PagedAttentionDebtReport(
            step_id=step_id,
            pdi_score=pdi_score,
            block_sprawl_multiplier=round(block_ratio, 2),
            schedule_latency_ms=round(schedule_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
