# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../vllm/engine/production_debt.py",
)
spec = importlib.util.spec_from_file_location("vllm_engine_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["vllm_engine_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtPagedAttentionGate = production_debt_mod.ProductionDebtPagedAttentionGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtPagedAttentionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtPagedAttentionGate(
            never_equate_intent_to_approval=True,
            max_acceptable_pdi=12.0,
        )

    def test_clean_engine_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_engine_step(
            step_id="vllm_paged_attention_v2_h100_step",
            allocated_kv_blocks=16000,
            utilized_kv_blocks=16800,
            schedule_latency_ms=3.2,
            preemption_swaps=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.pdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_engine_step_fails_debt(self) -> None:
        report = self.gate.evaluate_engine_step(
            step_id="uncalibrated_vllm_engine_step",
            allocated_kv_blocks=16000,
            utilized_kv_blocks=45000,  # 2.81x physical block fragmentation sprawl
            schedule_latency_ms=45.0,  # High engine schedule latency
            preemption_swaps=3,  # 3 preemption swaps
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.pdi_score, 50.0)
        self.assertIn("HIGH_PHYSICAL_BLOCK_FRAGMENTATION_2.81X", report.critical_smells)
        self.assertIn("HIGH_ENGINE_SCHEDULE_LATENCY_45.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_PREEMPTION_SWAPS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_SPECULATIVE_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_engine_step("step-1")
        self.gate.evaluate_engine_step("step-2")
        self.gate.evaluate_engine_step("step-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
