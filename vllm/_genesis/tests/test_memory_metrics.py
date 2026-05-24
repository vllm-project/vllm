# SPDX-License-Identifier: Apache-2.0
"""Tests for `vllm._genesis.memory_metrics.genesis_memory_summary`."""
from __future__ import annotations

import json

import torch


class TestMemorySummaryShape:
    def test_returns_required_keys(self, reset_genesis_prealloc):
        from vllm._genesis.memory_metrics import genesis_memory_summary
        s = genesis_memory_summary()
        assert set(s.keys()) >= {
            "total_genesis_bytes",
            "total_genesis_human",
            "per_pool",
            "torch_cuda",
        }
        assert isinstance(s["total_genesis_bytes"], int)
        assert isinstance(s["per_pool"], dict)

    def test_per_pool_has_all_managers(self, reset_genesis_prealloc):
        from vllm._genesis.memory_metrics import genesis_memory_summary
        s = genesis_memory_summary()
        # v7.3: P37 + P39a. v7.7: +P46 (gdn_gating_buffer). Expected set
        # MUST track `genesis_memory_summary()` body line-for-line or
        # tests silently pass on stale pool set.
        assert set(s["per_pool"].keys()) == {
            "turboquant_buffer_manager",
            "gdn_core_attn_manager",
            "moe_intermediate_cache",
            "fla_kkt_buffer",
            "gdn_gating_buffer",
            "prealloc_framework",
        }

    def test_json_serialisable(self, reset_genesis_prealloc):
        from vllm._genesis.memory_metrics import genesis_memory_summary
        s = genesis_memory_summary()
        # default=str so torch.device / dtype get stringified
        json.dumps(s, default=str)

    def test_empty_registry_zero_bytes(self, reset_genesis_prealloc):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis.kernels.gdn_core_attn_manager import GdnCoreAttnManager
        from vllm._genesis.memory_metrics import genesis_memory_summary
        TurboQuantBufferManager.clear_for_tests()
        GdnCoreAttnManager.clear_for_tests()
        s = genesis_memory_summary()
        assert s["total_genesis_bytes"] == 0
        assert s["total_genesis_human"] == "0 B"


class TestMemorySummaryWithAllocations:
    def test_tq_allocation_reflected(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis.kernels.gdn_core_attn_manager import GdnCoreAttnManager
        from vllm._genesis.memory_metrics import genesis_memory_summary
        db.TurboQuantBufferManager.clear_for_tests()
        GdnCoreAttnManager.clear_for_tests()

        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply",
            classmethod(lambda cls: True),
        )
        # Allocate a P36 shared decode buf
        t = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        assert t is not None
        expected_bytes = t.element_size() * t.numel()

        s = genesis_memory_summary()
        assert s["total_genesis_bytes"] >= expected_bytes
        tq = s["per_pool"]["turboquant_buffer_manager"]
        assert "total_bytes" in tq
        assert tq["total_bytes"] >= expected_bytes

    def test_human_format_escalates(self, reset_genesis_prealloc):
        from vllm._genesis.memory_metrics import _humanize_bytes
        assert _humanize_bytes(0) == "0 B"
        assert _humanize_bytes(1023) == "1023 B"
        assert _humanize_bytes(1024).endswith(" KiB")
        assert _humanize_bytes(1 << 20).endswith(" MiB")
        assert _humanize_bytes(1 << 30).endswith(" GiB")


class TestLogHelper:
    def test_log_genesis_memory_does_not_raise(self, caplog):
        """log_genesis_memory must never propagate exceptions."""
        import logging
        from vllm._genesis.memory_metrics import log_genesis_memory
        with caplog.at_level(logging.INFO, logger="genesis.memory_metrics"):
            log_genesis_memory()
        # At least the line got emitted (no assertion on content —
        # depends on state of other tests in suite).

    def test_log_genesis_memory_level_warning(self, caplog):
        import logging
        from vllm._genesis.memory_metrics import log_genesis_memory
        with caplog.at_level(logging.WARNING, logger="genesis.memory_metrics"):
            log_genesis_memory(level=logging.WARNING)
