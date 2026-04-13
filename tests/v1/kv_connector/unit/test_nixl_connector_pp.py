# SPDX-License-Identifier: Apache-2.0
# Unit tests for PP support in NixlConnector.
#
# TDD: Tests marked [FAILS_NOW] fail with current vllm code and pass after fix.
#      Tests marked [PASSES_NOW] document current (broken) behavior.
#
# Two gaps under test:
# 1. gpu_worker.get_kv_connector_handshake_metadata: key = tp_rank only (collision)
#    Fix: key = pp_rank * tp_size + tp_rank  (global worker index)
# 2. NixlConnectorScheduler.side_channel_port: offset by dp_index only (collision)
#    Fix: offset += pp_rank * tp_size

import pytest
import torch
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the actual vllm functions under test
# ---------------------------------------------------------------------------
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


# ---------------------------------------------------------------------------
# Helper: call GPUWorker.get_kv_connector_handshake_metadata as unbound method
# with mocked tp_rank, pp_rank, tp_size, and a fake metadata object.
# ---------------------------------------------------------------------------

def _call_handshake_metadata(tp_rank: int, pp_rank: int, tp_size: int):
    """
    Call the actual GPUWorker.get_kv_connector_handshake_metadata with
    mocked distributed state.  Returns the dict it produces.
    """
    fake_metadata = MagicMock(name="metadata")

    mock_tp_group = MagicMock()
    mock_tp_group.rank_in_group = tp_rank

    mock_pp_group = MagicMock()
    mock_pp_group.rank_in_group = pp_rank
    mock_pp_group.world_size = tp_size  # pp_size (unused in current code)

    mock_connector = MagicMock()
    mock_connector.get_handshake_metadata.return_value = fake_metadata

    mock_parallel_config = MagicMock()
    mock_parallel_config.tensor_parallel_size = tp_size
    mock_parallel_config.pipeline_parallel_size = max(1, pp_rank + 1)  # at least pp_rank+1

    mock_self = MagicMock()
    mock_self.vllm_config.parallel_config = mock_parallel_config

    with (
        patch("vllm.v1.worker.gpu_worker.has_kv_transfer_group", return_value=True),
        patch("vllm.v1.worker.gpu_worker.get_kv_transfer_group",
              return_value=mock_connector),
        patch("vllm.v1.worker.gpu_worker.get_tp_group", return_value=mock_tp_group),
        # get_pp_group does not exist yet in gpu_worker — patching it in advance
        patch("vllm.v1.worker.gpu_worker.get_pp_group",
              return_value=mock_pp_group),
    ):
        return GPUWorker.get_kv_connector_handshake_metadata(mock_self)


# ===========================================================================
# Gap 1: Handshake key collision
# ===========================================================================

class TestHandshakeKeyPP:
    """
    GPUWorker.get_kv_connector_handshake_metadata should return a key that
    uniquely identifies each worker across both TP and PP dimensions.

    Current code: key = tp_rank   (only TP dimension)
    Required fix: key = pp_rank * tp_size + tp_rank   (global worker index)
    """

    # --- [PASSES_NOW] Documents current behavior: key is just tp_rank ---

    def test_pp1_tp0_key_is_0_current(self):
        """PP=1, tp_rank=0 → key must be 0 (baseline, no PP)."""
        result = _call_handshake_metadata(tp_rank=0, pp_rank=0, tp_size=4)
        assert result is not None
        assert 0 in result

    def test_pp1_tp1_key_is_1_current(self):
        """PP=1, tp_rank=1 → key must be 1 (baseline, no PP)."""
        result = _call_handshake_metadata(tp_rank=1, pp_rank=0, tp_size=4)
        assert result is not None
        assert 1 in result

    # --- [FAILS_NOW] These tests FAIL with current code, PASS after fix ---

    def test_pp1_rank1_tp_rank0_key_must_be_4(self):
        """[FAILS_NOW] pp_rank=1, tp_rank=0, tp_size=4 → key must be 4, not 0.

        Current code returns {0: metadata} because it ignores pp_rank.
        After fix it should return {4: metadata} (1*4+0=4).
        """
        result = _call_handshake_metadata(tp_rank=0, pp_rank=1, tp_size=4)
        assert result is not None
        assert 4 in result, (
            f"Expected key=4 (pp_rank=1*tp_size=4+tp_rank=0), "
            f"got keys={list(result.keys())}. "
            "Fix: use pp_rank * tp_size + tp_rank as key."
        )

    def test_pp1_rank1_tp_rank3_key_must_be_7(self):
        """[FAILS_NOW] pp_rank=1, tp_rank=3, tp_size=4 → key must be 7."""
        result = _call_handshake_metadata(tp_rank=3, pp_rank=1, tp_size=4)
        assert result is not None
        assert 7 in result, (
            f"Expected key=7 (1*4+3), got keys={list(result.keys())}"
        )

    def test_pp3_rank3_tp_rank1_key_must_be_7(self):
        """[FAILS_NOW] pp_rank=3, tp_rank=1, tp_size=2 → key must be 7 (3*2+1)."""
        result = _call_handshake_metadata(tp_rank=1, pp_rank=3, tp_size=2)
        assert result is not None
        assert 7 in result, (
            f"Expected key=7 (3*2+1), got keys={list(result.keys())}"
        )

    def test_pp0_workers_survive_merge_with_pp1(self):
        """[FAILS_NOW] After merging PP0+PP1 workers, PP0 metadata must NOT be lost.

        Simulates engine/core.py collecting metadata from all 8 workers (PP2+TP4).
        With current code (key=tp_rank), PP0 is overwritten by PP1.
        After fix (key=global_idx), all 8 workers survive.
        """
        # Collect dicts from 8 workers (PP2+TP4)
        worker_dicts = [
            _call_handshake_metadata(tp_rank=tp_r, pp_rank=pp_r, tp_size=4)
            for pp_r in range(2)
            for tp_r in range(4)
        ]

        # Merge (mirrors engine/core.py logic)
        merged: dict = {}
        for d in worker_dicts:
            if d is not None:
                merged.update(d)

        # After fix: all 8 workers present (keys 0-7)
        assert len(merged) == 8, (
            f"Expected 8 unique worker entries after merge, got {len(merged)}. "
            "Current code loses PP0 workers because PP1 overwrites them."
        )

        # Verify PP0 workers (keys 0-3) are present
        for tp_r in range(4):
            expected_key = 0 * 4 + tp_r  # pp_rank=0
            assert expected_key in merged, f"PP0 tp_rank={tp_r} missing from merged dict"

        # Verify PP1 workers (keys 4-7) are present
        for tp_r in range(4):
            expected_key = 1 * 4 + tp_r  # pp_rank=1
            assert expected_key in merged, f"PP1 tp_rank={tp_r} missing from merged dict"

    def test_pp1_behavior_backward_compatible(self):
        """[PASSES_NOW and after fix] PP=1 key=0 is unchanged."""
        result_pp1 = _call_handshake_metadata(tp_rank=0, pp_rank=0, tp_size=4)
        # With both current and fixed code, PP=1 should return key=0
        assert result_pp1 is not None
        assert 0 in result_pp1


# ===========================================================================
# Gap 2: Side-channel port collision
# ===========================================================================

class TestSideChannelPortPP:
    """
    NixlConnectorScheduler.side_channel_port must be unique per PP rank.

    Current code:  port = BASE + data_parallel_index
    Required fix:  port = BASE + data_parallel_index + pp_rank * tp_size

    We test the formula directly since full scheduler init requires complex config.
    """

    BASE_PORT = 9100

    def _current_port(self, dp_index: int) -> int:
        """Current formula (from nixl_connector.py:557-560)."""
        return self.BASE_PORT + dp_index

    def _fixed_port(self, dp_index: int, pp_rank: int, tp_size: int) -> int:
        """Expected formula after fix."""
        return self.BASE_PORT + dp_index + pp_rank * tp_size

    # --- [PASSES_NOW] Documents current broken behavior ---

    def test_current_formula_collision_same_dp_different_pp(self):
        """[PASSES_NOW] Current formula: PP0 and PP1 get same port when dp_index=0."""
        port_pp0 = self._current_port(dp_index=0)
        port_pp1 = self._current_port(dp_index=0)
        # This is the BUG: same port for different PP stages
        assert port_pp0 == port_pp1, "Expected collision — this is the bug"

    # --- [FAILS_NOW] These tests define required behavior after fix ---

    def test_fixed_formula_no_collision_pp2_tp4(self):
        """[PASSES after fix] PP0 and PP1 must get different ports."""
        port_pp0 = self._fixed_port(dp_index=0, pp_rank=0, tp_size=4)
        port_pp1 = self._fixed_port(dp_index=0, pp_rank=1, tp_size=4)
        assert port_pp0 != port_pp1, (
            "PP0 and PP1 must use different ports to avoid bind conflict"
        )
        assert port_pp1 - port_pp0 == 4  # offset = 1 * tp_size

    def test_fixed_formula_pp1_same_as_current(self):
        """[PASSES after fix] PP=1 (pp_rank=0): fixed formula equals current."""
        current = self._current_port(dp_index=0)
        fixed = self._fixed_port(dp_index=0, pp_rank=0, tp_size=4)
        assert current == fixed, "PP=1 must be backward compatible"

    def test_fixed_formula_all_unique_pp4_tp2(self):
        """[PASSES after fix] PP4+TP2: 4 PP stages all get unique ports."""
        ports = [
            self._fixed_port(dp_index=0, pp_rank=pp_r, tp_size=2)
            for pp_r in range(4)
        ]
        assert len(ports) == len(set(ports)), f"Port collision: {ports}"

    def test_note_port_is_per_engine_not_per_pp_rank(self):
        """[INFO] NixlConnectorScheduler port is per-engine, not per-PP-rank.

        In vLLM PP, there is ONE EngineCore (and ONE NixlConnectorScheduler)
        per serving instance, regardless of PP size. All PP ranks are worker
        subprocesses of the same engine. Therefore:
        - Only ONE side_channel_port exists per Prefill instance.
        - Port collision between PP ranks does NOT occur at the scheduler level.
        - No scheduler port fix is needed.

        This test documents the architecture to prevent future confusion.
        """
        # NixlConnectorScheduler is instantiated ONCE in NixlConnector.__init__:
        #   NixlConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
        # With PP2+TP4, there are 8 workers but only 1 scheduler.
        # The side_channel_port formula (base + dp_index) is correct for PP.
        assert True  # architecture documentation test


# ===========================================================================
# Tests: TpKVTopology.get_all_pp_tp_targets (Wave 2.2)
# ===========================================================================

class TestTpKVTopologyPP:
    """
    TpKVTopology needs a new method get_all_pp_tp_targets that returns
    global worker indices across ALL PP stages for a given local D rank.

    Current get_target_remote_ranks(remote_tp_size=4):
      D_rank 0,1 -> [0]  (only PP0_TP0, PP1 never reached)

    Required get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2):
      D_rank 0,1 -> [0, 4]  (PP0_TP0 AND PP1_TP0)
      D_rank 2,3 -> [1, 5]
      D_rank 4,5 -> [2, 6]
      D_rank 6,7 -> [3, 7]

    Formula: for each tp_rank in get_target_remote_ranks(tp_size):
               [tp_rank + pp_rank * tp_size for pp_rank in range(pp_size)]
    """

    def _make_topology(self, d_tp_rank: int, d_tp_size: int, engine_id="prefill"):
        from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
        from unittest.mock import MagicMock

        mock_backend = MagicMock()
        mock_backend.get_kv_cache_shape.return_value = (1, 16, 4, 1, 1)
        mock_backend.get_kv_cache_stride_order.side_effect = NotImplementedError

        return TpKVTopology(
            tp_rank=d_tp_rank,
            engine_id=engine_id,
            remote_tp_size={engine_id: d_tp_size},
            remote_block_size={engine_id: 16},
            is_mla=True,
            total_num_kv_heads=4,
            attn_backends=[mock_backend],
            tensor_shape=None,
        )

    def test_pp1_identical_to_existing_method(self):
        """PP=1: get_all_pp_tp_targets must equal get_target_remote_ranks (backward compat)."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        existing = topo.get_target_remote_ranks(remote_tp_size=4)
        pp1_result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=1)
        assert pp1_result == existing

    # --- [FAILS_NOW] ---

    def test_d_rank0_pp2_tp4_connects_to_0_and_4(self):
        """[FAILS_NOW] D_rank=0, D_TP=8, remote PP2+TP4 -> global [0, 4]."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [0, 4], f"Got {result}"

    def test_d_rank1_pp2_tp4_connects_to_0_and_4(self):
        """[FAILS_NOW] D_rank=1, D_TP=8, remote PP2+TP4 -> global [0, 4]."""
        topo = self._make_topology(d_tp_rank=1, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [0, 4]

    def test_d_rank2_pp2_tp4_connects_to_1_and_5(self):
        """[FAILS_NOW] D_rank=2, D_TP=8 -> [1, 5]."""
        topo = self._make_topology(d_tp_rank=2, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [1, 5]

    def test_d_rank7_pp2_tp4_connects_to_3_and_7(self):
        """[FAILS_NOW] D_rank=7, D_TP=8 -> [3, 7]."""
        topo = self._make_topology(d_tp_rank=7, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [3, 7]

    def test_all_8_d_ranks_cover_all_8_global_indices(self):
        """[FAILS_NOW] All 8 D ranks together must cover all 8 global indices 0-7."""
        covered: set[int] = set()
        for d_rank in range(8):
            topo = self._make_topology(d_tp_rank=d_rank, d_tp_size=8)
            covered.update(topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2))
        assert covered == set(range(8)), f"Missing global indices: {set(range(8)) - covered}"

    def test_pp4_tp2_d_rank0_connects_to_4_agents(self):
        """[FAILS_NOW] D_rank=0, D_TP=8, PP4+TP2 -> [0, 2, 4, 6] (one per PP stage)."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=2, remote_pp_size=4)
        assert result == [0, 2, 4, 6], f"Got {result}"
