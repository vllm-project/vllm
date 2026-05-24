# SPDX-License-Identifier: Apache-2.0
"""TDD for P14 — BlockTable runtime wiring.

Uses a fake BlockTable class injected into sys.modules to simulate the
vLLM module without requiring a real vLLM install.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import sys
import types
import pytest


class _FakeBlockTableModule:
    """Helper to mint a fresh fake block_table module per test."""

    @staticmethod
    def make(max_per_req: int = 8):
        import numpy as np

        class FakeBlockTableNP:
            def __init__(self, shape):
                self.np = np.full(shape, 99, dtype=np.int32)

        class FakeBlockTable:
            def __init__(self, num_rows: int = 4):
                self.max_num_blocks_per_req = max_per_req
                self.block_table = FakeBlockTableNP((num_rows, max_per_req))
                self.num_blocks_per_row = np.zeros(num_rows, dtype=np.int32)

            def append_row(self, block_ids, row_idx):
                if not block_ids:
                    return
                num = len(block_ids)
                start = int(self.num_blocks_per_row[row_idx])
                self.num_blocks_per_row[row_idx] = start + num
                self.block_table.np[row_idx, start:start + num] = block_ids

            def move_row(self, src: int, tgt: int) -> None:
                num_blocks = int(self.num_blocks_per_row[src])
                self.block_table.np[tgt, :num_blocks] = (
                    self.block_table.np[src, :num_blocks]
                )
                self.num_blocks_per_row[tgt] = num_blocks

        return FakeBlockTable


@pytest.fixture
def fake_block_table_module(monkeypatch):
    """Inject fake vllm.v1.worker.block_table into sys.modules."""
    mod_name = "vllm.v1.worker.block_table"
    FakeBlockTable = _FakeBlockTableModule.make()

    # Ensure parent packages exist
    for p in ["vllm", "vllm.v1", "vllm.v1.worker"]:
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)

    fake_mod = types.ModuleType(mod_name)
    fake_mod.BlockTable = FakeBlockTable
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    yield FakeBlockTable
    sys.modules.pop(mod_name, None)


class TestPatch14Wiring:
    def test_apply_wraps_both_methods(self, fake_block_table_module):
        from vllm._genesis.wiring.legacy import patch_14_block_table as p14

        status, reason = p14.apply()
        assert status == "applied", f"{status}: {reason}"
        assert p14.is_applied()

        # Wrappers have marker attrs
        cls = fake_block_table_module
        assert getattr(cls.append_row, "_genesis_p14_append_wrapped", False)
        assert getattr(cls.move_row, "_genesis_p14_move_wrapped", False)

        # Live test: append + tail is zeroed
        instance = cls()
        instance.append_row([10, 20, 30], row_idx=0)
        assert list(instance.block_table.np[0, 0:3]) == [10, 20, 30]
        # Tail positions 3..8 must be zero (were 99 before)
        assert (instance.block_table.np[0, 3:] == 0).all()

        p14.revert()

    def test_idempotent(self, fake_block_table_module):
        from vllm._genesis.wiring.legacy import patch_14_block_table as p14

        s1, _ = p14.apply()
        s2, _ = p14.apply()
        assert s1 == "applied"
        assert s2 == "applied"  # idempotent reason string in s2

        # Check no double-wrap: the stashed originals must be the real originals,
        # not wrappers.
        cls = fake_block_table_module
        orig_append = getattr(cls.append_row, "_genesis_p14_original_append", None)
        assert orig_append is not None
        assert not getattr(orig_append, "_genesis_p14_append_wrapped", False)

        orig_move = getattr(cls.move_row, "_genesis_p14_original_move", None)
        assert orig_move is not None
        assert not getattr(orig_move, "_genesis_p14_move_wrapped", False)

        p14.revert()

    def test_move_row_zeros_target_tail(self, fake_block_table_module):
        """The critical regression: moving shorter into longer slot — tail must zero."""
        from vllm._genesis.wiring.legacy import patch_14_block_table as p14

        p14.apply()
        cls = fake_block_table_module
        instance = cls()

        # Row 0: 2 blocks [10, 20] (tail will be zeroed by append)
        instance.append_row([10, 20], row_idx=0)
        # Row 1: 6 blocks (direct manipulation to simulate previous state)
        instance.block_table.np[1, :6] = [91, 92, 93, 94, 95, 96]
        instance.num_blocks_per_row[1] = 6

        # Move row 0 → row 1 (replaces with shorter content)
        instance.move_row(src=0, tgt=1)

        assert list(instance.block_table.np[1, :2]) == [10, 20]
        # Tail (positions 2-7) must be zero — stale 93..96 must NOT leak
        assert (instance.block_table.np[1, 2:] == 0).all()
        assert instance.num_blocks_per_row[1] == 2

        p14.revert()

    def test_skip_when_module_missing(self, monkeypatch):
        """If BlockTable isn't importable, wiring skips cleanly."""
        from vllm._genesis.wiring.legacy import patch_14_block_table as p14
        monkeypatch.setattr(p14, "_import_block_table", lambda: None)
        status, reason = p14.apply()
        assert status == "skipped"
        assert "block_table" in reason

    def test_revert_restores_originals(self, fake_block_table_module):
        from vllm._genesis.wiring.legacy import patch_14_block_table as p14
        cls = fake_block_table_module
        orig_append = cls.append_row
        orig_move = cls.move_row

        p14.apply()
        assert cls.append_row is not orig_append
        assert cls.move_row is not orig_move

        assert p14.revert()
        assert cls.append_row is orig_append
        assert cls.move_row is orig_move
