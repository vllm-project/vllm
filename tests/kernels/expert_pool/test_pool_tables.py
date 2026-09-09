# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the global expert pool: step semantics, ownership.

Ported from the lab expert tier; the torch reference defines the semantics
that the Triton step program must match exactly (GPU equivalence is a
separate test)."""

import random
import unittest

import torch

from vllm.model_executor.layers.fused_moe.expert_pool import pool as pool_mod
from vllm.model_executor.layers.fused_moe.expert_pool import tables as gp


def make_sources(layers, experts, width=3):
    """Per-layer RAM banks: row e of layer l holds value l*100 + e in every tensor."""
    return [
        {
            name: torch.arange(experts, dtype=torch.int32)
            .add(layer * 100)
            .unsqueeze(1)
            .repeat(1, width)
            .contiguous()
            for name in gp.TENSORS
        }
        for layer in range(layers)
    ]


class GlobalPoolTests(unittest.TestCase):
    def setup(self, layers=2, experts=6, slots=(2, 2), staging=2):
        device = torch.device("cpu")
        sources = make_sources(layers, experts)
        pool = pool_mod.GlobalPool(device, sources[0], list(slots), staging)
        for layer, count in enumerate(slots):
            start = pool.offset(layer)
            for name in gp.TENSORS:
                pool.bank[name][start : start + count].copy_(
                    sources[layer][name][:count]
                )
        buffers = [
            gp.allocate_step_buffers(device, experts, staging) for _ in range(layers)
        ]
        return pool, sources, buffers

    def run_step(self, pool, sources, buffers, layer, ids):
        gp.step(pool.tables, layer, torch.tensor([ids]), buffers[layer])
        pool_mod.copy_in(sources[layer], pool.bank, buffers[layer])
        return buffers[layer]

    def assert_bank_holds_owners(self, pool):
        tables = pool.tables
        E = tables.num_experts
        for row, key in enumerate(tables.row_key.tolist()):
            if key < 0:
                continue
            layer, expert = divmod(key, E)
            for name in gp.TENSORS:
                self.assertTrue(
                    bool((pool.bank[name][row] == layer * 100 + expert).all()),
                    f"{name} row {row} key {key}",
                )

    def test_initial_layout_packs_layers_and_validates(self):
        pool, sources, buffers = self.setup()
        tables = pool.tables
        self.assertEqual(
            tables.hot_phys.tolist(), [0, 1, -1, -1, -1, -1] + [2, 3, -1, -1, -1, -1]
        )
        self.assertEqual(tables.cold_phys.tolist(), [-1, -1, 2, 3, 4, 5] * 2)
        self.assertEqual(tables.row_key.tolist(), [0, 1, 6, 7, -1, -1])
        self.assertEqual(tables.staging_rows.tolist(), [4, 5])
        self.assertEqual(pool.snapshot(), [2, 2])
        self.assert_bank_holds_owners(pool)

    def test_gate_closed_stages_only_and_leaves_recency(self):
        pool, sources, buffers = self.setup()
        b = self.run_step(pool, sources, buffers, 1, [4, 4])
        self.assertEqual(int(b.promoted_count[0]), 0)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(int(b.gather_count[0]), 1)
        self.assertEqual(b.step_map.tolist(), [2, 3, -1, -1, 4, -1])
        self.assertEqual(int(pool.tables.clock[0]), 0)
        self.assertEqual(pool.tables.row_use[: pool.tables.pool_rows].sum().item(), 0)
        # Staging row 4 now holds layer 1 expert 4.
        self.assertTrue(bool((pool.bank[gp.TENSORS[0]][4] == 104).all()))
        self.assertEqual(pool.snapshot(), [2, 2])

    def test_miss_evicts_least_recent_of_any_layer_and_never_writes_ram(self):
        pool, sources, buffers = self.setup()
        before = [{n: t.clone() for n, t in s.items()} for s in sources]
        gp.set_gate(pool.tables, True)
        # Layer 0 touches its residents 0,1; layer 1 touches only 7 (key).
        self.run_step(pool, sources, buffers, 0, [0, 1])
        self.run_step(pool, sources, buffers, 1, [1, -1])
        # Layer 0 misses 5: victim is key 6 (layer 1 expert 0, last_use 0).
        b = self.run_step(pool, sources, buffers, 0, [5, 0])
        self.assertEqual(int(b.promoted_count[0]), 1)
        self.assertEqual(b.gather_src.tolist()[:1], [5])
        self.assertEqual(b.gather_dst.tolist()[:1], [2])
        tables = pool.tables
        self.assertEqual(tables.hot_phys.tolist()[5], 2)
        self.assertEqual(tables.hot_phys.tolist()[6], -1)
        self.assertEqual(tables.cold_phys.tolist()[6], 0)
        self.assertEqual(tables.row_key.tolist()[2], 5)
        self.assertEqual(b.step_map.tolist(), [0, 1, -1, -1, -1, 2])
        self.assertEqual(pool.snapshot(), [3, 1])
        self.assert_bank_holds_owners(pool)
        # Layer 1 now misses expert 0 again: victim is the least recent of
        # the remaining residents, layer 0's expert 1 (clock 1 < clock 3).
        b = self.run_step(pool, sources, buffers, 1, [0, 1])
        self.assertEqual(tables.hot_phys.tolist()[1], -1)
        self.assertEqual(tables.hot_phys.tolist()[6], 1)
        self.assertEqual(pool.snapshot(), [2, 2])
        self.assert_bank_holds_owners(pool)
        for layer, source in enumerate(sources):
            for name in gp.TENSORS:
                self.assertTrue(torch.equal(source[name], before[layer][name]))

    def test_routes_follow_the_step_map_per_lane(self):
        """buffers.routes holds each lane's physical row: hits, promotions,
        staged rows, duplicates resolved, padding and invalid ids -1."""
        pool, sources, buffers = self.setup(layers=1, experts=6, slots=(2,), staging=4)
        b = self.run_step(pool, sources, buffers, 0, [1, 4, 4, -1])
        self.assertEqual(b.routes.tolist(), [1, 2, 2, -1])  # staged into row 2
        gp.set_gate(pool.tables, True)
        b = self.run_step(pool, sources, buffers, 0, [3, 9, 1, 3])
        # Expert 3 evicts row 0 (expert 0, older than expert 1); 9 is invalid.
        self.assertEqual(b.routes.tolist(), [0, -1, 1, 0])
        with self.assertRaises(RuntimeError):
            pool.snapshot()

    def test_multi_row_step_deduplicates_across_rows_and_routes_every_lane(self):
        """A verify step: rows x top_k lanes; an expert selected by two rows is
        promoted once and both lanes route to its row; padding rows stay -1."""
        pool, sources, buffers = self.setup(layers=1, experts=8, slots=(3,), staging=6)
        gp.set_gate(pool.tables, True)
        ids = torch.tensor([[0, 5], [5, 6], [-1, -1]])
        gp.step(pool.tables, 0, ids, buffers[0])
        pool_mod.copy_in(sources[0], pool.bank, buffers[0])
        b = buffers[0]
        # Residents 0,1,2; 0 is a hit; 5 and 6 miss and evict 1 then 2.
        self.assertEqual(int(b.promoted_count[0]), 2)
        self.assertEqual(int(b.staged_count[0]), 0)
        hot = pool.tables.hot_phys.tolist()
        self.assertEqual(b.routes.tolist(), [0, hot[5], hot[5], hot[6], -1, -1])
        self.assertEqual(pool.snapshot(), [3])
        self.assert_bank_holds_owners(pool)

    def test_promote_limit_and_interval_stage_the_rest(self):
        """Limit caps promotions per layer call; interval promotes on every
        N-th forward only; both keep every miss served from staging."""
        pool, sources, buffers = self.setup(layers=1, experts=8, slots=(3,), staging=4)
        gp.set_gate(pool.tables, True)
        gp.set_control(pool.tables, promote_limit=1)
        b = self.run_step(pool, sources, buffers, 0, [3, 4, 5, 6])
        self.assertEqual((int(b.promoted_count[0]), int(b.staged_count[0])), (1, 3))
        self.assertTrue(all(int(r) >= 0 for r in b.routes.tolist()))
        gp.set_control(pool.tables, promote_limit=0, promote_interval=2)
        # Forward 2 is outside the interval (forwards 1, 3, 5, ... promote):
        # everything staged, placement frozen.
        before = pool.tables.hot_phys.clone()
        b = self.run_step(pool, sources, buffers, 0, [4, 5, -1, -1])
        self.assertEqual((int(b.promoted_count[0]), int(b.staged_count[0])), (0, 2))
        self.assertTrue(torch.equal(pool.tables.hot_phys, before))
        self.assertEqual(int(pool.tables.forwards[0]), 2)
        # Forward 3: promotion allowed again.
        b = self.run_step(pool, sources, buffers, 0, [6, 7, -1, -1])
        self.assertEqual(int(b.promoted_count[0]), 2)
        pool.snapshot()

    def test_min_misses_and_protect_recent(self):
        pool, sources, buffers = self.setup(layers=1, experts=8, slots=(3,), staging=4)
        gp.set_gate(pool.tables, True)
        gp.set_control(pool.tables, promote_min_misses=2)
        # First miss of expert 5 is staged; the second promotes it.
        b = self.run_step(pool, sources, buffers, 0, [5, -1, -1, -1])
        self.assertEqual(int(b.promoted_count[0]), 0)
        b = self.run_step(pool, sources, buffers, 0, [5, -1, -1, -1])
        self.assertEqual(int(b.promoted_count[0]), 1)
        self.assertEqual(int(pool.tables.miss_count[5]), 0)
        # Residents now: row 0 = expert 5 (used in the previous forward),
        # rows 1 and 2 = experts 1 and 2 (never used). With protect_recent=1
        # the previous forward's row is not a victim: hits on 1 and 2 leave
        # no victim, so both misses are staged; without protection, 6 takes
        # row 0.
        gp.set_control(pool.tables, promote_min_misses=1, protect_recent=1)
        b = self.run_step(pool, sources, buffers, 0, [1, 2, 6, 7])
        self.assertEqual((int(b.promoted_count[0]), int(b.staged_count[0])), (0, 2))
        self.assertEqual(int(pool.tables.hot_phys[5]), 0)
        gp.set_control(pool.tables, protect_recent=0)
        b = self.run_step(pool, sources, buffers, 0, [1, 2, 6, 7])
        self.assertEqual(int(b.promoted_count[0]), 1)
        self.assertEqual(int(pool.tables.hot_phys[6]), 0)
        pool.snapshot()
        with self.assertRaises(ValueError):
            gp.set_control(pool.tables, promote_interval=0)
        with self.assertRaises(ValueError):
            gp.set_control(pool.tables, unknown=1)

    def test_no_victim_falls_back_to_staging(self):
        pool, sources, buffers = self.setup(layers=1, experts=4, slots=(2,), staging=3)
        gp.set_gate(pool.tables, True)
        # Every resident is selected, so the two misses have no victim.
        b = self.run_step(pool, sources, buffers, 0, [0, 1, 2, 3][:3])
        self.assertEqual(int(b.promoted_count[0]), 0)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(b.step_map.tolist(), [0, 1, 2, -1])
        pool.snapshot()

    def test_every_current_route_is_protected_when_the_pool_is_saturated(self):
        """All residents are this step's hits and one more expert misses:
        no victim may be a current route, so the miss is staged only, and
        a promotion in the same step is never re-evicted by a later miss."""
        pool, sources, buffers = self.setup(layers=1, experts=8, slots=(3,), staging=4)
        gp.set_gate(pool.tables, True)
        b = self.run_step(pool, sources, buffers, 0, [0, 1, 2, 5])
        self.assertEqual(int(b.promoted_count[0]), 0)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(pool.tables.hot_phys.tolist()[:3], [0, 1, 2])
        self.assertEqual(b.routes.tolist(), [0, 1, 2, 3])
        # Two misses against one stale row: the first takes it, the second
        # cannot take it back and is staged.
        self.run_step(pool, sources, buffers, 0, [0, 1, -1, -1])
        b = self.run_step(pool, sources, buffers, 0, [0, 5, 6, 1])
        self.assertEqual(int(b.promoted_count[0]), 1)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(pool.tables.hot_phys.tolist()[5], 2)
        self.assertEqual(b.routes.tolist(), [0, 2, 3, 1])
        pool.snapshot()

    def test_invalid_ids_set_the_sticky_error(self):
        pool, sources, buffers = self.setup()
        self.run_step(pool, sources, buffers, 0, [9, 0])
        with self.assertRaises(RuntimeError):
            pool.snapshot()

    def test_host_swap_while_gated_matches_the_tables(self):
        pool, sources, buffers = self.setup()
        pool.host_swap(0, 0, 3)
        tables = pool.tables
        self.assertEqual(tables.hot_phys.tolist()[:6], [-1, 1, -1, 0, -1, -1])
        self.assertEqual(tables.cold_phys.tolist()[:6], [0, -1, 2, -1, 4, 5])
        self.assertEqual(int(tables.row_key[0]), 3)
        pool.snapshot()
        gp.set_gate(tables, True)
        with self.assertRaises(RuntimeError):
            pool.host_swap(0, 3, 0)

    def test_random_steps_keep_ownership_consistent(self):
        rng = random.Random(3)
        for trial in range(20):
            layers = rng.choice([1, 2, 3])
            experts = rng.choice([4, 6, 8])
            staging = rng.randint(1, 3)
            slots = tuple(rng.randint(1, experts - 1) for _ in range(layers))
            pool, sources, buffers = self.setup(layers, experts, slots, staging)
            gp.set_gate(pool.tables, rng.random() < 0.8)
            for step in range(30):
                layer = rng.randrange(layers)
                ids = [rng.choice([-1, rng.randrange(experts)]) for _ in range(staging)]
                b = self.run_step(pool, sources, buffers, layer, ids)
                resident = pool.snapshot()
                self.assertEqual(sum(resident), pool.tables.pool_rows)
                self.assert_bank_holds_owners(pool)
                for expert in {e for e in ids if e >= 0}:
                    self.assertGreaterEqual(int(b.step_map[expert]), 0)


if __name__ == "__main__":
    unittest.main()


class VerifyBankRowsTests(unittest.TestCase):
    def test_sampled_rows_match_and_a_corruption_is_caught(self):
        device = torch.device("cpu")
        sources = make_sources(2, 6)
        pool = pool_mod.GlobalPool(device, sources[0], [2, 2], 2)
        for layer, count in enumerate((2, 2)):
            start = pool.offset(layer)
            for name in gp.TENSORS:
                pool.bank[name][start : start + count].copy_(
                    sources[layer][name][:count]
                )
        report = pool_mod.verify_bank_rows(pool, sources, sample=1)
        self.assertEqual(report, {"rows_checked": 2, "rows_resident": 4})
        pool.bank["w2_weight"][0, 0] += 1
        with self.assertRaises(AssertionError):
            pool_mod.verify_bank_rows(pool, sources, sample=4)

    def test_scalar_per_expert_globals_are_compared_as_bytes(self):
        device = torch.device("cpu")
        sources = make_sources(1, 4)
        for src in sources:  # a [E] per-expert global, as Marlin's scale_2 is
            src["w13_weight_scale_2"] = torch.arange(4, dtype=torch.float32) + 0.5
            src["w2_weight_scale_2"] = torch.arange(4, dtype=torch.float32) + 1.5
        pool = pool_mod.GlobalPool(device, sources[0], [2], 1)
        for name in gp.TENSORS:
            pool.bank[name][:2].copy_(sources[0][name][:2])
        self.assertEqual(
            pool_mod.verify_bank_rows(pool, sources, sample=2)["rows_checked"], 2
        )
        pool.bank["w2_weight_scale_2"][1] += 1
        with self.assertRaises(AssertionError):
            pool_mod.verify_bank_rows(pool, sources, sample=2)
