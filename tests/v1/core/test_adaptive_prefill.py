# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest

from vllm.v1.core.sched.adaptive_prefill import AdaptivePrefillController, PrefillBudget


class TestAdaptivePrefill(unittest.TestCase):
    def test_respects_aggregate_time(self):
        b = PrefillBudget(10, 0.1, 1000)
        self.assertEqual(b.limit("a", 1000, 0), 100)
        b.admit("a", 60, 0)
        self.assertEqual(b.limit("b", 1000, 0), 40)
        b.refund("a")
        self.assertEqual(b.limit("b", 1000, 0), 100)

    def test_context_and_token_limit(self):
        b = PrefillBudget(10, 0.1, 80)
        self.assertEqual(b.limit("a", 1000, 0), 80)
        self.assertEqual(b.limit("a", 1000, 8192), 50)
        self.assertEqual(b.limit("a", 8, 8192), 8)

    def test_pending_work_is_reserved(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=1)
        b = c.begin(0, 0, 100, {"a"})
        c.commit(1, b, {"a": 10}, {"a": 0}, 0, 0)
        self.assertEqual(c.begin(0, 1, 100, {"a"}).limit_ms, 0)
        c.complete(1, 10)
        self.assertGreater(c.begin(0, 1, 100, {"a"}).limit_ms, 0)

    def test_only_successful_allocations_spend_credit(self):
        c = AdaptivePrefillController()
        b = c.begin(0, 1, 100, {"a", "b"})
        b.admit("a", 100, 0)
        c.commit(1, b, {"b": 1}, {"b": 0}, 0, 0)
        self.assertAlmostEqual(c.pending[1].units, 1)
        self.assertAlmostEqual(c.credit_ms, 3.98)

    def test_age_credit_prevents_zero_allowance_forever(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=1, max_wait_ms=20)
        c.decode_ms = 10
        c.begin(0, 2, 100, {"a"})
        b = c.begin(0.03, 2, 100, {"a"})
        self.assertGreater(b.limit("a", 100, 0), 0)
        self.assertLessEqual(b.limit_ms, c.target_ms)

    def test_idle_and_removed_requests(self):
        c = AdaptivePrefillController()
        c.begin(0, 0, 10, {"a"})
        c.begin(1, 0, 10, set())
        self.assertEqual(c.last_progress, {})
        self.assertEqual(c.credit_ms, 0)

    def test_feedback_backoff_does_not_fit_one_prefill_sample(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=0.01)
        b = c.begin(0, 0, 1000, {"a"})
        c.commit(1, b, {"a": 100}, {"a": 0}, 0, 0)
        c.complete(1, 20)
        self.assertEqual(c.scale, 0.7)
        self.assertEqual(c.token_ms, 0.01)
        c.commit(2, b, {"d": 1}, {}, 1, 0)
        c.complete(2, 1)
        self.assertAlmostEqual(c.scale, 0.72)

    def test_invalid_or_absent_timing_releases_reservation(self):
        for value in (None, float("nan"), float("inf"), 0, -1):
            c = AdaptivePrefillController()
            b = c.begin(0, 0, 10, {"a"})
            c.commit(1, b, {"a": 1}, {"a": 0}, 0, 0)
            c.complete(1, value)
            self.assertEqual(c.pending, {})
            self.assertEqual(c.token_ms, 0.02)

    def test_invalid_configuration(self):
        for kwargs in (
            {"target_ms": 0},
            {"token_ms": float("nan")},
            {"min_share": 1},
            {"max_wait_ms": -1},
        ):
            with self.assertRaises(ValueError):
                AdaptivePrefillController(**kwargs)

    def test_bounded_credit_under_permanent_overload(self):
        c = AdaptivePrefillController()
        c.decode_ms = 100
        for i in range(10000):
            c.begin(i, 10, 100, {"a"})
        self.assertLessEqual(c.credit_ms, c.target_ms)
        self.assertEqual(len(c.last_progress), 1)

    def test_out_of_order_completion(self):
        c = AdaptivePrefillController()
        b = c.begin(0, 0, 100, {"a"})
        for step in (1, 2):
            c.commit(step, b, {"a": 1}, {"a": 0}, 0, 0)
        c.complete(2, 1)
        self.assertEqual(set(c.pending), {1})
        c.complete(1, 1)
        self.assertEqual(c.pending, {})

    def test_aged_expensive_token_makes_soft_slo_progress(self):
        c = AdaptivePrefillController(target_ms=1, token_ms=10, max_wait_ms=1)
        c.begin(0, 1, 100, {"a"})
        b = c.begin(0.01, 1, 100, {"a"})
        self.assertEqual(b.limit("a", 100, 100000), 1)
        b.admit("a", 1, 100000)
        self.assertEqual(b.limit("b", 100, 100000), 0)
        c.commit(1, b, {"a": 1}, {"a": 100000}, 1, 0.01)
        second = c.begin(1, 1, 100, {"a"})
        self.assertEqual(second.limit("a", 100, 100000), 1)
        c.commit(2, second, {"a": 1}, {"a": 100000}, 1, 1)
        self.assertEqual(c.begin(2, 1, 100, {"a"}).limit("a", 100, 100000), 0)
        c.complete(1, None)
        self.assertEqual(c.begin(3, 1, 100, {"a"}).limit("a", 100, 100000), 1)

    def test_decode_batch_overhead_is_not_multiplied_by_population(self):
        c = AdaptivePrefillController()
        for step, count in enumerate((1, 8, 4)):
            b = c.begin(step, count, 1000, set())
            c.commit(step, b, {}, {}, count, step)
            c.complete(step, 14.76 + 0.53 * count)
        b = c.begin(4, 16, 1000, {"a"})
        self.assertGreater(b.limit("a", 1000, 0), 0)

    def test_isolated_prefill_spike_does_not_poison_marginal_cost(self):
        c = AdaptivePrefillController()
        b = c.begin(0, 1, 1000, set())
        c.commit(0, b, {}, {}, 1, 0)
        c.complete(0, 15.29)
        for step, elapsed in enumerate((163.6, 21.76, 21.76), 1):
            b = c.begin(step, 0, 1000, {"a"})
            c.commit(step, b, {"a": 100}, {"a": 0}, 0, step)
            c.complete(step, elapsed)
        b = c.begin(4, 0, 1000, {"a", "b", "c", "d"})
        self.assertGreaterEqual(b.limit("a", 1000, 0), 32)

    def test_zero_fixed_overhead_still_learns_prefill_cost(self):
        c = AdaptivePrefillController()
        for step, count in enumerate((1, 8)):
            b = c.begin(step, count, 1000, set())
            c.commit(step, b, {}, {}, count, step)
            c.complete(step, 0.5 * count)
        for step in range(2, 5):
            b = c.begin(step, 0, 1000, {"a"})
            c.commit(step, b, {"a": 100}, {"a": 0}, 0, step)
            c.complete(step, 7)
        self.assertLess(c.begin(5, 0, 1000, {"a"}).limit("a", 1000, 0), 1000)

    def test_prefill_sizes_identify_overhead_without_decode(self):
        c = AdaptivePrefillController()
        for step, count in enumerate((100, 200, 300, 100)):
            b = c.begin(step, 0, 1000, {"a"})
            c.commit(step, b, {"a": count}, {"a": 0}, 0, step)
            c.complete(step, 14.76 + 0.07 * count)
        self.assertGreater(c.step_ms, 14)
        self.assertGreater(c.token_ms, 0.02)

    def test_prefill_pipeline_is_bounded_and_releases_on_feedback(self):
        c = AdaptivePrefillController()
        for step in range(2):
            b = c.begin(step, 0, 1000, {"a"})
            count = b.limit("a", 1000, 0)
            self.assertGreater(count, 0)
            c.commit(step, b, {"a": count}, {"a": 0}, 0, step)
        self.assertEqual(c.begin(2, 0, 1000, {"a"}).limit("a", 1000, 0), 0)
        c.complete(0, None)
        self.assertGreater(c.begin(3, 0, 1000, {"a"}).limit("a", 1000, 0), 0)

    def test_fair_share_leaves_room_for_waiting_prompt(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=0.1)
        b = c.begin(0, 0, 1000, {"long", "short"})
        n = b.limit("long", 1000, 0)
        b.admit("long", n, 0)
        self.assertEqual(n, 50)
        self.assertEqual(b.limit("short", 16, 0), 16)


if __name__ == "__main__":
    unittest.main()
