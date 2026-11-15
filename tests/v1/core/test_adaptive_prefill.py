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

    def test_feedback_backoff_and_slow_growth(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=0.01)
        b = c.begin(0, 0, 1000, {"a"})
        c.commit(1, b, {"a": 100}, {"a": 0}, 0, 0)
        c.complete(1, 20)
        self.assertEqual(c.scale, 0.7)
        self.assertGreater(c.token_ms, 0.01)
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
        for kwargs in ({"target_ms": 0}, {"token_ms": float("nan")},
                       {"min_share": 1}, {"max_wait_ms": -1}):
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
        b = c.begin(.01, 1, 100, {"a"})
        self.assertEqual(b.limit("a", 100, 100000), 1)
        b.admit("a", 1, 100000)
        self.assertEqual(b.limit("b", 100, 100000), 0)
        c.commit(1, b, {"a": 1}, {"a": 100000}, 1, .01)
        self.assertEqual(c.begin(1, 1, 100, {"a"}).limit("a", 100, 100000), 0)

    def test_fair_share_leaves_room_for_waiting_prompt(self):
        c = AdaptivePrefillController(target_ms=10, token_ms=.1)
        b = c.begin(0, 0, 1000, {"long", "short"})
        n = b.limit("long", 1000, 0)
        b.admit("long", n, 0)
        self.assertEqual(n, 50)
        self.assertEqual(b.limit("short", 16, 0), 16)


if __name__ == "__main__":
    unittest.main()
