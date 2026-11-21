# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for load-time thread scoping, without loading a model.

Contract: cap threads only within the decorated call; preserve return values,
exceptions and prior thread settings. Mock Torch thread control to avoid changing
process-wide settings in the test runner.
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "load_threads", ROOT / "vllm/model_executor/load_threads.py"
)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class LoadThreadsTest(unittest.TestCase):
    def run_scoped(self, configured, original, fail=False):
        state = [original]
        changes = []
        seen = []

        def set_threads(value):
            changes.append(value)
            state[0] = value

        @HELPER.limit_load_threads
        def load(value):
            seen.append(state[0])
            if fail:
                raise RuntimeError("weight loading failed")
            return value

        with patch.dict(os.environ, {"VLLM_LOAD_TORCH_THREADS": configured}), \
             patch.object(HELPER.torch, "get_num_threads", lambda: state[0]), \
             patch.object(HELPER.torch, "set_num_threads", set_threads):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "weight loading failed"):
                    load(23)
            else:
                self.assertEqual(load(23), 23)
        self.assertEqual(state[0], original)
        return changes, seen

    def test_cap_is_scoped_and_result_preserved(self):
        self.assertEqual(self.run_scoped("4", 64), ([4, 64], [4]))

    def test_original_threads_restored_after_loader_exception(self):
        self.assertEqual(self.run_scoped("4", 64, fail=True), ([4, 64], [4]))

    def test_disabled_and_already_small_thread_pools_are_untouched(self):
        for configured, original in [("0", 64), ("-1", 64), ("4", 2), ("4", 4)]:
            with self.subTest(configured=configured, original=original):
                self.assertEqual(self.run_scoped(configured, original), ([], [original]))


if __name__ == "__main__":
    unittest.main()
