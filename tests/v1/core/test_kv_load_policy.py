# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StaticThresholdWindowPolicy.

The window vectors come from the SHARED semantics table (__shapes__), which is
also consumed by the LMCache-Ascend host tests — the two implementations must
not drift. Pure policy + config tests; no model, no GPU.
"""

from __future__ import annotations

# Standard
import unittest

# Local
from vllm.config.kv_transfer import KVLoadPolicyConfig
from vllm.v1.core.kv_load_policy import StaticThresholdWindowPolicy

# Shared semantic vector table (mirror into LMCache-Ascend host tests too).
from .__shapes__ import INVALID_CONFIGS, WINDOW_VECTORS  # isort: skip


def make(tau1: int, tau2: object) -> StaticThresholdWindowPolicy:
    return StaticThresholdWindowPolicy(KVLoadPolicyConfig(tau1=tau1, tau2=tau2))


class WindowSemanticsTest(unittest.TestCase):
    def test_shared_vectors(self):
        for matched, tau1, tau2, expect in WINDOW_VECTORS:
            with self.subTest(matched=matched, tau1=tau1, tau2=tau2):
                p = make(tau1, tau2)
                self.assertEqual(p.should_load(matched_tokens=matched), expect)

    def test_describe_snapshot(self):
        d = make(1024, 23296).describe()
        self.assertEqual(d["mode"], "static_threshold_window")
        self.assertEqual(d["tau1"], 1024)
        self.assertEqual(d["tau2"], 23296)


class ConfigValidationTest(unittest.TestCase):
    def test_invalid_configs_rejected(self):
        for kwargs in INVALID_CONFIGS:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KVLoadPolicyConfig(**kwargs)  # type: ignore[arg-type]

    def test_unknown_extra_field_rejected(self):
        # @config dataclass forbids extra fields -> typo fails at parse time
        with self.assertRaises(ValueError):
            KVLoadPolicyConfig(tao1=1024)  # type: ignore[call-arg]

    def test_dict_form_parses(self):
        cfg = KVLoadPolicyConfig(mode="static_threshold_window", tau1=1024, tau2=23296)
        self.assertEqual((cfg.tau1, cfg.tau2), (1024, 23296))


if __name__ == "__main__":
    unittest.main()
