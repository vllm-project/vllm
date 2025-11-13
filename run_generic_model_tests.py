#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Runner script for generic model support tests.

This bypasses pytest's conftest loading issues while still demonstrating
all the functionality.
"""

import sys
import traceback

from tests.test_generic_model_support import TestGenericModelSupport


def run_tests():
    """Run all tests and report results."""
    test_suite = TestGenericModelSupport()

    tests = [
        ("Flash Attention Forward", test_suite.test_flash_attention_forward),
        ("Flash Attention Backward", test_suite.test_flash_attention_backward),
        ("Model Forward", test_suite.test_model_forward),
        ("Model Training", test_suite.test_model_training),
        ("vLLM Wrapper", test_suite.test_vllm_wrapper),
    ]

    print("\n" + "=" * 70)
    print("Running Generic Model Support Tests")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            print(f"Testing {name}...", end=" ")
            test_fn()
            print("✓ PASS")
            passed += 1
        except Exception as e:
            print("✗ FAIL")
            failed += 1
            errors.append((name, e, traceback.format_exc()))

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if errors:
        print("\nErrors:\n")
        for name, error, tb in errors:
            print(f"{name}:")
            print(tb)
            print()

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
