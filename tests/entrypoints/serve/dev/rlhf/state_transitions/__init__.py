# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP lifecycle coverage: pause/resume, sleep/wake, transfer, and checker.

Each test module owns its endpoint or compound lifecycle contract. Engine,
worker, allocator, and loader regressions remain in their component suites;
the parent package docstring indexes those complementary tests.
"""
