# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for grammar-compilation thread pool sizing.

The pool must stay small on many-core hosts: ``multiprocessing.cpu_count()``
is not cgroup-aware, so sizing the pool by host CPU count oversubscribes
container CPU quotas (CFS throttling) and stalls decode while grammars
compile — e.g. an 86-worker pool inside a Kubernetes pod limited to 6 CPUs
on a 172-core node.
"""

from unittest.mock import patch

import pytest

from vllm.v1.structured_output.utils import grammar_compile_max_workers

pytestmark = pytest.mark.cpu_test


class TestGrammarCompileMaxWorkers:
    @pytest.mark.parametrize(
        ("cpu_count", "expected"),
        [
            (1, 1),
            (2, 1),
            (4, 2),
            (16, 8),
            (17, 8),
            (172, 8),  # many-core host: must stay capped
        ],
    )
    def test_capped_at_8(self, cpu_count: int, expected: int):
        assert grammar_compile_max_workers(cpu_count) == expected

    def test_defaults_to_host_cpu_count(self):
        with patch("multiprocessing.cpu_count", return_value=32):
            assert grammar_compile_max_workers() == 8

    def test_env_override(self):
        with patch("vllm.envs.VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", 4):
            assert grammar_compile_max_workers(172) == 4

    def test_env_override_zero_means_auto(self):
        with patch("vllm.envs.VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", 0):
            assert grammar_compile_max_workers(172) == 8

    def test_at_least_one_worker(self):
        assert grammar_compile_max_workers(0) == 1
