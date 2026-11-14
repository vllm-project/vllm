# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic compile-pool limits and explicit operator overrides."""

import multiprocessing
import os

import pytest

import vllm.envs as envs
from vllm.utils.torch_utils import available_cpu_count
from vllm.v1.structured_output.utils import grammar_compile_max_workers

pytestmark = pytest.mark.cpu_test


class TestGrammarCompileMaxWorkers:
    @pytest.fixture(autouse=True)
    def automatic_sizing(self, monkeypatch):
        envs.disable_envs_cache()
        monkeypatch.setenv("VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", "0")

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
        assert grammar_compile_max_workers(cpu_count) == min(
            expected, available_cpu_count()
        )

    def test_defaults_to_host_cpu_count(self):
        assert grammar_compile_max_workers() == max(
            1, min(multiprocessing.cpu_count() // 2, available_cpu_count(), 8)
        )

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", "4")
        assert grammar_compile_max_workers(172) == 4

    def test_env_override_zero_means_auto(self, monkeypatch):
        monkeypatch.setenv("VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", "0")
        assert grammar_compile_max_workers(172) == min(available_cpu_count(), 8)

    def test_at_least_one_worker(self):
        assert grammar_compile_max_workers(0) == 1

    @pytest.mark.parametrize("usable_cpus", [1, 4, 6])
    def test_automatic_sizing_respects_available_cpus(self, monkeypatch, usable_cpus):
        # Supply OS CPU facts, exercising the real available-CPU helper.
        # A real cgroup quota may impose a still tighter limit on Linux CI.
        monkeypatch.setattr(os, "cpu_count", lambda: usable_cpus)
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: set(range(usable_cpus)), raising=False
        )

        assert grammar_compile_max_workers(172) == available_cpu_count()

    def test_explicit_override_can_exceed_available_cpus(self, monkeypatch):
        monkeypatch.setenv("VLLM_STRUCTURED_OUTPUT_MAX_COMPILE_WORKERS", "12")
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)

        assert grammar_compile_max_workers(172) == 12
