# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.utils.ompmultiprocessing — OMP-aware multiprocessing."""

import logging
import os
import platform
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.ompmultiprocessing import OMPProcessManager, parse_omp_places

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        platform.system() != "Linux",
        reason="OMP multiprocessing tests require Linux",
    ),
]


def _get_cpu_count():
    """Return the number of CPUs available to this process."""
    return len(os.sched_getaffinity(0))


def _make_config(local_world_size=1, dp_rank=None, dp_count=1, kv_transfer=None):
    """Create a mock VllmConfig for OMPProcessManager."""
    config = MagicMock()
    config.parallel_config.local_world_size = local_world_size
    config.parallel_config.data_parallel_rank_local = dp_rank
    config.parallel_config._api_process_count = dp_count
    config.kv_transfer_config = kv_transfer
    return config


class TestParseOmpPlaces:
    """Unit tests for the parse_omp_places helper."""

    def test_explicit_two_groups(self):
        result = parse_omp_places("{0-3},{4-7}")
        assert result is not None
        assert len(result) == 2
        assert result[0] == [0, 1, 2, 3]
        assert result[1] == [4, 5, 6, 7]

    def test_single_group(self):
        result = parse_omp_places("{0,1,2}")
        assert result is not None
        assert len(result) == 1
        assert result[0] == [0, 1, 2]

    def test_abstract_cores(self):
        assert parse_omp_places("cores") is None

    def test_abstract_threads(self):
        assert parse_omp_places("threads") is None

    def test_abstract_sockets(self):
        assert parse_omp_places("sockets") is None

    def test_no_braces(self):
        assert parse_omp_places("0-3") is None

    def test_empty_string(self):
        assert parse_omp_places("") is None


class TestOMPProcessManagerNobind:
    """Tests for nobind mode with OMP_PLACES parsing."""

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_without_omp_places(self, mock_platform, monkeypatch):
        """nobind without OMP_PLACES should skip OMP setup entirely."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.delenv("OMP_PLACES", raising=False)

        om = OMPProcessManager(_make_config())
        assert om.skip_setup is True
        assert om.cpu_lists == []

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_with_explicit_omp_places(self, mock_platform, monkeypatch):
        """nobind + explicit OMP_PLACES should parse and distribute."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        cpu_count = _get_cpu_count()
        half = cpu_count // 2
        places_str = "{" + f"0-{half - 1}" + "},{" + f"{half}-{cpu_count - 1}" + "}"

        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.setenv("OMP_PLACES", places_str)

        om = OMPProcessManager(_make_config())
        assert om.skip_setup is False
        assert len(om.cpu_lists) == 2
        assert om.cpu_lists[0] == list(range(half))
        assert om.cpu_lists[1] == list(range(half, cpu_count))

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_with_abstract_omp_places(self, mock_platform, monkeypatch):
        """nobind + abstract OMP_PLACES (e.g. 'cores') stays in skip mode."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.setenv("OMP_PLACES", "cores")

        om = OMPProcessManager(_make_config())
        assert om.skip_setup is True
        assert om.cpu_lists == []

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_configure_omp_envs_distributes(self, mock_platform, monkeypatch):
        """configure_omp_envs should set OMP env per worker in nobind mode."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        cpu_count = _get_cpu_count()
        half = cpu_count // 2
        places_str = "{" + f"0-{half - 1}" + "},{" + f"{half}-{cpu_count - 1}" + "}"

        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.setenv("OMP_PLACES", places_str)

        om = OMPProcessManager(_make_config())

        captured = []
        for local_rank in range(2):
            with om.configure_omp_envs(rank=local_rank, local_rank=local_rank):
                captured.append(
                    {
                        "OMP_PLACES": os.environ.get("OMP_PLACES"),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
                    }
                )

        assert len(captured) == 2
        assert captured[0]["OMP_NUM_THREADS"] == str(half)
        assert captured[1]["OMP_NUM_THREADS"] == str(cpu_count - half)
        assert captured[0]["OMP_PROC_BIND"] == "true"
        assert captured[1]["OMP_PROC_BIND"] == "true"
        assert captured[0]["OMP_PLACES"] != captured[1]["OMP_PLACES"]

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_warns_on_invalid_cpus(
        self, mock_platform, monkeypatch, caplog_vllm
    ):
        """OMP_PLACES with invalid CPU IDs should warn but still work."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        cpu_count = os.cpu_count() or 1
        bad_cpu = cpu_count + 10
        places_str = "{0},{" + str(bad_cpu) + "}"

        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.setenv("OMP_PLACES", places_str)

        with caplog_vllm.at_level(logging.WARNING):
            om = OMPProcessManager(_make_config())

        assert any("only has" in r.message for r in caplog_vllm.records)
        assert om.skip_setup is False
        assert len(om.cpu_lists) == 2

    @patch("vllm.utils.ompmultiprocessing.current_platform")
    def test_nobind_skip_when_no_omp_places(self, mock_platform, monkeypatch):
        """configure_omp_envs should yield without setting env in pure nobind."""
        mock_platform.is_cpu.return_value = True
        mock_platform.get_cpu_architecture.return_value = "x86"
        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        monkeypatch.delenv("OMP_PLACES", raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        om = OMPProcessManager(_make_config())
        with om.configure_omp_envs(rank=0, local_rank=0):
            assert os.environ.get("OMP_NUM_THREADS") is None
