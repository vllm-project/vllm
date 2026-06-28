# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import mock_open, patch

import pytest

from vllm._cpu_detection import _get_cfs_cpu_limit, get_available_cpus


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear the lru_cache on get_available_cpus between tests."""
    get_available_cpus.cache_clear()
    yield
    get_available_cpus.cache_clear()


def _patch_sched_getaffinity(**kwargs):
    """Patch os.sched_getaffinity, using create=True for macOS compat."""
    return patch("os.sched_getaffinity", create=True, **kwargs)


class TestGetAvailableCpus:
    """Tests for CFS-aware CPU detection."""

    def test_cgroup_v2_detection(self):
        """CFS quota from cgroup v2 (/sys/fs/cgroup/cpu.max)."""
        mock_file = mock_open(read_data="400000 100000\n")
        with patch("builtins.open", mock_file):
            assert get_available_cpus() == 4

    def test_cgroup_v2_fractional_rounds_up(self):
        """1.5 CPUs (150000/100000) should round up to 2."""
        mock_file = mock_open(read_data="150000 100000\n")
        with patch("builtins.open", mock_file):
            assert get_available_cpus() == 2

    def test_cgroup_v2_max_means_unlimited(self):
        """'max 100000' means no CFS limit -- should fall through."""

        def fake_open(path, *args, **kwargs):
            if path == "/sys/fs/cgroup/cpu.max":
                return mock_open(read_data="max 100000\n")()
            raise FileNotFoundError(path)

        with (
            patch("builtins.open", side_effect=fake_open),
            _patch_sched_getaffinity(return_value=set(range(8))),
        ):
            assert get_available_cpus() == 8

    def test_cgroup_v1_detection(self):
        """CFS quota from cgroup v1 files."""

        def fake_open(path, *args, **kwargs):
            if path == "/sys/fs/cgroup/cpu.max":
                raise FileNotFoundError(path)
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
                return mock_open(read_data="800000\n")()
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
                return mock_open(read_data="100000\n")()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=fake_open):
            assert get_available_cpus() == 8

    def test_cgroup_v1_no_limit(self):
        """cgroup v1 quota of -1 means no limit -- should fall through."""

        def fake_open(path, *args, **kwargs):
            if path == "/sys/fs/cgroup/cpu.max":
                raise FileNotFoundError(path)
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
                return mock_open(read_data="-1\n")()
            raise FileNotFoundError(path)

        with (
            patch("builtins.open", side_effect=fake_open),
            _patch_sched_getaffinity(return_value=set(range(16))),
        ):
            assert get_available_cpus() == 16

    def test_sched_getaffinity_fallback(self):
        """When no cgroup files exist, use sched_getaffinity."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            _patch_sched_getaffinity(return_value=set(range(32))),
        ):
            assert get_available_cpus() == 32

    def test_cpu_count_last_resort(self):
        """When everything else fails, fall back to os.cpu_count()."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            _patch_sched_getaffinity(side_effect=OSError),
            patch("os.cpu_count", return_value=64),
        ):
            assert get_available_cpus() == 64


class TestGetCfsCpuLimit:
    """Tests for _get_cfs_cpu_limit (CFS-only detection)."""

    def test_returns_value_when_cfs_quota_set(self):
        """Returns CPU count when CFS quota is configured."""
        mock_file = mock_open(read_data="400000 100000\n")
        with patch("builtins.open", mock_file):
            assert _get_cfs_cpu_limit() == 4

    def test_returns_none_when_no_quota(self):
        """Returns None on bare metal (no cgroup files)."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _get_cfs_cpu_limit() is None

    def test_returns_none_when_unlimited(self):
        """Returns None when cgroup v2 says 'max' (unlimited)."""

        def fake_open(path, *args, **kwargs):
            if path == "/sys/fs/cgroup/cpu.max":
                return mock_open(read_data="max 100000\n")()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=fake_open):
            assert _get_cfs_cpu_limit() is None
