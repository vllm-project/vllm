# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`lmcache.v1.storage_backend.path_sharder`."""

# Standard
from unittest.mock import patch
import os
import shutil
import tempfile

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.path_sharder import PathSharder


class TestPathSharder:
    """Tests for PathSharder class."""

    def test_single_path(self):
        d = tempfile.mkdtemp()
        try:
            s = PathSharder(d, strategy="by_gpu", dst_device="cuda:0")
            assert s.selected == d
            assert s.all_paths == [d]
            assert s.strategy == "by_gpu"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_multi_path_selects_by_device_id(self):
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            for i, d in enumerate(dirs):
                s = PathSharder(csv, strategy="by_gpu", dst_device=f"cuda:{i}")
                assert s.selected == d
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_modulo_wraps(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:4")
            # 4 % 2 == 0
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_create_dirs(self):
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            csv = ",".join(paths)
            PathSharder(csv, strategy="by_gpu", dst_device="cuda:0", create_dirs=True)
            for p in paths:
                assert os.path.isdir(p)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_no_create_dirs_by_default(self):
        base = tempfile.mkdtemp()
        try:
            new_dir = os.path.join(base, "should_not_exist")
            PathSharder(new_dir, strategy="by_gpu", dst_device="cuda:0")
            assert not os.path.exists(new_dir)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_empty_csv_raises(self):
        with pytest.raises(ValueError, match="At least one path"):
            PathSharder("", strategy="by_gpu", dst_device="cuda:0")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="At least one path"):
            PathSharder("  , ,  ", strategy="by_gpu", dst_device="cuda:0")

    def test_unsupported_strategy_raises(self):
        d = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="Unsupported path sharding"):
                PathSharder(d, strategy="round_robin", dst_device="cuda:0")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_strips_whitespace(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = f"  {dirs[0]}  ,  {dirs[1]}  "
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:0")
            assert s.all_paths == dirs
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_cpu_device_selects_first_path(self, _avail):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cpu")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_all_paths_returns_copy(self):
        d = tempfile.mkdtemp()
        try:
            s = PathSharder(d, strategy="by_gpu", dst_device="cuda:0")
            paths = s.all_paths
            paths.append("/rogue")
            assert "/rogue" not in s.all_paths
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # -- device-resolution edge cases (exercised via public API) -----------

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=1,
    )
    def test_bare_cuda_uses_current_device(self, _cur, _avail):
        """Bare 'cuda' resolves to torch_dev.current_device()."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda")
            assert s.selected == dirs[1]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_bare_cuda_no_gpu_selects_first(self, _avail):
        """Bare 'cuda' with no GPU falls back to device 0."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=2,
    )
    def test_cpu_device_always_selects_first(self, _cur, _avail):
        """'cpu' always resolves to index 0, even when CUDA is available."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cpu")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=2,
    )
    def test_malformed_device_empty_index_falls_back(self, _cur, _avail):
        """'cuda:' (no int) falls back to current_device."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:")
            assert s.selected == dirs[2]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_malformed_device_non_numeric_falls_back(self, _avail):
        """'cuda:foo' falls back to 0 when CUDA is unavailable."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:foo")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
