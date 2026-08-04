# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache.v1.storage_backend.path_sharder import PathSharder


class TestNixlMultipath:
    """Test cases for NIXL multipath functionality using PathSharder."""

    def test_path_sharder_single_path(self):
        """Test PathSharder with a single path string."""
        path = "/tmp/nixl/cache"
        path_sharding = "by_gpu"
        dst_device = "cuda:0"

        sharder = PathSharder(path, path_sharding, dst_device)

        # Should return the same path since there's only one
        assert sharder.selected == path
        assert sharder.all_paths == [path]

    def test_path_sharder_multiple_paths(self):
        """Test PathSharder with a CSV string of paths."""
        paths = "/tmp/nixl/cache0,/tmp/nixl/cache1,/tmp/nixl/cache2"
        path_sharding = "by_gpu"

        # Test with cuda:0 (device 0)
        sharder = PathSharder(paths, path_sharding, "cuda:0")
        assert sharder.selected == "/tmp/nixl/cache0"

        # Test with cuda:1 (device 1)
        sharder = PathSharder(paths, path_sharding, "cuda:1")
        assert sharder.selected == "/tmp/nixl/cache1"

        # Test with cuda:2 (device 2)
        sharder = PathSharder(paths, path_sharding, "cuda:2")
        assert sharder.selected == "/tmp/nixl/cache2"

        # Test with cuda:3 (should wrap around to cache0)
        sharder = PathSharder(paths, path_sharding, "cuda:3")
        assert sharder.selected == "/tmp/nixl/cache0"

    def test_path_sharder_empty_path(self):
        """Test PathSharder with empty path."""
        with pytest.raises(ValueError, match="At least one path must be provided"):
            PathSharder("", "by_gpu", "cuda:0")

    def test_path_sharder_unsupported_sharding(self):
        """Test PathSharder with unsupported path sharding."""
        path = "/tmp/nixl/cache"
        with pytest.raises(ValueError, match="Unsupported path sharding"):
            PathSharder(path, "unsupported_sharding", "cuda:0")

    def test_path_sharder_cpu_device(self):
        """Test PathSharder with CPU device."""
        path = "/tmp/nixl/cache"
        path_sharding = "by_gpu"
        dst_device = "cpu"

        sharder = PathSharder(path, path_sharding, dst_device)
        assert sharder.selected == path
