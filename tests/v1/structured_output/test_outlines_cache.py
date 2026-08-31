# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import outlines_core as oc
import pytest

from vllm.v1.structured_output.utils import OutlinesDiskCache

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def vocab():
    return oc.Vocabulary(3, {b"a": [0], b"b": [1], b"ab": [2]})


@pytest.fixture
def index(vocab):
    return oc.Index("ab", vocab)


@pytest.fixture
def cache(tmp_path):
    return OutlinesDiskCache(str(tmp_path))


class TestOutlinesDiskCache:
    def test_store_and_retrieve_index(self, cache, index):
        cache["test_key"] = index
        restored = cache["test_key"]
        assert restored.get_initial_state() == index.get_initial_state()
        assert restored.get_transitions() == index.get_transitions()
        assert restored.get_final_states() == index.get_final_states()

    def test_store_and_retrieve_string(self, cache):
        cache.set("__version__", "0.2.14")
        assert cache.get("__version__") == "0.2.14"

    def test_contains(self, cache, index):
        assert "missing" not in cache
        cache["key"] = index
        assert "key" in cache

    def test_get_default(self, cache):
        assert cache.get("missing", "fallback") == "fallback"
        assert cache.get("missing") is None

    def test_missing_key_raises(self, cache):
        with pytest.raises(KeyError):
            _ = cache["missing"]

    def test_clear(self, cache, index):
        cache["key"] = index
        cache.set("__version__", "1.0")
        cache.clear()
        assert "key" not in cache
        assert "__version__" not in cache

    def test_overwrite(self, cache, vocab):
        index1 = oc.Index("a", vocab)
        index2 = oc.Index("b", vocab)
        cache["key"] = index1
        cache["key"] = index2
        restored = cache["key"]
        assert restored.get_transitions() == index2.get_transitions()

    def test_persistence_across_instances(self, tmp_path, index):
        cache1 = OutlinesDiskCache(str(tmp_path))
        cache1["key"] = index

        cache2 = OutlinesDiskCache(str(tmp_path))
        restored = cache2["key"]
        assert restored.get_transitions() == index.get_transitions()

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = f"{tmpdir}/nested/cache/dir"
            cache = OutlinesDiskCache(subdir)
            cache.set("__version__", "1.0")
            assert cache.get("__version__") == "1.0"

    def test_version_invalidation_flow(self, cache, index):
        """Simulates the version-check logic in get_outlines_cache()."""
        cache.set("__version__", "0.2.13")
        cache["key"] = index

        cached_version = cache.get("__version__")
        new_version = "0.2.14"
        if cached_version != new_version:
            cache.clear()
        cache.set("__version__", new_version)

        assert cache.get("__version__") == "0.2.14"
        assert "key" not in cache
