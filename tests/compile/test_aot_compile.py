# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import pickle
import tempfile
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.compilation.caching import VllmSerializableFunction
from vllm.compilation.decorators import save_compile_cache, support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer


def reference_fn(x: torch.Tensor):
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(3000):
        x = x + x.shape[0]
    return x


@support_torch_compile
class CompiledMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn(x)


def make_vllm_config() -> VllmConfig:
    return VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
        )
    )


@contextmanager
def use_vllm_config(vllm_config: VllmConfig):
    with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
        yield


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_no_dynamo_cache_entry(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        vllm_config = make_vllm_config()
        args = (torch.randn(10, 10),)
        expected = reference_fn(*args)
        with use_vllm_config(vllm_config):
            m.setenv("VLLM_USE_AOT_COMPILE", "0")
            m.setenv("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            with (
                pytest.raises(RuntimeError, match="Detected recompile"),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                CompiledMod(vllm_config=vllm_config)(*args)

            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_force_aot_load(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdirname, monkeypatch.context() as m:
        args = (torch.randn(10, 10),)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", "1")
        m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        vllm_config = make_vllm_config()
        with use_vllm_config(vllm_config), pytest.raises(FileNotFoundError):
            CompiledMod(vllm_config=vllm_config)(*args)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_save_and_load(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                expected = compiled_mod(*args)
                save_compile_cache(compiled_mod)

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                ret = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(ret, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_shape_env(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the shape environment is correctly serialized and preserved
    when loading from cache.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"
                save_compile_cache(compiled_mod)

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"


class TestInductorCompiledArtifacts:
    def test_init(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()
        assert cache.submodule_bytes == {}
        assert cache.submodule_bytes_store == {}
        assert cache.loaded_submodule_store == {}

    def test_insert_new_artifact(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()
        test_data = b"test_artifact_data"
        submod_name = "test_submod"
        shape = "s1"

        hasher = hashlib.sha256()
        hasher.update(test_data)
        expected_hash = hasher.hexdigest()

        cache.insert(submod_name, shape, test_data)

        assert f"{submod_name}_{shape}" in cache.submodule_bytes
        assert cache.submodule_bytes[f"{submod_name}_{shape}"] == expected_hash
        assert expected_hash in cache.submodule_bytes_store
        assert cache.submodule_bytes_store[expected_hash] == test_data

    def test_insert_duplicate_artifact(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        test_data = b"duplicate_test_data"
        submod_name1 = "submod1"
        submod_name2 = "submod2"
        shape = "s2"

        cache.insert(submod_name1, shape, test_data)
        cache.insert(submod_name2, shape, test_data)

        hash1 = cache.submodule_bytes[f"{submod_name1}_{shape}"]
        hash2 = cache.submodule_bytes[f"{submod_name2}_{shape}"]
        assert hash1 == hash2

        assert len(cache.submodule_bytes_store) == 1
        assert len(cache.submodule_bytes) == 2

    def test_get_artifact(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()
        test_data = b"retrievable_data"
        submod_name = "mod1"
        shape = "shape16"

        cache.insert(submod_name, shape, test_data)
        retrieved_data = cache.get(submod_name, shape)

        assert retrieved_data == test_data

    def test_get_nonexistent_artifact(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        with pytest.raises(KeyError):
            cache.get("nonexistent", "shape")

    def test_size_bytes(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        assert cache.size_bytes() == 0

        data1 = b"x" * 100
        data2 = b"y" * 200
        cache.insert("mod1", "shape1", data1)
        cache.insert("mod2", "shape2", data2)

        assert cache.size_bytes() == 300

    def test_num_artifacts_and_entries(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        assert cache.num_artifacts() == 0
        assert cache.num_entries() == 0

        cache.insert("mod1", "shape1", b"data1")
        cache.insert("mod2", "shape2", b"data2")
        assert cache.num_artifacts() == 2
        assert cache.num_entries() == 2

        cache.insert("mod3", "shape3", b"data1")
        assert cache.num_artifacts() == 2
        assert cache.num_entries() == 3

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_load_all_success(self, mock_deserialize):
        """Test successful loading of all artifacts"""
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        mock_artifact1 = Mock()
        mock_artifact2 = Mock()
        mock_deserialize.side_effect = [mock_artifact1, mock_artifact2]

        cache.insert("mod1", "shape1", pickle.dumps(b"data1"))
        cache.insert("mod2", "shape2", pickle.dumps(b"data2"))

        cache.load_all()

        assert len(cache.loaded_submodule_store) == 2
        assert mock_deserialize.call_count == 2

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_load_all_already_loaded(self, mock_deserialize):
        """Test that load_all skips if already loaded"""
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        mock_artifact = Mock()
        cache.submodule_bytes_store["hash1"] = pickle.dumps(b"data1")
        cache.loaded_submodule_store["hash1"] = mock_artifact

        cache.load_all()

        mock_deserialize.assert_not_called()

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_get_loaded_artifact(self, mock_deserialize):
        """Test retrieving loaded artifacts"""
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        mock_artifact = Mock()
        mock_deserialize.return_value = mock_artifact

        submod_name = "test_mod"
        shape = "test_shape"
        cache.insert(submod_name, shape, pickle.dumps(b"test_data"))
        cache.load_all()

        retrieved_artifact = cache.get_loaded(submod_name, shape)
        assert retrieved_artifact == mock_artifact

    def test_getstate_setstate(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        cache.insert("mod1", "shape1", b"data1")
        cache.insert("mod2", "shape2", b"data2")

        cache.loaded_submodule_store["hash1"] = Mock()

        state = cache.__getstate__()

        assert "submodule_bytes" in state
        assert "submodule_bytes_store" in state
        assert "loaded_submodule_store" not in state

        new_cache = VllmSerializableFunction.InductorCompiledArtifacts()
        new_cache.__setstate__(state)

        assert new_cache.submodule_bytes == cache.submodule_bytes
        assert new_cache.submodule_bytes_store == cache.submodule_bytes_store
        assert new_cache.loaded_submodule_store == {}

    def test_pickle_roundtrip(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        test_data1 = b"pickle_test_data_1"
        test_data2 = b"pickle_test_data_2"
        cache.insert("mod1", "shape1", test_data1)
        cache.insert("mod2", "shape2", test_data2)

        pickled_data = pickle.dumps(cache)
        restored_cache = pickle.loads(pickled_data)

        assert restored_cache.get("mod1", "shape1") == test_data1
        assert restored_cache.get("mod2", "shape2") == test_data2
        assert restored_cache.num_artifacts() == cache.num_artifacts()
        assert restored_cache.num_entries() == cache.num_entries()
        assert restored_cache.size_bytes() == cache.size_bytes()

        assert len(restored_cache.loaded_submodule_store) == 0


class TestInductorCompiledArtifactsIntegration:
    def test_add_pickle_unpickle(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        artifacts = {
            ("mod1", "shape1"): b"m1s1_artifact",
            ("mod1", "shape2"): b"m1s2_artifact",
            ("mod2", "shape1"): b"m2s1_artifact",
            ("mod2", "shape2"): b"m2s2_artifact",
        }

        for (submod, shape), data in artifacts.items():
            cache.insert(submod, shape, data)

        assert cache.num_entries() == 4
        assert cache.num_artifacts() == 4

        for (submod, shape), expected_data in artifacts.items():
            retrieved_data = cache.get(submod, shape)
            assert retrieved_data == expected_data

        pickled = pickle.dumps(cache)
        restored_cache = pickle.loads(pickled)

        for (submod, shape), expected_data in artifacts.items():
            retrieved_data = restored_cache.get(submod, shape)
            assert retrieved_data == expected_data

    def test_deduplication(self):
        cache = VllmSerializableFunction.InductorCompiledArtifacts()

        shared_data = b"shared_artifact_data" * 1000

        cache.insert("mod1", "shape1", shared_data)
        cache.insert("mod2", "shape1", shared_data)
        cache.insert("mod1", "shape2", shared_data)
        cache.insert("mod3", "shape3", shared_data)

        assert cache.num_entries() == 4
        assert cache.num_artifacts() == 1
        assert cache.size_bytes() == len(shared_data)

        for submod, shape in [
            ("mod1", "shape1"),
            ("mod2", "shape1"),
            ("mod1", "shape2"),
            ("mod3", "shape3"),
        ]:
            assert cache.get(submod, shape) == shared_data
