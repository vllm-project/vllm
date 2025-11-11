# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import hashlib
import multiprocessing
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import vllm.model_executor.layers.activation
from vllm.compilation.caching import (
    StandaloneCompiledArtifacts,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.envs import disable_envs_cache
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ..utils import create_new_process_for_each_test


@pytest.fixture
def vllm_tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture that sets VLLM_CACHE_ROOT to a temporary directory."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    return tmp_path


def reference_fn(x: torch.Tensor):
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(3000):
        x = x + x.shape[0]
    return x


def reference_fn_tuple(x: torch.Tensor):
    """Reference function that returns a tuple of tensors."""
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(3000):
        x = x + x.shape[0]
    return x, x * 2


@support_torch_compile
class CompiledMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn(x)


@support_torch_compile
class CompiledModTuple(torch.nn.Module):
    """A compiled module that returns a tuple of tensors."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn_tuple(x)


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
    not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10"
)
def test_no_dynamo_cache_entry(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        vllm_config = make_vllm_config()
        args = (torch.randn(10, 10),)
        expected = reference_fn(*args)
        with use_vllm_config(vllm_config):
            m.setenv("VLLM_USE_AOT_COMPILE", "0")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            with (
                pytest.raises(RuntimeError, match="Detected recompile"),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                CompiledMod(vllm_config=vllm_config)(*args)
            disable_envs_cache()

            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10"
)
def test_force_aot_load(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdirname, monkeypatch.context() as m:
        args = (torch.randn(10, 10),)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
        m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        vllm_config = make_vllm_config()
        with use_vllm_config(vllm_config), pytest.raises(FileNotFoundError):
            CompiledMod(vllm_config=vllm_config)(*args)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10"
)
def test_save_and_load(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                expected = compiled_mod(*args)

            disable_envs_cache()

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                cached_mod = CompiledMod(vllm_config=vllm_config)
                ret = cached_mod(*args)
            assert cached_mod.was_aot_compile_fn_loaded_from_disk, (
                "Expected was_aot_compile_fn_loaded_from_disk to be True"
            )
            assert torch.allclose(ret, expected)


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10"
)
def test_cache_load_returns_tuple_consistency(monkeypatch: pytest.MonkeyPatch):
    """
    Test that cache loading correctly handles the returns_tuple logic.

    This verifies that when a model returns a single tensor (not a tuple),
    the output type is consistent between fresh compilation and cache load.
    Without the fix, cached artifacts would return [tensor] instead of tensor.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()

            # Fresh compilation
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                fresh_result = compiled_mod(*args)
                fresh_result_type = type(fresh_result)

            # Verify fresh result is a tensor, not a tuple/list
            assert isinstance(fresh_result, torch.Tensor), (
                f"Fresh compile should return tensor, got {fresh_result_type}"
            )

            disable_envs_cache()

            # Load from cache
            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                cached_mod = CompiledMod(vllm_config=vllm_config)
                cached_result = cached_mod(*args)
                cached_result_type = type(cached_result)

            # Verify cache was actually loaded
            assert cached_mod.was_aot_compile_fn_loaded_from_disk, (
                "Expected was_aot_compile_fn_loaded_from_disk to be True after "
                "loading from cache"
            )

            # Verify cached result has same type as fresh result
            assert isinstance(cached_result, torch.Tensor), (
                f"Cache load should return tensor, got {cached_result_type}. "
                "This indicates the returns_tuple logic is not being applied "
                "correctly when loading from cache."
            )

            # Verify values match
            assert torch.allclose(cached_result, fresh_result), (
                "Cached result values should match fresh compilation"
            )


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_cache_load_returns_tuple_consistency_tuple_output(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test that cache loading correctly handles models that return tuples.

    This verifies that when a model returns a tuple of tensors, the output
    type is preserved as a tuple between fresh compilation and cache load.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()

            # Fresh compilation with tuple-returning model
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledModTuple(vllm_config=vllm_config)
                fresh_result = compiled_mod(*args)
                fresh_result_type = type(fresh_result)

            # Verify fresh result is a tuple
            assert isinstance(fresh_result, tuple), (
                f"Fresh compile should return tuple, got {fresh_result_type}"
            )
            assert len(fresh_result) == 2, (
                f"Fresh compile should return 2-tuple, got {len(fresh_result)}"
            )

            disable_envs_cache()

            # Load from cache
            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                cached_mod = CompiledModTuple(vllm_config=vllm_config)
                cached_result = cached_mod(*args)
                cached_result_type = type(cached_result)

            # Verify cache was actually loaded
            assert cached_mod.was_aot_compile_fn_loaded_from_disk, (
                "Expected was_aot_compile_fn_loaded_from_disk to be True after "
                "loading from cache"
            )

            # Verify cached result is also a tuple
            assert isinstance(cached_result, tuple), (
                f"Cache load should return tuple, got {cached_result_type}. "
                "This indicates the returns_tuple logic is not preserving "
                "tuple outputs when loading from cache."
            )
            assert len(cached_result) == 2, (
                f"Cache load should return 2-tuple, got {len(cached_result)}"
            )

            # Verify values match
            assert torch.allclose(cached_result[0], fresh_result[0]), (
                "Cached result[0] values should match fresh compilation"
            )
            assert torch.allclose(cached_result[1], fresh_result[1]), (
                "Cached result[1] values should match fresh compilation"
            )


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
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"

            disable_envs_cache()

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                assert compiled_mod.was_aot_compile_fn_loaded_from_disk, (
                    "Expected was_aot_compile_fn_loaded_from_disk to be True"
                )
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10"
)
def test_partition_wrapper_applied_on_aot_load(
    monkeypatch: pytest.MonkeyPatch, vllm_tmp_cache: Path, mocker
):
    """
    Test that partition wrappers are applied when loading AOT cached functions.

    This test verifies the fix for GitHub issue #31439 where AOT compile
    caused 2x latency regression when use_inductor_graph_partition=True.
    The root cause was that partition wrapper context was bypassed when
    loading from AOT cache.
    """
    from vllm.config import CUDAGraphMode

    args = (torch.randn(10, 10),)
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")

    # Create config with partition enabled
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # First compilation - save to cache
    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)
        compiled_mod(*args)

    disable_envs_cache()

    # Second run - load from cache, verify partition wrapper applied
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # Use mocker to spy on set_customized_partition_wrappers
    spy = mocker.spy(torch._inductor.utils, "set_customized_partition_wrappers")

    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)

        # First call after restart: loads from AOT cache.
        # This tests the fix for the first call after a restart.
        compiled_mod(*args)

        # Verify cache was loaded
        assert compiled_mod.was_aot_compile_fn_loaded_from_disk, (
            "Expected was_aot_compile_fn_loaded_from_disk to be True"
        )

        # Verify partition wrapper was called on AOT load.
        assert spy.call_count >= 2, (
            "Expected partition wrapper to be set and cleared on AOT load, "
            f"got {spy.call_count} calls"
        )
        # First call should set a wrapper, last call should clear it
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on AOT load should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on AOT load should clear the wrapper"
        )

        # Reset for the next check.
        spy.reset_mock()

        # Subsequent call: uses the cached `aot_compiled_fn`.
        # This tests the fix for subsequent calls.
        compiled_mod(*args)

        # Verify partition wrapper was called on the subsequent call.
        assert spy.call_count >= 2, (
            "Expected partition wrapper set and cleared on subsequent "
            f"call, got {spy.call_count} calls"
        )
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on subsequent call should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on subsequent call should clear the wrapper"
        )


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
@create_new_process_for_each_test("spawn")
def test_gpt2_cache_hit(monkeypatch: pytest.MonkeyPatch):
    """
    Test that compiling gpt2 twice results in a cache hit and
    capture torch dynamic symbol creations to ensure make_symbol
    not called on cache hit.
    """

    import torch.fx.experimental.symbolic_shapes as symbolic_shapes_module
    from torch.utils._sympy.symbol import make_symbol

    from vllm import LLM

    create_symbol_counter = multiprocessing.Value("i", 0)
    original_make_symbol = make_symbol

    @functools.wraps(original_make_symbol)
    def counting_make_symbol(prefix, idx, **kwargs):
        with create_symbol_counter.get_lock():
            create_symbol_counter.value += 1
        return original_make_symbol(prefix, idx, **kwargs)

    symbolic_shapes_module.make_symbol = counting_make_symbol
    try:
        with monkeypatch.context() as m, tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            # First compilation - initialize model and generate
            llm_model = LLM(
                model="gpt2",
                compilation_config=CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                ),
                max_model_len=256,
            )

            llm_model.generate("Hello, my name is")
            assert create_symbol_counter.value == 2
            create_symbol_counter.value = 0

            # Clean up first model
            del llm_model
            disable_envs_cache()
            vllm.model_executor.layers.activation._ACTIVATION_REGISTRY._dict.clear()

            # Second compilation - should hit cache
            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            llm_model = LLM(
                model="gpt2",
                compilation_config=CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                ),
                max_model_len=256,
            )
            llm_model.generate("Hello, my name is")

            assert create_symbol_counter.value == 0

    finally:
        # Restore original method
        symbolic_shapes_module.make_symbol = original_make_symbol


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
class TestStandaloneCompiledArtifacts:
    def test_init(self):
        cache = StandaloneCompiledArtifacts()
        assert cache.submodule_bytes == {}
        assert cache.submodule_bytes_store == {}
        assert cache.loaded_submodule_store == {}

    def test_insert_new_artifact(self):
        cache = StandaloneCompiledArtifacts()
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
        cache = StandaloneCompiledArtifacts()

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
        cache = StandaloneCompiledArtifacts()
        test_data = b"retrievable_data"
        submod_name = "mod1"
        shape = "shape16"

        cache.insert(submod_name, shape, test_data)
        retrieved_data = cache.get(submod_name, shape)

        assert retrieved_data == test_data

    def test_get_nonexistent_artifact(self):
        cache = StandaloneCompiledArtifacts()

        with pytest.raises(KeyError):
            cache.get("nonexistent", "shape")

    def test_size_bytes(self):
        cache = StandaloneCompiledArtifacts()

        assert cache.size_bytes() == 0

        data1 = b"x" * 100
        data2 = b"y" * 200
        cache.insert("mod1", "shape1", data1)
        cache.insert("mod2", "shape2", data2)

        assert cache.size_bytes() == 300

    def test_num_artifacts_and_entries(self):
        cache = StandaloneCompiledArtifacts()

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
        cache = StandaloneCompiledArtifacts()

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
        cache = StandaloneCompiledArtifacts()

        mock_artifact = Mock()
        cache.submodule_bytes_store["hash1"] = pickle.dumps(b"data1")
        cache.loaded_submodule_store["hash1"] = mock_artifact

        cache.load_all()

        mock_deserialize.assert_not_called()

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_get_loaded_artifact(self, mock_deserialize):
        """Test retrieving loaded artifacts"""
        cache = StandaloneCompiledArtifacts()

        mock_artifact = Mock()
        mock_deserialize.return_value = mock_artifact

        submod_name = "test_mod"
        shape = "test_shape"
        cache.insert(submod_name, shape, pickle.dumps(b"test_data"))
        cache.load_all()

        retrieved_artifact = cache.get_loaded(submod_name, shape)
        assert retrieved_artifact == mock_artifact

    def test_getstate_setstate(self):
        cache = StandaloneCompiledArtifacts()

        cache.insert("mod1", "shape1", b"data1")
        cache.insert("mod2", "shape2", b"data2")

        cache.loaded_submodule_store["hash1"] = Mock()

        state = cache.__getstate__()

        assert "submodule_bytes" in state
        assert "submodule_bytes_store" in state
        assert "loaded_submodule_store" not in state

        new_cache = StandaloneCompiledArtifacts()
        new_cache.__setstate__(state)

        assert new_cache.submodule_bytes == cache.submodule_bytes
        assert new_cache.submodule_bytes_store == cache.submodule_bytes_store
        assert new_cache.loaded_submodule_store == {}

    def test_pickle_roundtrip(self):
        cache = StandaloneCompiledArtifacts()

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


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
class TestStandaloneCompiledArtifactsIntegration:
    def test_add_pickle_unpickle(self):
        cache = StandaloneCompiledArtifacts()

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
        cache = StandaloneCompiledArtifacts()

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
