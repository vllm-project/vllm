# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.import_utils import PlaceholderModule, _has_module, import_plugin


def _raises_module_not_found():
    return pytest.raises(ModuleNotFoundError, match="No module named")


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    with _raises_module_not_found():
        int(placeholder)

    with _raises_module_not_found():
        placeholder()

    with _raises_module_not_found():
        _ = placeholder.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __name attribute
        _ = placeholder.name

    # OK to print the placeholder or use it in a f-string
    _ = repr(placeholder)
    _ = str(placeholder)

    # No error yet; only error when it is used downstream
    placeholder_attr = placeholder.placeholder_attr("attr")

    with _raises_module_not_found():
        int(placeholder_attr)

    with _raises_module_not_found():
        placeholder_attr()

    with _raises_module_not_found():
        _ = placeholder_attr.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __module attribute
        _ = placeholder_attr.module


class TestHasModule:
    """Tests for _has_module with trial import verification."""

    def setup_method(self):
        # Clear the @cache between tests so each test gets a fresh call
        _has_module.cache_clear()

    def test_returns_true_for_importable_stdlib_module(self):
        assert _has_module("json") is True

    def test_returns_false_for_nonexistent_module(self):
        assert _has_module("nonexistent_module_xyz_12345") is False

    def test_returns_false_when_find_spec_succeeds_but_import_fails(self):
        """Simulate a native extension whose shared library is missing.

        ``find_spec`` finds the package on disk, but the actual import
        raises ``ImportError`` (e.g. missing ``libcudart.so``).
        """
        fake_spec = MagicMock()

        with (
            patch(
                "vllm.utils.import_utils.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(
                    "libcudart.so.12: cannot open shared object file"
                ),
            ),
        ):
            assert _has_module("fake_native_ext") is False

    def test_returns_false_when_find_spec_raises(self):
        """``find_spec`` itself can raise for dotted names whose parent package
        fails to import. This should be treated as the module being unavailable.
        """
        with patch(
            "vllm.utils.import_utils.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'fake_parent'"),
        ):
            assert _has_module("fake_parent.child") is False

    def test_result_is_cached(self):
        """Verify the @cache decorator prevents repeated imports."""
        _has_module("json")  # prime the cache

        with patch("vllm.utils.import_utils.importlib.util.find_spec") as mock_spec:
            result = _has_module("json")  # should hit cache
            mock_spec.assert_not_called()
            assert result is True


class TestImportPlugin:
    def test_importing_from_site_packages(self):
        import json

        result = import_plugin("json")
        assert result is json

    def test_importing_from_file(self, tmp_path):
        plugin_file = tmp_path / "my_test_plugin.py"
        plugin_file.write_text("VALUE = 42\n")

        try:
            result = import_plugin(str(plugin_file))
            assert result is not None
            assert result.VALUE == 42
        finally:
            sys.modules.pop("my_test_plugin", None)

    def test_returns_none_when_both_attempts_fail(self):
        with (
            patch(
                "vllm.utils.import_utils.import_from_path",
                side_effect=FileNotFoundError("no such file"),
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ModuleNotFoundError("no such module"),
            ),
        ):
            result = import_plugin("nonexistent_plugin_xyz")
            assert result is None


@pytest.fixture
def sparse_mla_autotune(monkeypatch):
    from vllm.utils import flashinfer as fi

    state = SimpleNamespace(is_tuning_mode=False, calls=[])

    @contextlib.contextmanager
    def autotune(*, tune_mode, skip_ops):
        state.calls.append((tune_mode, skip_ops))
        previous = state.is_tuning_mode
        state.is_tuning_mode = previous or tune_mode
        try:
            yield
        finally:
            state.is_tuning_mode = previous

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(AutoTuner=SimpleNamespace(get=lambda: state)),
    )
    monkeypatch.setattr(fi, "autotune", autotune)
    monkeypatch.setattr(fi, "has_flashinfer", lambda: True)
    implementations = SimpleNamespace()
    monkeypatch.setattr(fi, "_get_submodule", lambda _: implementations)
    wrappers = [
        fi.flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        fi.flashinfer_trtllm_batch_decode_sparse_mla_dsv4,
        fi.flashinfer_trtllm_bf16_moe,
    ]
    caches = [
        inspect.getclosurevars(inspect.unwrap(fn)).nonlocals["_get_impl"]
        for fn in wrappers
    ]
    for cache in caches:
        cache.cache_clear()
    yield fi, state, implementations
    for cache in caches:
        cache.cache_clear()


@pytest.mark.parametrize(
    "op_name",
    ["trtllm_batch_decode_with_kv_cache_mla", "trtllm_batch_decode_sparse_mla_dsv4"],
)
@pytest.mark.parametrize("fails", [False, True])
def test_sparse_mla_lazy_wrappers_tune_only_attention(
    sparse_mla_autotune, monkeypatch, op_name, fails
):
    fi, state, implementations = sparse_mla_autotune
    query, output = object(), object()

    def decode(arg, *, out):
        assert arg is query and out is output
        if fails and state.is_tuning_mode:
            raise ValueError("decode failed")
        return state.is_tuning_mode

    setattr(implementations, op_name, decode)
    implementations.trtllm_bf16_moe = lambda: state.is_tuning_mode
    wrapper = getattr(fi, "flashinfer_" + op_name)
    skip_ops = {"some_attention_op"}
    error = (
        pytest.raises(ValueError, match="decode failed")
        if fails
        else contextlib.nullcontext()
    )
    with error, fi.autotune_sparse_mla_only(skip_ops=skip_ops):
        assert not state.is_tuning_mode
        assert not fi.flashinfer_trtllm_bf16_moe()
        assert wrapper(query, out=output)
        assert not fi.flashinfer_trtllm_bf16_moe()
    assert state.calls == [(True, skip_ops)]
    assert not state.is_tuning_mode

    def unexpected_autotune(*args, **kwargs):
        pytest.fail("Inactive sparse MLA must not access the autotuner")

    monkeypatch.setattr(fi, "autotune", unexpected_autotune)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", None)
    assert not wrapper(query, out=output)


def test_sparse_mla_scope_restores_nested_marker_after_exception(sparse_mla_autotune):
    fi, state, implementations = sparse_mla_autotune
    implementations.trtllm_batch_decode_sparse_mla_dsv4 = lambda: None
    decode = fi.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    with fi.autotune_sparse_mla_only(skip_ops={"outer"}):
        with (
            pytest.raises(ValueError, match="model failed"),
            fi.autotune_sparse_mla_only(),
        ):
            decode()
            raise ValueError("model failed")
        decode()
    decode()
    assert state.calls == [(True, set()), (True, {"outer"})]
    assert not state.is_tuning_mode


def test_sparse_mla_scope_rejects_existing_full_model_tuning(sparse_mla_autotune):
    fi, state, implementations = sparse_mla_autotune
    state.is_tuning_mode = True
    with (
        pytest.raises(RuntimeError, match="active FlashInfer autotuning"),
        fi.autotune_sparse_mla_only(),
    ):
        pytest.fail("Must reject the outer tuning context before running the model")
    assert state.is_tuning_mode
    assert state.calls == []
    state.is_tuning_mode = False
    implementations.trtllm_batch_decode_sparse_mla_dsv4 = lambda: None
    fi.flashinfer_trtllm_batch_decode_sparse_mla_dsv4()
    assert state.calls == []
