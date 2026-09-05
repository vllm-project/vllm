# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import pytest


@contextmanager
def _without_transformers_module(
    monkeypatch: pytest.MonkeyPatch,
    import_name: str,
    missing_module: str | None = None,
) -> Iterator[None]:
    real_import = builtins.__import__
    missing_module = missing_module or import_name

    def import_without_module(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == import_name or name.startswith(f"{import_name}."):
            raise ModuleNotFoundError(
                f"No module named '{missing_module}'", name=missing_module
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_module)
    yield


def _clear_modules(module_name: str, missing_transformers_module: str):
    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{missing_transformers_module}."):
            sys.modules.pop(name, None)
    sys.modules.pop(missing_transformers_module, None)


@pytest.mark.parametrize(
    ("module_name", "missing_transformers_module"),
    [
        (
            "vllm.model_executor.models.exaone4_5",
            "transformers.models.exaone4_5",
        ),
        (
            "vllm.model_executor.models.gemma4_unified",
            "transformers.models.gemma4_unified",
        ),
    ],
)
def test_model_module_imports_when_optional_transformers_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    missing_transformers_module: str,
):
    """Missing optional HF model modules should not break registry import."""
    _clear_modules(module_name, missing_transformers_module)

    with _without_transformers_module(monkeypatch, missing_transformers_module):
        module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    ("module_name", "missing_transformers_module", "internal_missing_module"),
    [
        (
            "vllm.model_executor.models.exaone4_5",
            "transformers.models.exaone4_5",
            "missing_internal_exaone4_5_dependency",
        ),
        (
            "vllm.model_executor.models.gemma4_unified",
            "transformers.models.gemma4_unified",
            "missing_internal_gemma4_unified_dependency",
        ),
    ],
)
def test_model_module_reraises_internal_transformers_import_errors(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    missing_transformers_module: str,
    internal_missing_module: str,
):
    """Only the optional HF module itself should be treated as absent."""
    _clear_modules(module_name, missing_transformers_module)

    with (
        _without_transformers_module(
            monkeypatch,
            missing_transformers_module,
            missing_module=internal_missing_module,
        ),
        pytest.raises(ModuleNotFoundError) as exc_info,
    ):
        importlib.import_module(module_name)

    assert exc_info.value.name == internal_missing_module
