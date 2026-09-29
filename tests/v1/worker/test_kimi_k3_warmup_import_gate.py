# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi warmup must avoid model imports and preserve dispatch for loaded layers."""

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup import kimi_k3_triton_warmup as warmup

pytestmark = pytest.mark.skip_global_cleanup

_KDA_MODULE = "vllm.models.kimi_k3.nvidia.kda"


class FakeKda:
    pass


@pytest.fixture
def warmup_calls(monkeypatch):
    original_import = builtins.__import__

    def reject_kimi_import(name, *args, **kwargs):
        if name.startswith("vllm.models.kimi_k3"):
            pytest.fail(f"Warmup must not import Kimi model code: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_kimi_import)
    monkeypatch.setattr(warmup.current_platform, "is_cuda", lambda: True)
    attn_res, recurrent_kda = Mock(), Mock()
    monkeypatch.setattr(warmup, "_warm_attn_res", attn_res)
    monkeypatch.setattr(warmup, "_warm_recurrent_kda", recurrent_kda)
    return attn_res, recurrent_kda


def _worker(static_context):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=static_context)
        ),
        model_config=SimpleNamespace(dtype=object()),
    )


def test_non_kimi_warmup_does_not_import_model_code(monkeypatch, warmup_calls):
    """Non-Kimi startup must not trigger the import-time numba cache requirement."""
    monkeypatch.delitem(sys.modules, _KDA_MODULE, raising=False)

    warmup.kimi_k3_triton_warmup(_worker({}))

    for callback in warmup_calls:
        callback.assert_not_called()


@pytest.mark.parametrize("static_context", [None, {"other": object()}])
def test_loaded_module_without_kda_layer_skips_warmup(
    monkeypatch, warmup_calls, static_context
):
    """A previously imported Kimi module alone must not trigger kernel warmup."""
    monkeypatch.setitem(
        sys.modules, _KDA_MODULE, SimpleNamespace(KimiK3DeltaAttention=FakeKda)
    )

    warmup.kimi_k3_triton_warmup(_worker(static_context))

    for callback in warmup_calls:
        callback.assert_not_called()


def test_loaded_kda_layer_preserves_warmup_dispatch(monkeypatch, warmup_calls):
    """Kimi models still warm both kernel families using the matching layer."""
    layer = FakeKda()
    monkeypatch.setitem(
        sys.modules, _KDA_MODULE, SimpleNamespace(KimiK3DeltaAttention=FakeKda)
    )
    worker = _worker({"other": object(), "kda": layer})

    warmup.kimi_k3_triton_warmup(worker)

    attn_res, recurrent_kda = warmup_calls
    attn_res.assert_called_once_with(worker)
    recurrent_kda.assert_called_once_with(layer, worker.model_config.dtype)
