# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi-K3 warmup must not import Kimi model code for non-Kimi models.

kernel_warmup() runs kimi_k3_triton_warmup() for every model.
_get_kda_layer() must gate on sys.modules instead of importing the Kimi
package, whose import chain ends in a module-level numba @njit(cache=True)
that requires a writable cache directory (#59250).
"""

import sys
from types import SimpleNamespace

from vllm.model_executor.warmup.kimi_k3_triton_warmup import _get_kda_layer

_KDA_MODULE = "vllm.models.kimi_k3.nvidia.kda"


def test_get_kda_layer_returns_none_without_importing_kimi():
    """A non-Kimi worker must not trigger any Kimi import."""
    saved = sys.modules.pop(_KDA_MODULE, None)
    try:
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                compilation_config=SimpleNamespace(static_forward_context={})
            )
        )
        modules_before = set(sys.modules)
        assert _get_kda_layer(worker) is None
        assert not set(sys.modules) - modules_before
    finally:
        if saved is not None:
            sys.modules[_KDA_MODULE] = saved


def test_get_kda_layer_returns_none_without_static_context(monkeypatch):
    """Even with kda loaded (Kimi-K3 deployment), a non-dict static context
    still yields None."""

    class FakeKda:
        pass

    monkeypatch.setitem(
        sys.modules, _KDA_MODULE, SimpleNamespace(KimiK3DeltaAttention=FakeKda)
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(compilation_config=SimpleNamespace())
    )
    assert _get_kda_layer(worker) is None


def test_get_kda_layer_finds_kda_layer_when_module_loaded(monkeypatch):
    """With kda loaded (Kimi-K3 deployment), the KDA layer is still found."""

    class FakeKda:
        pass

    fake_layer = FakeKda()
    monkeypatch.setitem(
        sys.modules, _KDA_MODULE, SimpleNamespace(KimiK3DeltaAttention=FakeKda)
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={"kda": fake_layer, "other": object()}
            )
        )
    )
    assert _get_kda_layer(worker) is fake_layer
