# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for FP8 KV cache wake-up with nested cache containers.

Covers ``vllm.v1.worker.gpu_model_runner._iter_kv_cache_tensors`` and the
``GPUModelRunner.init_fp8_kv_scales`` zeroing loop. Before the fix
backported here (upstream PR #41896), the zero-out loop iterated
``self.kv_caches`` directly and called ``.zero_()`` on each entry — which
fails with ``AttributeError: 'list' object has no attribute 'zero_'`` when
an entry is itself a container (list/tuple/Mapping of tensors), as
produced by ``--pipeline-parallel-size > 1`` with
``--kv-cache-dtype fp8``.

The patch introduces ``_iter_kv_cache_tensors`` to recursively descend
into nested containers and yield leaf tensors. The tests in this module
exercise both the helper directly and the integration path through
``init_fp8_kv_scales``.

No GPU is required.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, _iter_kv_cache_tensors


def _t(shape=(2,), value=1.0):
    """Build a small CPU tensor for use as a leaf KV cache entry."""
    return torch.full(shape, value, dtype=torch.float32)


# ----------------------------------------------------------------------
# _iter_kv_cache_tensors — direct unit tests of the recursive helper.
# ----------------------------------------------------------------------


class TestIterKVCacheTensors:
    def test_flat_list_yields_tensors_in_order(self):
        t0, t1 = _t(), _t()
        assert list(_iter_kv_cache_tensors([t0, t1])) == [t0, t1]

    def test_nested_list_descends_to_leaves(self):
        # The PP>1 + fp8 KV shape — list-of-list. This is the exact shape
        # that triggered AttributeError before the fix.
        t0, t1, t2 = _t(), _t(), _t()
        nested = [[t0, t1], [t2]]
        assert list(_iter_kv_cache_tensors(nested)) == [t0, t1, t2]

    def test_mapping_descends_into_values(self):
        t0, t1 = _t(), _t()
        result = list(_iter_kv_cache_tensors([{"k": t0, "v": t1}]))
        # dict iteration order is insertion order in CPython 3.7+, but
        # don't depend on it — compare as a set of identities.
        assert {id(x) for x in result} == {id(t0), id(t1)}

    def test_mixed_containers_resolve_correctly(self):
        t0, t1, t2 = _t(), _t(), _t()
        mixed = [t0, [t1], (t2,)]
        assert list(_iter_kv_cache_tensors(mixed)) == [t0, t1, t2]

    def test_skips_none_entries(self):
        t0 = _t()
        assert list(_iter_kv_cache_tensors([None, t0, None])) == [t0]

    def test_raises_typeerror_on_unexpected_leaf(self):
        with pytest.raises(TypeError, match="Expected KV cache entries"):
            list(_iter_kv_cache_tensors([42]))


# ----------------------------------------------------------------------
# init_fp8_kv_scales — integration tests of the patched method.
# ----------------------------------------------------------------------


def _make_runner(monkeypatch, *, kv_caches, cache_dtype="fp8", quantized=True):
    """Build a bare GPUModelRunner with just enough state for the method.

    Skips ``__init__`` entirely — we only set the four attributes
    ``init_fp8_kv_scales`` reads (cache_config, kv_caches,
    compilation_config) and stub ``is_quantized_kv_cache`` so the dtype
    string doesn't have to match any specific enum value.
    """
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    runner.kv_caches = kv_caches
    # The scale-set loop walks this; empty dict is enough — we're testing
    # the zeroing loop, not the scale-set loop.
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    monkeypatch.setattr(
        gpu_model_runner_module,
        "is_quantized_kv_cache",
        lambda _dt: quantized,
    )
    return runner


def test_init_fp8_kv_scales_nested_kv_caches_no_attribute_error(monkeypatch):
    """The PP>1 + fp8 wake-up shape (nested list-of-tensors) must not raise.

    Before the fix, the zero-out loop called ``.zero_()`` on each
    top-level entry and ``AttributeError``'d on the inner list. After the
    fix, the loop recurses via ``_iter_kv_cache_tensors`` and zeroes
    every leaf.
    """
    t0 = _t(value=7.0)
    t1 = _t(value=9.0)
    t2 = _t(value=11.0)
    runner = _make_runner(monkeypatch, kv_caches=[[t0, t1], [t2]])

    # Pre-fix code raises AttributeError on the line below. The PR makes
    # this call return cleanly with every leaf zeroed.
    runner.init_fp8_kv_scales()

    assert torch.equal(t0, torch.zeros_like(t0))
    assert torch.equal(t1, torch.zeros_like(t1))
    assert torch.equal(t2, torch.zeros_like(t2))


def test_init_fp8_kv_scales_flat_kv_caches_still_zeroes(monkeypatch):
    """Pre-fix flat-list path (single PP rank) must keep working."""
    t0 = _t(value=5.0)
    t1 = _t(value=6.0)
    runner = _make_runner(monkeypatch, kv_caches=[t0, t1])

    runner.init_fp8_kv_scales()

    assert torch.equal(t0, torch.zeros_like(t0))
    assert torch.equal(t1, torch.zeros_like(t1))


def test_init_fp8_kv_scales_short_circuits_for_non_quantized(monkeypatch):
    """When the KV cache dtype is not quantized, the method must return
    early and leave cache tensors untouched."""
    t0 = _t(value=3.0)
    runner = _make_runner(
        monkeypatch,
        kv_caches=[t0],
        cache_dtype="auto",
        quantized=False,
    )

    runner.init_fp8_kv_scales()

    assert torch.equal(t0, _t(value=3.0))
