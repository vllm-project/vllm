# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that attention backends produce KV cache physical layouts consistent
with what get_kv_cache_layout() requests.

The bug: TritonAttentionBackend.get_kv_cache_stride_order() ignores
get_kv_cache_layout() and always returns identity (NHD) ordering, even
when HND is requested. This breaks heterogeneous TP head splitting in
P/D disaggregation because the NIXL connector assumes heads are contiguous
in physical memory (HND layout).

Run:
    pytest tests/v1/kv_connector/unit/test_hetero_tp_layout_bug.py -v
"""

import importlib
import logging

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    set_kv_cache_layout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamically collect all attention backends that have the standard 5-dim
# KV cache shape: [*, *, block_size, num_kv_heads, head_dim]
# ---------------------------------------------------------------------------
_STANDARD_BACKENDS = []

# Set a layout override so get_kv_cache_layout() doesn't need a vllm config.
set_kv_cache_layout("NHD")
get_kv_cache_layout.cache_clear()

for entry in AttentionBackendEnum:
    path = entry.value
    if not path:
        continue
    module_path, class_name = path.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except Exception as e:
        logger.info("Skipping %s: %s", entry.name, e)
        continue

    # Only test backends with the standard 5-dim shape (has a `2` dim for K/V).
    try:
        shape = cls.get_kv_cache_shape(
            num_blocks=4, block_size=16, num_kv_heads=8, head_size=64
        )
    except Exception:
        continue
    if len(shape) != 5 or 2 not in shape:
        continue

    # Only test backends that implement get_kv_cache_stride_order.
    try:
        cls.get_kv_cache_stride_order()
    except NotImplementedError:
        continue

    _STANDARD_BACKENDS.append(
        pytest.param(cls, id=entry.name),
    )

if not _STANDARD_BACKENDS:
    pytest.skip("No standard attention backends available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_layout_cache():
    """Reset the lru_cache on get_kv_cache_layout between tests."""
    get_kv_cache_layout.cache_clear()
    yield
    set_kv_cache_layout("NHD")
    get_kv_cache_layout.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_physical_dim_names(backend_cls, layout):
    """Return the physical dimension order as human-readable names."""
    set_kv_cache_layout(layout)
    get_kv_cache_layout.cache_clear()
    order = backend_cls.get_kv_cache_stride_order()

    # Map logical dim indices to names based on the backend's shape.
    shape = backend_cls.get_kv_cache_shape(
        num_blocks=4, block_size=16, num_kv_heads=8, head_size=64
    )

    # Identify which dim index is which by value.
    # shape has exactly one dim with value 2 (K/V split).
    dim_names = []
    for i, size in enumerate(shape):
        if size == 4:
            dim_names.append("num_blocks")
        elif size == 2:
            dim_names.append("kv")
        elif size == 16:
            dim_names.append("block_size")
        elif size == 8:
            dim_names.append("num_kv_heads")
        elif size == 64:
            dim_names.append("head_dim")
        else:
            dim_names.append(f"dim{i}({size})")

    return [dim_names[i] for i in order]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrideOrderRespectsLayout:
    """
    Core invariant: if get_kv_cache_layout() says "HND", then
    get_kv_cache_stride_order() must place num_kv_heads before
    block_size in the physical layout. And vice versa for "NHD".
    """

    @pytest.mark.parametrize("backend_cls", _STANDARD_BACKENDS)
    def test_hnd_puts_heads_before_tokens(self, backend_cls):
        physical = _get_physical_dim_names(backend_cls, "HND")
        h = physical.index("num_kv_heads")
        n = physical.index("block_size")
        assert h < n, (
            f"{backend_cls.__name__} with layout=HND: physical order is "
            f"{physical}, but num_kv_heads (pos {h}) should come before "
            f"block_size (pos {n})"
        )

    @pytest.mark.parametrize("backend_cls", _STANDARD_BACKENDS)
    def test_nhd_puts_tokens_before_heads(self, backend_cls):
        physical = _get_physical_dim_names(backend_cls, "NHD")
        h = physical.index("num_kv_heads")
        n = physical.index("block_size")
        assert n < h, (
            f"{backend_cls.__name__} with layout=NHD: physical order is "
            f"{physical}, but block_size (pos {n}) should come before "
            f"num_kv_heads (pos {h})"
        )

    @pytest.mark.parametrize("backend_cls", _STANDARD_BACKENDS)
    @pytest.mark.parametrize("layout", ["NHD", "HND"])
    def test_stride_order_is_valid_permutation(self, backend_cls, layout):
        set_kv_cache_layout(layout)
        get_kv_cache_layout.cache_clear()
        order = backend_cls.get_kv_cache_stride_order()
        assert sorted(order) == list(range(5)), (
            f"{backend_cls.__name__} stride order {order} is not a valid "
            f"permutation of [0,1,2,3,4]"
        )
