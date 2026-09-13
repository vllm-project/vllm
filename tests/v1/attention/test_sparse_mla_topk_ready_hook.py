# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA implementations must provide record_logical_topk_ready.

``MultiHeadLatentAttentionWrapper.forward_native`` calls
``impl.record_logical_topk_ready()`` on every sparse layer. Backends that do
not inherit ``SparseMLACommonImpl`` (ROCm AITER, XPU) crashed with
AttributeError at model load before the default hook was added to
``MLAAttentionImpl``.
"""

import pytest

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.v1.attention.backend import MLAAttentionImpl

pytestmark = [
    pytest.mark.skip_global_cleanup,
]

# Direct-import backends: these inherit MLAAttentionImpl (not
# SparseMLACommonImpl) and previously missed the hook.
try:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    _HAS_ROCM_BACKEND = True
except Exception:  # pragma: no cover - ROCm deps not installed
    ROCMAiterMLASparseImpl = None
    _HAS_ROCM_BACKEND = False

try:
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        XPUMLASparseImpl,
    )

    _HAS_XPU_BACKEND = True
except Exception:  # pragma: no cover - XPU deps not installed
    XPUMLASparseImpl = None
    _HAS_XPU_BACKEND = False


def test_mla_attention_impl_base_provides_default_hook():
    """The base class must provide a callable no-op default hook.

    Backends without a ``SparseMLAIndexGroup`` have nothing to record, so the
    inherited default must be safe to call on a bare subclass.
    """
    assert callable(MLAAttentionImpl.record_logical_topk_ready)

    class BareSparseImpl(MLAAttentionImpl):
        is_sparse = True

        def __init__(self) -> None:
            pass

        def forward_mqa(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    # Exercises the inherited base-class default, not an override.
    BareSparseImpl().record_logical_topk_ready()


def test_sparse_common_impl_still_overrides_hook():
    """SparseMLACommonImpl keeps its index-group-aware override."""
    assert SparseMLACommonImpl.record_logical_topk_ready is not (
        MLAAttentionImpl.record_logical_topk_ready
    )


@pytest.mark.parametrize(
    "impl_cls",
    [
        pytest.param(
            ROCMAiterMLASparseImpl,
            marks=pytest.mark.skipif(
                not _HAS_ROCM_BACKEND, reason="ROCm backend not importable"
            ),
            id="ROCMAiterMLASparseImpl",
        ),
        pytest.param(
            XPUMLASparseImpl,
            marks=pytest.mark.skipif(
                not _HAS_XPU_BACKEND, reason="XPU backend not importable"
            ),
            id="XPUMLASparseImpl",
        ),
    ],
)
def test_direct_sparse_impls_satisfy_hook(impl_cls):
    """Impls without SparseMLACommonImpl must resolve the hook.

    Regression test for the ROCm GLM-5.3-Flash load-time AttributeError:
    these backends never own an index group, so they must pick up the
    base-class no-op.
    """
    assert issubclass(impl_cls, MLAAttentionImpl)
    assert not issubclass(impl_cls, SparseMLACommonImpl)
    assert (
        impl_cls.record_logical_topk_ready is MLAAttentionImpl.record_logical_topk_ready
    )
