# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU sparse MLA backend",
)


def test_xpu_sparse_impl_implements_logical_topk_hook():
    """``MLAAttention.forward`` calls ``impl.record_logical_topk_ready()``
    unconditionally for sparse layers, so every sparse MLA impl must define it.
    Without it DeepSeek-V3.2 / GLM-5 crash on the first forward on XPU."""
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseImpl

    hook = getattr(XPUMLASparseImpl, "record_logical_topk_ready", None)
    assert callable(hook)
    # The XPU impl does not use sparse-MLA index groups: the hook is a no-op.
    impl = object.__new__(XPUMLASparseImpl)
    impl.record_logical_topk_ready()
