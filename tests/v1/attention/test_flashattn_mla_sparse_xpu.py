# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


def test_flashattn_mla_sparse_uses_xpu_varlen_func():
    from vllm._xpu_ops import xpu_ops
    from vllm.v1.attention.backends.mla import flashattn_mla_sparse

    assert flashattn_mla_sparse.flash_attn_varlen_func is (
        xpu_ops.flash_attn_varlen_func
    )
