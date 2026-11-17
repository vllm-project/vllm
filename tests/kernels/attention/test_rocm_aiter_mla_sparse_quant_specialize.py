# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm sparse indexer quant kernel.",
        allow_module_level=True,
    )


def test_indexer_k_quant_does_not_specialize_num_tokens():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _indexer_k_quant_and_cache_kernel,
    )

    assert "num_tokens" in _indexer_k_quant_and_cache_kernel.do_not_specialize
