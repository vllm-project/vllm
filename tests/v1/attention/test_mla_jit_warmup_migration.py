# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate MLA JIT dispatch against pre-contract behavior."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.v1.attention.backends.mla.sparse_swa import ComputePrefillMetadataKernel


@pytest.mark.parametrize(
    ("num_prefills", "expected_block_size"),
    [(1, 1), (3, 4), (8, 8), (9, 16)],
)
def test_compute_prefill_metadata_dispatch_matches_legacy_meta(
    num_prefills: int,
    expected_block_size: int,
) -> None:
    kernel = ComputePrefillMetadataKernel()

    assert kernel.dispatch(num_prefills=num_prefills) == kernel.CompileKey(
        block_size=expected_block_size
    )
