# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native segmented FP8 conversion eligibility."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import segmented_attention as segmented


@pytest.mark.parametrize("gfx12", (False, True))
@pytest.mark.parametrize(
    "dtype,kv_dtype,eligible",
    [
        (torch.bfloat16, torch.float8_e4m3fn, True),
        (torch.bfloat16, torch.float8_e4m3fnuz, False),
        (torch.float16, torch.float8_e4m3fn, False),
        (torch.bfloat16, torch.bfloat16, False),
    ],
)
def test_native_fp8_requires_supported_query_cache_and_architecture(
    monkeypatch, gfx12, dtype, kv_dtype, eligible
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(on_gfx12x=lambda: gfx12),
    )
    assert segmented.supports_segmented_native_fp8(dtype, kv_dtype) == (
        gfx12 and eligible
    )
