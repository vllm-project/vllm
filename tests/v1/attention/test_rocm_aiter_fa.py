# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("kv_cache_dtype", "uses_descales"),
    [
        ("auto", False),
        ("float16", False),
        ("bfloat16", False),
        ("fp8", True),
    ],
)
def test_aiter_fa_uses_descales_only_for_quantized_kv_cache(
    kv_cache_dtype: str, uses_descales: bool
) -> None:
    from vllm.v1.attention.backends.rocm_aiter_fa import _get_kv_cache_descales

    k_scale = torch.tensor(2.0)
    v_scale = torch.tensor(3.0)
    k_descale, v_descale = _get_kv_cache_descales(
        kv_cache_dtype, k_scale, v_scale, (2, 4)
    )

    if not uses_descales:
        assert k_descale is None
        assert v_descale is None
        return

    torch.testing.assert_close(k_descale, torch.full((2, 4), 2.0))
    torch.testing.assert_close(v_descale, torch.full((2, 4), 3.0))
