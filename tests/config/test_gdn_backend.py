# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_GDN_BACKEND`` selects the whole-layer GDN implementation on XPU."""

from unittest.mock import patch

import pytest

from vllm import envs
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _resolve_gdn_backend,
)


@pytest.mark.parametrize(
    "env,is_xpu,expected",
    [
        # XPU has both implementations.
        ("auto", True, "sycl"),
        ("triton", True, "triton"),
        ("sycl", True, "sycl"),
        ("SYCL", True, "sycl"),
        # Elsewhere only the Triton chain exists.
        ("auto", False, "triton"),
        ("triton", False, "triton"),
    ],
)
def test_resolve_gdn_backend(env, is_xpu, expected):
    with (
        patch.object(envs, "VLLM_GDN_BACKEND", env),
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_xpu",
            return_value=is_xpu,
        ),
    ):
        assert _resolve_gdn_backend() == expected


def test_sycl_rejected_off_xpu():
    with (
        patch.object(envs, "VLLM_GDN_BACKEND", "sycl"),
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_xpu",
            return_value=False,
        ),
        pytest.raises(ValueError, match="requires XPU"),
    ):
        _resolve_gdn_backend()
