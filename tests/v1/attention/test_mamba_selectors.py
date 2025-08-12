# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mamba attention backend selectors."""

import pytest

from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.mamba_selectors import get_mamba_attn_backend


@pytest.mark.parametrize(argnames=["mamba_type", "expected_backend"],
                         argvalues=[("mamba2", Mamba2AttentionBackend)])
def test_get_mamba_attn_backend_mamba2(mamba_type, expected_backend):
    backend_class = get_mamba_attn_backend(mamba_type)

    assert backend_class is expected_backend


def test_get_mamba_attn_backend_unsupported():
    unsupported_types = ["mamba", ""]

    for mamba_type in unsupported_types:
        err_message = f"Mamba Attention type {mamba_type} is not supported yet."
        with pytest.raises(NotImplementedError, match=err_message):
            get_mamba_attn_backend(mamba_type)
