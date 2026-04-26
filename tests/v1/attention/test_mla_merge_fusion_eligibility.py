# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``MLACommonImpl.mha_merge_state_fusion_supported``.

Covers the eligibility check that gates the chunked-prefill merge-state +
static FP8 quant fusion plumbing in ``MLAAttention.forward_impl`` /
``MLACommonImpl.forward_mha``.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)


def _md(*, has_prefill: bool = True, has_context: bool = True):
    """Build a stub attn_metadata with the only fields the helper reads."""
    if not has_prefill:
        return SimpleNamespace(prefill=None)
    chunked_context = object() if has_context else None
    return SimpleNamespace(prefill=SimpleNamespace(chunked_context=chunked_context))


def _impl(dcp_world_size: int = 1):
    """Stub ``self`` exposing only what the helper touches."""
    return SimpleNamespace(dcp_world_size=dcp_world_size)


# Bind the unbound method once for direct invocation against a stub ``self``.
_supported = MLACommonImpl.mha_merge_state_fusion_supported


def test_static_fp8_with_chunked_context_is_supported():
    assert _supported(_impl(), kFp8StaticTensorSym, _md())


@pytest.mark.parametrize(
    "quant_key", [kFp8Dynamic128Sym, kFp8Dynamic64Sym, kNvfp4Dynamic, None]
)
def test_non_static_fp8_quant_keys_are_not_supported(quant_key):
    assert not _supported(_impl(), quant_key, _md())


def test_no_prefill_metadata_is_not_supported():
    assert not _supported(_impl(), kFp8StaticTensorSym, _md(has_prefill=False))


def test_no_chunked_context_is_not_supported():
    assert not _supported(_impl(), kFp8StaticTensorSym, _md(has_context=False))


def test_dcp_world_size_above_one_is_not_supported():
    assert not _supported(_impl(dcp_world_size=2), kFp8StaticTensorSym, _md())


def test_attn_metadata_none_is_not_supported():
    assert not _supported(_impl(), kFp8StaticTensorSym, None)
