# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm Triton MLA tests", allow_module_level=True)

from vllm.v1.attention.backend import AttentionCGSupport  # noqa: E402
from vllm.v1.attention.backends.mla.triton_mla import (  # noqa: E402
    TritonMLABackend,
    TritonMLAMetadataBuilder,
)


def test_noncausal_multitoken_dcp_capability():
    assert TritonMLAMetadataBuilder.supports_non_causal_multi_token_decode
    assert TritonMLAMetadataBuilder.supports_non_causal_multi_token_dcp
    assert TritonMLABackend.supports_non_causal()
    assert TritonMLABackend.supports_non_causal_dcp()


def test_cudagraph_support_tracks_multitoken_decode_group():
    single_token_spec = SimpleNamespace(non_causal_multi_token_decode=False)
    multi_token_spec = SimpleNamespace(non_causal_multi_token_decode=True)

    assert (
        TritonMLAMetadataBuilder.get_cudagraph_support(None, single_token_spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    assert (
        TritonMLAMetadataBuilder.get_cudagraph_support(None, multi_token_spec)
        == AttentionCGSupport.UNIFORM_BATCH
    )
