import re

import pytest
import torch

from vllm.vllm_flash_attn import flash_attn_interface as fai


@pytest.fixture
def minimal_varlen_inputs():
    q = torch.randn(1, 1, 1)
    k = torch.randn(1, 1, 1)
    v = torch.randn(1, 1, 1)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seqused_k = torch.tensor([1], dtype=torch.int32)
    return q, k, v, cu_seqlens_q, seqused_k


@pytest.mark.parametrize(
    ("fa_version", "kwargs", "expected_message"),
    [
        (
            2,
            {"s_aux": torch.tensor([0.0])},
            "FlashAttention V2 does not support the 's_aux' feature.",
        ),
        (
            3,
            {"alibi_slopes": torch.tensor([1.0])},
            "FlashAttention V3 does not support the 'alibi_slopes' feature.",
        ),
        (
            4,
            {"alibi_slopes": torch.tensor([1.0])},
            "FlashAttention V4 does not support the 'alibi_slopes' feature.",
        ),
    ],
)
def test_unsupported_features_raise_standardized_error(
    minimal_varlen_inputs, fa_version, kwargs, expected_message
):
    q, k, v, cu_seqlens_q, seqused_k = minimal_varlen_inputs

    with pytest.raises(NotImplementedError, match=re.escape(expected_message)):
        fai.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=1,
            seqused_k=seqused_k,
            fa_version=fa_version,
            **kwargs,
        )


def test_unsupported_fa_version_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported FA version: 999"):
        fai.is_fa_version_supported(999)

    with pytest.raises(ValueError, match="Unsupported FA version: 999"):
        fai.fa_version_unsupported_reason(999)
