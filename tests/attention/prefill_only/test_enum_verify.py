import pytest

from vllm.attention.prefill_only.abstract import (AttentionType,
                                                  PrefillOnlyAttentionBackend)
from vllm.attention.prefill_only.selector import (AttentionImpls, AttnBackend,
                                                  _Backend)


def get_attn_backend(attention_impl: str, attn_type: str):
    selected_backend = _Backend.backend_name_to_enum(attention_impl)
    backend_cls = AttnBackend.get_backend_cls(selected_backend)

    attn_type_enum = AttentionType.attn_type_name_to_enum(attn_type)

    attn_backend = backend_cls(attn_type_enum)
    return attn_backend


@pytest.mark.parametrize("attn_type", ["DECODER", "ENCODER"])
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
def test_backend(dtype: str, attn_type: str):
    attention_impls = AttentionImpls[dtype]

    for attention_impl in attention_impls:
        attn_backend = get_attn_backend(attention_impl, attn_type)

        assert isinstance(attn_backend, PrefillOnlyAttentionBackend)


@pytest.mark.parametrize("attn_type", ["ENCODER_DECODER"])
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
def test_ENCODER_DECODER_not_supported(dtype: str, attn_type: str):
    attention_impls = AttentionImpls[dtype]

    for attention_impl in attention_impls:
        with pytest.raises(NotImplementedError):
            get_attn_backend(attention_impl, attn_type)


def test_not_supported_backend():
    attention_impls = ["not_supported_backend", 0, 1.0]

    for attention_impl in attention_impls:
        with pytest.raises(ValueError):
            selected_backend = _Backend.backend_name_to_enum(attention_impl)
            AttnBackend.get_backend_cls(selected_backend)


def test_not_supported_attn_type():
    attn_types = ["not_supported_attn_type", 0, 1.0]

    for attn_type in attn_types:
        with pytest.raises(ValueError):
            AttentionType.attn_type_name_to_enum(attn_type)
