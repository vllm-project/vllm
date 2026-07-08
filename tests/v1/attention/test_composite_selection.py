# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-role (prefill/decode) backend selection and the composite
attention backend (PR1, Part B). Config / helper tests are CPU-only; selection
tests that resolve real backends are gated on CUDA."""

import pytest
import torch
from pydantic import ValidationError

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.attention import AttentionConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    QueryLenSupport,
)
from vllm.v1.attention.backends.composite import (
    CompositeMetadata,
    make_composite_backend,
)
from vllm.v1.attention.backends.utils import (
    kv_layouts_compatible,
    split_common_attn_metadata,
)
from vllm.v1.attention.selector import AttentionSelectorConfig

# --------------------------------------------------------------------------- #
# Config: mutual exclusion + resolved_backends()
# --------------------------------------------------------------------------- #


def test_resolved_backends_only_backend():
    c = AttentionConfig(backend="FLASH_ATTN")
    p, d = c.resolved_backends()
    assert p is d
    assert p.name == "FLASH_ATTN"


def test_resolved_backends_only_prefill():
    c = AttentionConfig(prefill_backend="FLASH_ATTN")
    p, d = c.resolved_backends()
    assert p.name == "FLASH_ATTN"
    assert d is None


def test_resolved_backends_only_decode():
    c = AttentionConfig(decode_backend="FLASHINFER")
    p, d = c.resolved_backends()
    assert p is None
    assert d.name == "FLASHINFER"


def test_resolved_backends_prefill_and_decode():
    c = AttentionConfig(prefill_backend="FLASHINFER", decode_backend="FLASH_ATTN")
    p, d = c.resolved_backends()
    assert p.name == "FLASHINFER"
    assert d.name == "FLASH_ATTN"


def test_resolved_backends_default_is_none():
    assert AttentionConfig().resolved_backends() == (None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "FLASH_ATTN", "prefill_backend": "FLASHINFER"},
        {"backend": "FLASH_ATTN", "decode_backend": "FLASHINFER"},
        {
            "backend": "FLASH_ATTN",
            "prefill_backend": "FLASHINFER",
            "decode_backend": "FLASHINFER",
        },
    ],
)
def test_backend_mutually_exclusive_with_role_backends(kwargs):
    # pydantic wraps the model_validator's ValueError in a ValidationError.
    with pytest.raises(ValidationError):
        AttentionConfig(**kwargs)


def test_auto_string_maps_to_none():
    c = AttentionConfig(prefill_backend="auto", decode_backend="auto")
    assert c.resolved_backends() == (None, None)


# --------------------------------------------------------------------------- #
# split_common_attn_metadata round-trip
# --------------------------------------------------------------------------- #


def _make_common(query_lens, seq_lens):
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    num_tokens = int(qsl[-1].item())
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.arange(len(query_lens) * 4).reshape(-1, 4),
        slot_mapping=torch.arange(num_tokens),
    )


def test_split_common_attn_metadata_roundtrip():
    cm = _make_common([1, 1, 3, 4], [10, 11, 12, 13])
    decode, prefill = split_common_attn_metadata(cm, num_decodes=2, num_decode_tokens=2)

    assert decode.num_reqs == 2
    assert decode.num_actual_tokens == 2
    assert decode.max_query_len == 1
    assert decode.query_start_loc.tolist() == [0, 1, 2]
    assert decode.slot_mapping.tolist() == [0, 1]

    assert prefill.num_reqs == 2
    assert prefill.num_actual_tokens == 7
    assert prefill.max_query_len == 4
    # query_start_loc rebased to start at 0
    assert prefill.query_start_loc.tolist() == [0, 3, 7]
    assert prefill.slot_mapping.tolist() == list(range(2, 9))
    assert prefill.seq_lens.tolist() == [12, 13]


def test_split_common_attn_metadata_all_decode():
    cm = _make_common([1, 1, 1], [5, 6, 7])
    decode, prefill = split_common_attn_metadata(cm, num_decodes=3, num_decode_tokens=3)
    assert decode.num_reqs == 3
    assert prefill.num_reqs == 0
    assert prefill.num_actual_tokens == 0


# --------------------------------------------------------------------------- #
# kv_layouts_compatible
# --------------------------------------------------------------------------- #


class _WritesInForwardBackend(AttentionBackend):
    """A minimal non-externalized backend (writes KV inside forward)."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "WRITES_IN_FORWARD"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


def _selector_config():
    return AttentionSelectorConfig(
        head_size=128, dtype=torch.bfloat16, kv_cache_dtype=None, block_size=16
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="resolves real CUDA backends"
)
def test_kv_layouts_compatible_same_backend():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    with set_current_vllm_config(VllmConfig()):
        assert kv_layouts_compatible(
            FlashAttentionBackend, FlashAttentionBackend, _selector_config()
        )


def test_kv_layouts_incompatible_when_not_externalized():
    # A backend that writes KV inside forward can never compose.
    assert not kv_layouts_compatible(
        _WritesInForwardBackend, _WritesInForwardBackend, _selector_config()
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="imports MLA + standard backends"
)
def test_kv_layouts_incompatible_across_families():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend

    with set_current_vllm_config(VllmConfig()):
        assert not kv_layouts_compatible(
            FlashMLABackend, FlashAttentionBackend, _selector_config()
        )


# --------------------------------------------------------------------------- #
# Decode-first selection via get_attn_backend
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="resolves real CUDA backends"
)
def test_selection_default_is_single_backend():
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    _cached_get_attn_backend.cache_clear()
    with set_current_vllm_config(VllmConfig()):
        backend = get_attn_backend(128, torch.bfloat16, None, num_heads=32)
    assert "COMPOSITE" not in backend.get_name()


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="resolves real CUDA backends"
)
def test_selection_same_role_backend_short_circuits():
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    _cached_get_attn_backend.cache_clear()
    ac = AttentionConfig(prefill_backend="FLASH_ATTN", decode_backend="FLASH_ATTN")
    with set_current_vllm_config(VllmConfig(attention_config=ac)):
        backend = get_attn_backend(128, torch.bfloat16, None, num_heads=32)
    assert "COMPOSITE" not in backend.get_name()
    assert backend.get_name() == "FLASH_ATTN"


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="resolves real CUDA backends"
)
def test_selection_differing_backends_builds_composite():
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    _cached_get_attn_backend.cache_clear()
    ac = AttentionConfig(prefill_backend="FLASHINFER", decode_backend="FLASH_ATTN")
    with set_current_vllm_config(VllmConfig(attention_config=ac)):
        backend = get_attn_backend(128, torch.bfloat16, None, num_heads=32)
    assert backend.get_name() == "COMPOSITE[FLASHINFER|FLASH_ATTN]"
    assert backend.is_composite()
    assert backend.forward_includes_kv_cache_update is False
    # decode-first: composite inherits decode backend's KV shape
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    assert backend.get_kv_cache_shape(
        4, 16, 8, 128
    ) == FlashAttentionBackend.get_kv_cache_shape(4, 16, 8, 128)


# --------------------------------------------------------------------------- #
# Composite metadata build routing (fake sub-builders, no GPU)
# --------------------------------------------------------------------------- #


class _RecordingBuilder(AttentionMetadataBuilder):
    query_len_support = QueryLenSupport.SINGLE_ONLY
    tag = "?"

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.reorder_batch_threshold = 1
        self.calls = []

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        self.calls.append(
            (common_attn_metadata.num_reqs, common_attn_metadata.num_actual_tokens)
        )
        return (self.tag, common_attn_metadata.num_reqs)


def _fake_backend(name, builder_cls):
    return type(
        f"Fake{name}Backend",
        (AttentionBackend,),
        {
            "get_name": staticmethod(lambda: name),
            "get_builder_cls": classmethod(lambda cls: builder_cls),
            "get_impl_cls": staticmethod(lambda: object),
            "get_kv_cache_shape": staticmethod(lambda *a, **k: (1, 2, 3, 4, 5)),
            "is_mla": classmethod(lambda cls: False),
            "forward_includes_kv_cache_update": False,
        },
    )


def test_composite_build_routes_slices_to_sub_builders():
    decode_builder = type("DecodeRecBuilder", (_RecordingBuilder,), {"tag": "D"})
    prefill_builder = type("PrefillRecBuilder", (_RecordingBuilder,), {"tag": "P"})
    decode_backend = _fake_backend("DEC", decode_builder)
    prefill_backend = _fake_backend("PRE", prefill_builder)

    composite = make_composite_backend(
        prefill_backend=prefill_backend, decode_backend=decode_backend
    )
    builder = composite.get_builder_cls()(
        kv_cache_spec=None,
        layer_names=[],
        vllm_config=None,
        device=torch.device("cpu"),
    )

    cm = _make_common([1, 1, 3, 4], [10, 11, 12, 13])
    meta = builder.build(0, cm)

    assert isinstance(meta, CompositeMetadata)
    assert meta.num_decodes == 2
    assert meta.num_prefills == 2
    assert meta.num_decode_tokens == 2
    assert meta.num_prefill_tokens == 7
    # decode sub-builder saw the decode slice (2 reqs / 2 tokens)
    assert builder.decode_builder.calls == [(2, 2)]
    # prefill sub-builder saw the prefill slice (2 reqs / 7 tokens)
    assert builder.prefill_builder.calls == [(2, 7)]
    assert meta.decode_metadata == ("D", 2)
    assert meta.prefill_metadata == ("P", 2)


def test_composite_build_all_decode_skips_prefill_builder():
    decode_builder = type("DecodeRecBuilder2", (_RecordingBuilder,), {"tag": "D"})
    prefill_builder = type("PrefillRecBuilder2", (_RecordingBuilder,), {"tag": "P"})
    decode_backend = _fake_backend("DEC2", decode_builder)
    prefill_backend = _fake_backend("PRE2", prefill_builder)

    composite = make_composite_backend(
        prefill_backend=prefill_backend, decode_backend=decode_backend
    )
    builder = composite.get_builder_cls()(
        kv_cache_spec=None,
        layer_names=[],
        vllm_config=None,
        device=torch.device("cpu"),
    )

    cm = _make_common([1, 1, 1], [5, 6, 7])
    meta = builder.build(0, cm)

    assert meta.num_decodes == 3
    assert meta.num_prefills == 0
    assert builder.decode_builder.calls == [(3, 3)]
    assert builder.prefill_builder.calls == []  # never built
    assert meta.prefill_metadata is None
