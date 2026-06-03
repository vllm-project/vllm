# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MemoryModel,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _mla_attention_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )


def _chunked_local_attention_spec() -> ChunkedLocalAttentionSpec:
    return ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        attention_chunk_size=128,
    )


def _sliding_window_spec() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        sliding_window=128,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((4, 8), (2, 8)),
        dtypes=(torch.float16, torch.float32),
    )


def _encoder_only_attention_spec() -> EncoderOnlyAttentionSpec:
    return EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _cross_attention_spec() -> CrossAttentionSpec:
    return CrossAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _sink_full_attention_spec() -> SinkFullAttentionSpec:
    return SinkFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        sink_len=4,
    )


def _uniform_type_kv_cache_specs() -> UniformTypeKVCacheSpecs:
    return UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": _full_attention_spec(),
            "layer.1": _full_attention_spec(),
        },
    )


TOKEN_PROPORTIONAL_SPEC_FACTORIES: list[tuple[str, Callable[[], KVCacheSpec]]] = [
    ("FullAttentionSpec", _full_attention_spec),
    ("MLAAttentionSpec", _mla_attention_spec),
    ("ChunkedLocalAttentionSpec", _chunked_local_attention_spec),
    ("SlidingWindowSpec", _sliding_window_spec),
    ("EncoderOnlyAttentionSpec", _encoder_only_attention_spec),
    ("CrossAttentionSpec", _cross_attention_spec),
    ("SinkFullAttentionSpec", _sink_full_attention_spec),
    ("UniformTypeKVCacheSpecs", _uniform_type_kv_cache_specs),
]


@pytest.mark.parametrize(
    ("spec_name", "make_spec"),
    TOKEN_PROPORTIONAL_SPEC_FACTORIES,
    ids=[name for name, _ in TOKEN_PROPORTIONAL_SPEC_FACTORIES],
)
def test_default_memory_model_is_token_proportional(
    spec_name: str, make_spec: Callable[[], KVCacheSpec]
) -> None:
    spec = make_spec()

    assert spec.memory_model == MemoryModel.TOKEN_PROPORTIONAL, spec_name
    assert spec.accounting_page_size_bytes == spec.page_size_bytes, spec_name
    assert spec.requires_block_zeroing_on_alloc is True, spec_name


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align", "all"])
def test_mamba_zeroing_metadata_matches_current_zeroer(
    mamba_cache_mode: str,
) -> None:
    spec = MambaSpec(
        block_size=16,
        shapes=((4, 8), (2, 8)),
        dtypes=(torch.float16, torch.float32),
        mamba_cache_mode=mamba_cache_mode,
    )

    expected_memory_model = (
        MemoryModel.TOKEN_PROPORTIONAL
        if mamba_cache_mode == "all"
        else MemoryModel.REQUEST_CONSTANT
    )
    assert spec.memory_model == expected_memory_model
    assert spec.accounting_page_size_bytes == spec.page_size_bytes
    assert spec.requires_block_zeroing_on_alloc is False


def test_mamba_physical_page_size_excludes_accounting_padding() -> None:
    spec = MambaSpec(
        block_size=16,
        shapes=((4, 8), (2, 8)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=1024,
    )
    expected_physical = (4 * 8 * get_dtype_size(torch.float16)) + (
        2 * 8 * get_dtype_size(torch.float32)
    )

    assert spec.physical_page_size_bytes == expected_physical
    assert spec.page_size_bytes == 1024
    assert spec.accounting_page_size_bytes == 1024


def test_uniform_type_physical_page_size_sums_children() -> None:
    full_spec = _full_attention_spec()
    mla_spec = _mla_attention_spec()
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "full": full_spec,
            "mla": mla_spec,
        },
    )

    assert uniform_spec.physical_page_size_bytes == (
        full_spec.physical_page_size_bytes + mla_spec.physical_page_size_bytes
    )
