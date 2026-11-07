# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which index group, if any, a sparse MLA layer joins."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.attention import HiSparseConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla import index_group as index_group_module
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Impl,
)
from vllm.v1.attention.backends.mla.index_group import (
    HiSparseMLAIndexGroup,
    SparseMLAIndexGroupBuilder,
)

NUM_TOKENS, TOPK_WIDTH, DECODE_ROWS, NUM_LAYERS = 64, 256, 8, 3
NUM_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, V_HEAD_DIM = 16, 512, 256, 256

pytestmark = pytest.mark.usefixtures("default_vllm_config", "dist_init")


def _make_impl(
    impl_cls,
    buffer: torch.Tensor,
    *,
    has_indexer: bool = True,
    kv_cache_dtype: str = "auto",
    **kwargs,
):
    kv_b_proj = ColumnParallelLinear(
        KV_LORA_RANK,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        bias=False,
        prefix="kv_b_proj",
    )
    return impl_cls(
        num_heads=NUM_HEADS,
        head_size=KV_LORA_RANK,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=1536,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=0,
        qk_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
        # The impl only reads the buffer from its indexer.
        indexer=SimpleNamespace(topk_indices_buffer=buffer) if has_indexer else None,
        topk_indices_buffer=buffer,
        **kwargs,
    )


def _buffer() -> torch.Tensor:
    return torch.zeros(NUM_TOKENS, TOPK_WIDTH, dtype=torch.int32)


def test_impl_that_converts_inline_allocates_no_workspace():
    impls = [
        _make_impl(FlashInferMLASparseSM90Impl, _buffer()) for _ in range(NUM_LAYERS)
    ]

    assert all(impl.index_group is None for impl in impls)


def test_opted_out_impl_ignores_an_explicit_builder():
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    impl = _make_impl(FlashInferMLASparseSM90Impl, buffer, index_group_builder=builder)

    assert impl.index_group is None
    assert builder.current_group is None


def test_opted_out_producer_ends_the_group_of_later_skip_layers():
    # The opted-out layer's indexer rewrites the shared top-k buffer, so a later
    # skip layer must not reuse the conversion of the first layer's indices.
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    producer = _make_impl(FlashInferMLASparseImpl, buffer, index_group_builder=builder)
    _make_impl(FlashInferMLASparseSM90Impl, buffer, index_group_builder=builder)
    skip = _make_impl(
        FlashInferMLASparseImpl,
        buffer,
        has_indexer=False,
        index_group_builder=builder,
    )

    assert skip.index_group is not None
    assert skip.index_group is not producer.index_group
    assert not skip.index_group.has_indexer


def test_skip_layers_with_another_kv_cache_dtype_form_their_own_group():
    # With --kv-cache-dtype-skip-layers a skip layer can route a batch to another
    # prefill path than its producer, so it cannot read the producer's rows.
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    producer = _make_impl(FlashInferMLASparseImpl, buffer, index_group_builder=builder)
    first, second = (
        _make_impl(
            FlashInferMLASparseImpl,
            buffer,
            has_indexer=False,
            kv_cache_dtype="fp8",
            index_group_builder=builder,
        )
        for _ in range(2)
    )

    assert first.index_group is not producer.index_group
    assert not first.index_group.has_indexer
    assert second.index_group is first.index_group
    assert second.index_group_index == 1


def test_hisparse_keeps_the_group_of_an_opted_out_impl(
    monkeypatch, default_vllm_config
):
    # HiSparse binds each layer's cache handle from its group, so a configured
    # HiSparse must not lose it. Platform validation of the field is bypassed,
    # and the handle, which needs the CUDA HiSparse runtime, is stubbed.
    monkeypatch.setattr(
        default_vllm_config.attention_config, "hisparse_config", HiSparseConfig()
    )
    monkeypatch.setattr(
        index_group_module,
        "create_hisparse_cache_handle",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    impl = _make_impl(FlashInferMLASparseSM90Impl, buffer, index_group_builder=builder)

    assert isinstance(impl.index_group, HiSparseMLAIndexGroup)
    assert impl.index_group is builder.current_group
