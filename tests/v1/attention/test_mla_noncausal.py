# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec


class _NonCausalMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    supports_non_causal_multi_token_decode = True


class _CausalOnlyMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    pass


class _DraftMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    """Triton-like: non-causal decode is supported, DCP draft decode is not."""

    supports_non_causal_multi_token_decode = True


class _DcpCapableMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    supports_non_causal_multi_token_decode = True
    supports_non_causal_multi_token_dcp = True


def _metadata(
    query_start_loc: list[int],
    num_tokens: int | None = None,
    causal: bool = False,
) -> CommonAttentionMetadata:
    num_reqs = len(query_start_loc) - 1
    num_tokens = query_start_loc[-1] if num_tokens is None else num_tokens
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.arange(1, num_reqs + 1, dtype=torch.int32) * 100 + 8,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(
            end - start for start, end in zip(query_start_loc, query_start_loc[1:])
        ),
        max_seq_len=num_reqs * 100 + 8,
        block_table_tensor=torch.arange(num_reqs * 3, dtype=torch.int32).view(
            num_reqs, 3
        ),
        slot_mapping=torch.arange(num_tokens),
        causal=causal,
        seq_lens_cpu_upper_bound=None,
    )


def _builder(marked: bool = True) -> _NonCausalMLAMetadataBuilder:
    builder = object.__new__(_NonCausalMLAMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.reorder_batch_threshold = 1
    builder.query_len_support = QueryLenSupport.SINGLE_ONLY
    builder.non_causal_multi_token_decode = marked
    builder.dcp_world_size = 1
    builder.use_pcp = False
    builder.metadata_cls = MLACommonMetadata
    builder.model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_head_size=lambda: 576
    )
    return builder


def _mla_layer(*, non_causal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        non_causal_multi_token_decode=non_causal,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        prefill_backend=SimpleNamespace(clone=lambda: None),
    )


def _merged_mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )


def _dspark_dcp_vllm_config(
    static_forward_context: dict[str, SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark", num_speculative_tokens=None
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=128,
            get_num_attention_heads=lambda _parallel_config: 16,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        attention_config=SimpleNamespace(use_prefill_query_quantization=False),
    )


def test_group_capability_keeps_runtime_causality_per_step():
    builder = _builder()

    target_metadata = builder.build(0, _metadata([0, 1, 2], causal=True))
    assert (
        target_metadata.num_decodes,
        target_metadata.num_decode_tokens,
        target_metadata.num_prefills,
        target_metadata.causal,
    ) == (2, 2, 0, True)

    common_metadata = _metadata([0, 8, 16])
    metadata = builder.build(0, common_metadata)

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 16
    assert metadata.num_prefills == 0
    assert metadata.prefill is None
    assert metadata.decode is not None
    assert metadata.decode.block_table.shape == (2, 3)
    assert metadata.decode.seq_lens.shape == (2,)
    assert torch.equal(metadata.decode.block_table, common_metadata.block_table_tensor)
    assert torch.equal(metadata.decode.seq_lens, common_metadata.seq_lens)
    assert not metadata.causal


def test_noncausal_support_is_explicit_and_uniform():
    with pytest.raises(ValueError, match="explicitly supported"):
        _builder(marked=False).build(0, _metadata([0, 8, 16]))
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 3, 8]))


def test_noncausal_block_allows_trailing_cudagraph_padding():
    common_metadata = _metadata([0, 8, 16, 16], num_tokens=24)
    common_metadata.seq_lens[-1] = 0

    metadata = _builder().build(0, common_metadata)

    assert metadata.num_decodes == 3
    assert metadata.num_decode_tokens == 24
    assert metadata.decode is not None
    assert metadata.decode.seq_lens.tolist() == [108, 208, 0]


def test_noncausal_block_rejects_non_trailing_padding():
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 8, 8, 16], num_tokens=24))


def test_noncausal_decode_metadata_keeps_live_request_buffers():
    common_metadata = _metadata([0, 8, 16])
    metadata = _builder().build(0, common_metadata)

    assert metadata.decode is not None
    assert metadata.decode.seq_lens.data_ptr() == common_metadata.seq_lens.data_ptr()
    assert (
        metadata.decode.block_table.data_ptr()
        == common_metadata.block_table_tensor.data_ptr()
    )


def test_mla_cache_marker_is_promoted_to_group_capability():
    kwargs = {
        "block_size": 64,
        "num_kv_heads": 1,
        "head_size": 576,
        "dtype": torch.bfloat16,
    }
    marked = MLAAttentionSpec(**kwargs, non_causal_multi_token_decode=True)
    unmarked = MLAAttentionSpec(**kwargs)

    assert MLAAttentionSpec.merge([marked, marked]).non_causal_multi_token_decode
    assert not MLAAttentionSpec.merge(
        [unmarked, unmarked]
    ).non_causal_multi_token_decode
    assert MLAAttentionSpec.merge([unmarked, marked]).non_causal_multi_token_decode


def test_builder_scopes_noncausal_capability_to_its_layers():
    merged_spec = _merged_mla_spec()
    static_forward_context = {
        "target": _mla_layer(non_causal=False),
        "draft": _mla_layer(non_causal=True),
        # Indexer and compressor caches share an MLA group but predate the flag.
        "indexer": SimpleNamespace(),
    }
    vllm_config = _dspark_dcp_vllm_config(static_forward_context)
    device = torch.device("cpu")

    def _flag(layer_names: list[str]) -> bool:
        return _DcpCapableMLAMetadataBuilder(
            merged_spec,
            layer_names,
            vllm_config,
            device,
            supports_dcp_with_varlen=True,
        ).non_causal_multi_token_decode

    assert not _flag(["target"])
    assert _flag(["draft"])
    assert _flag(["target", "draft"])
    assert not _flag(["target", "indexer"])


def test_merged_group_spec_does_not_mark_a_target_only_builder():
    # Both builders receive the same merged spec; only the draft layer is
    # non-causal. Constructing through __init__ would still fail DCP
    # validation if the builder read the group flag.
    merged_spec = _merged_mla_spec()
    static_forward_context = {
        "target": _mla_layer(non_causal=False),
        "draft": _mla_layer(non_causal=True),
    }
    vllm_config = _dspark_dcp_vllm_config(static_forward_context)
    device = torch.device("cpu")
    assert merged_spec.non_causal_multi_token_decode

    target = _CausalOnlyMLAMetadataBuilder(
        merged_spec,
        ["target"],
        vllm_config,
        device,
        supports_dcp_with_varlen=True,
    )
    assert not target.non_causal_multi_token_decode

    with pytest.raises(ValueError, match="non-causal draft"):
        _DraftMLAMetadataBuilder(
            merged_spec,
            ["draft"],
            vllm_config,
            device,
            supports_dcp_with_varlen=True,
        )
