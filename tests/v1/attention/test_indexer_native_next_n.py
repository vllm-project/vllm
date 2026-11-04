# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

import pytest

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _paged_mqa_logits_schedule_slots
from vllm.v1.attention.backends.mla import indexer

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)


@pytest.mark.parametrize(
    "family,expected_native",
    [
        # SM90 gained next_n=4 (MTP=3) via 2-CTA multicast, but never 3.
        (9, {1, 2, 4}),
        # SM100 schedules any next_n with multi-atom tiles.
        (10, {1, 2, 3, 4, 5, 8}),
        # SM120 advertises multi-atom too but is unvalidated on hardware, so
        # it stays on the conservative gate. Loosen it only with measurements.
        (12, {1, 2}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, expected_native):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize(
    "cuda,deep_gemm", [(False, True), (True, False), (False, False)]
)
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("next_n", [1, 2, 4, 6])
def test_sm90_sparse_rows_preserve_dense_source_layout(tmp_path, next_n):
    """Sparse consumers expand native rows once without changing the source
    layer's native decode metadata or its dense prefill chunking.
    """
    import torch
    from transformers import LlamaConfig

    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import SpeculativeConfig
    from vllm.utils.deep_gemm import is_sm90_paged_sparse_mqa_logits_supported
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    if not current_platform.is_device_capability_family(90):
        pytest.skip("SM90 required")
    if not is_sm90_paged_sparse_mqa_logits_supported(32, 128, 64, 8):
        pytest.skip("DeepGEMM SM90 paged sparse support required")
    # Metadata only: a real local config avoids downloading model weights.
    LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=32,
        max_position_embeddings=8192,
        architectures=["LlamaForCausalLM"],
        index_n_heads=32,
        index_head_dim=128,
        candidate_block_size=8,
        candidate_topk_blocks=2048,
        index_topk=512,
    ).save_pretrained(tmp_path)
    config = create_vllm_config(
        model_name=str(tmp_path),
        max_model_len=8192,
        block_size=64,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )
    if next_n > 1:
        config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=next_n - 1,
        )
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    common = create_common_attn_metadata(
        BatchSpec([1024, 2048, 4096], [next_n, next_n, 96]),
        64,
        torch.device("cuda"),
        arange_block_indices=True,
    )
    kwargs = dict(
        kv_cache_spec=spec,
        layer_names=["indexer"],
        vllm_config=config,
        device=torch.device("cuda"),
        block_table_width=64,
    )
    dense_builder = indexer.DeepseekV32IndexerMetadataBuilder(**kwargs)
    sparse_builder = DeepseekV41SparseIndexerMetadataBuilder(**kwargs)
    dense, sparse = dense_builder.build(0, common), sparse_builder.build(0, common)
    assert sparse.sparse_prefill is None
    assert sparse.prefill is not None and dense.prefill is not None
    assert sparse.decode is not None and dense.decode is not None
    assert len(sparse.prefill.chunks) == len(dense.prefill.chunks)
    torch.testing.assert_close(sparse.decode.seq_lens, dense.decode.seq_lens)
    torch.testing.assert_close(sparse.decode.block_table, dense.decode.block_table)
    rows = sparse.sparse_decode
    assert rows is not None
    assert rows.row_ke.shape == (2 * next_n,)
    expected_lens = torch.cat(
        [
            torch.arange(
                length - next_n + 1, length + 1, device="cuda", dtype=torch.int32
            )
            for length in (1024, 2048)
        ]
    )
    torch.testing.assert_close(rows.row_ke, expected_lens)
    torch.testing.assert_close(
        rows.block_table, common.block_table_tensor[:2].repeat_interleave(next_n, dim=0)
    )
    # Metadata objects are step-local even though their scratch storage is reused.
    rows.kernel_metadata = torch.zeros(1, dtype=torch.int64, device="cuda")
    next_rows = sparse_builder.build(0, common).sparse_decode
    assert next_rows is not None and next_rows.kernel_metadata is None
