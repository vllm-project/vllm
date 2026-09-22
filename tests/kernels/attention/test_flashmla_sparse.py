# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


def test_deepseek_v4_c128a_dynamic_topk_packed_buffers():
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    capacity_width = 256
    active_width = 128
    global_decode_buffer = torch.empty(
        (2, capacity_width), dtype=torch.int32, device=device
    )
    decode_lens_buffer = torch.empty(2, dtype=torch.int32, device=device)
    prefill_buffer = torch.empty((2, capacity_width), dtype=torch.int32, device=device)

    global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
        positions=torch.tensor([255, 511], dtype=torch.int64, device=device),
        compress_ratio=128,
        num_decode_tokens=1,
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=device),
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=decode_lens_buffer,
        prefill_buffer=prefill_buffer,
        max_compressed_tokens=active_width,
    )

    assert global_decode.shape == (1, active_width)
    assert prefill_local.shape == (1, active_width)
    assert global_decode.stride() == (active_width, 1)
    assert prefill_local.stride() == (active_width, 1)
    assert global_decode[0, :2].cpu().tolist() == [768, 769]
    assert decode_lens.cpu().tolist() == [2]
    assert prefill_local[0, :4].cpu().tolist() == list(range(4))
    assert torch.all(global_decode[0, 2:] == -1)
    assert torch.all(prefill_local[0, 4:] == -1)


def test_sparse_flashmla_metadata_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )
    assert isinstance(tile_md, fm.FlashMLASchedMeta)
    assert tile_md.tile_scheduler_metadata is None
    assert tile_md.num_splits is None
    assert num_splits is None


def test_sparse_flashmla_decode_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 64
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    # q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )

    # Inputs
    q = torch.zeros(
        (batch_size, seqlen_q, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = torch.zeros(
        (1, page_block_size, num_heads_k, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    indices = torch.zeros(
        (batch_size, seqlen_q, topk), dtype=torch.int32, device=device
    )

    block_table = torch.zeros((batch_size, 128), dtype=torch.int32, device=device)
    out, lse = fm.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_md,
        num_splits,
        indices=indices,
        is_fp8_kvcache=True,
    )
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


@pytest.mark.parametrize("h_q", [64, 128])
def test_sparse_flashmla_prefill_smoke(h_q: int):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(0)
    s_q = 1
    s_kv = 8
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.randint(s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device)
    reference_indices = indices.clone()
    reference_indices[..., 1:] = -1
    kwargs = {"topk_length": torch.ones(1, dtype=torch.int32, device=device)}
    reference = fm.flash_mla_sparse_fwd(q, kv, reference_indices, 1.0, d_v, **kwargs)
    actual = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0, d_v, **kwargs)

    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert actual[0].shape == (s_q, h_q, d_v)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_sparse_flashmla_packed_request_offsets_are_bitwise(batch_size: int):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(17)
    h_q, d_qk, d_v, width = 64, 576, 512, 128
    lengths = [8 if index % 2 == 0 else 12 for index in range(batch_size)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    q = torch.randn((batch_size, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((offsets[-1], 1, d_qk), dtype=torch.bfloat16, device=device)
    local_indices = torch.full(
        (batch_size, 1, width), -1, dtype=torch.int32, device=device
    )
    local_indices[:, 0, :4] = torch.tensor([0, 1, 3, 7], device=device)
    packed_indices = local_indices.clone()
    packed_indices[:, :, :4] += torch.tensor(
        offsets[:-1], dtype=torch.int32, device=device
    ).view(-1, 1, 1)
    topk_length = torch.full((batch_size,), 4, dtype=torch.int32, device=device)

    packed = fm.flash_mla_sparse_fwd(
        q, kv, packed_indices, 1.0, d_v, topk_length=topk_length
    )[0]
    references = []
    for index in range(batch_size):
        references.append(
            fm.flash_mla_sparse_fwd(
                q[index : index + 1],
                kv[offsets[index] : offsets[index + 1]],
                local_indices[index : index + 1],
                1.0,
                d_v,
                topk_length=topk_length[index : index + 1],
            )[0]
        )
    reference = torch.cat(references)
    torch.testing.assert_close(packed, reference, rtol=0, atol=0)


def test_deepseek_v4_prefill_chunk_planning_expands_for_short_sequences():
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=5,
        prefill_seq_lens_cpu=torch.tensor([80, 96, 112, 128, 144], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([4, 4, 4, 4, 4], dtype=torch.int32),
        prefill_window_size=64,
        prefill_max_model_len=1024,
        prefill_max_num_batched_tokens=128,
    )

    chunk_plan = metadata.get_prefill_chunk_plan(compress_ratio=4, prefill_chunk_size=4)

    # the adaptive plan keeps all 5 in one chunk
    assert chunk_plan == [(0, 5, 36, 103)]


@pytest.mark.parametrize(
    "compress_ratio,expected",
    [
        (1, [(0, 1, 0, 80), (1, 2, 0, 96), (2, 3, 0, 112)]),
        (4, [(0, 1, 20, 100), (1, 2, 24, 120), (2, 3, 28, 140)]),
    ],
)
def test_deepseek_v4_batch_invariant_prefill_chunks_are_request_local(
    compress_ratio, expected
):
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        _batch_invariant_prefill_chunk_plan,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=3,
        prefill_seq_lens_cpu=torch.tensor([80, 96, 112], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([80, 96, 112], dtype=torch.int32),
        prefill_window_size=128,
    )

    assert (
        _batch_invariant_prefill_chunk_plan(metadata, compress_ratio, 128) == expected
    )


def test_deepseek_v4_batch_invariant_decode_is_request_local():
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        _batch_invariant_decode_request_ranges,
    )

    assert _batch_invariant_decode_request_ranges(
        torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        torch.tensor([2048, 6144, 6144], dtype=torch.int32),
        num_decodes=3,
        compress_ratio=4,
        window_size=128,
    ) == [(0, 0, 1, 1), (1, 1, 1, 2), (2, 3, 1, 3)]


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("seq_len", [2048, 6144])
def test_deepseek_v4_batch_invariant_identical_shapes_stay_request_local(
    batch_size: int, seq_len: int
):
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        _batch_invariant_decode_request_ranges,
        _batch_invariant_prefill_chunk_plan,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    query_len = 2
    offsets = torch.arange(0, (batch_size + 1) * query_len, query_len, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    decode_plan = _batch_invariant_decode_request_ranges(
        offsets, seq_lens, batch_size, compress_ratio=4, window_size=128
    )
    assert decode_plan == [
        (request_index, request_index * query_len, 1, query_len)
        for request_index in range(batch_size)
    ]

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=batch_size,
        prefill_seq_lens_cpu=seq_lens,
        prefill_query_lens_cpu=torch.full(
            (batch_size,), query_len, dtype=torch.int32
        ),
        prefill_window_size=128,
    )
    assert _batch_invariant_prefill_chunk_plan(metadata, 4, 128) == [
        (request_index, request_index + 1, seq_len // 4, seq_len // 4 + 129)
        for request_index in range(batch_size)
    ]


def test_deepseek_v4_batch_invariant_ragged_ranges_stay_request_local():
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        _batch_invariant_decode_request_ranges,
    )

    batch_size = 32
    query_lens = torch.tensor(([1, 2, 3, 4] * 8), dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0)))
    seq_lens = torch.tensor(([2048, 2048, 6144, 6144] * 8), dtype=torch.int32)
    plan = _batch_invariant_decode_request_ranges(
        offsets, seq_lens, batch_size, compress_ratio=4, window_size=128
    )
    assert len(plan) == batch_size
    assert all(request_count == 1 for _, _, request_count, _ in plan)


def test_flashinfer_sparse_indices_cache(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    builder_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return (
            torch.tensor([[builder_calls]], dtype=torch.int32),
            torch.tensor([builder_calls], dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    def make_attn(compress_ratio: int, topk_width: int):
        attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
        attn.compress_ratio = compress_ratio
        attn.window_size = 4
        attn.topk_indices_buffer = torch.tensor(
            [[0, 1], [2, 3], [4, 5]], dtype=torch.int32
        )[:, :topk_width]
        return attn

    def make_swa_metadata():
        return DeepseekSparseSWAMetadata(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            block_size=64,
            seq_lens=torch.tensor([8, 10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 1], dtype=torch.int32),
            decode_swa_indices=torch.tensor([[5, 6, -1, -1]], dtype=torch.int32),
            decode_swa_lens=torch.tensor([2], dtype=torch.int32),
            is_valid_token=torch.tensor([True], dtype=torch.bool),
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=1,
            num_prefill_tokens=2,
        )

    def make_flashmla_metadata():
        return DeepseekV4FlashMLAMetadata(
            num_reqs=2,
            max_query_len=2,
            max_seq_len=10,
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            block_size=256,
            topk_tokens=2,
            c128a_global_decode_topk_indices=torch.tensor(
                [[[9, 10]]], dtype=torch.int32
            ),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            c128a_prefill_topk_indices=torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.int32
            ),
        )

    swa_attn = make_attn(1, 0)
    swa_metadata = make_swa_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    assert builder_calls == 1
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c128a_attn = make_attn(128, 2)
    c128a_metadata = make_swa_metadata()
    c128a_flashmla_md = make_flashmla_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 2
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c4a_attn = make_attn(4, 2)
    c4a_metadata = make_swa_metadata()
    c4a_flashmla_md = make_flashmla_metadata()
    c4a_flashmla_md.c128a_global_decode_topk_indices = None
    c4a_flashmla_md.c128a_decode_topk_lens = None
    c4a_flashmla_md.c128a_prefill_topk_indices = None
    _, _, sparse_indices_third, sparse_lens_third = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_fourth, sparse_lens_fourth = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 4
    assert sparse_indices_third is not sparse_indices_fourth
    assert sparse_lens_third is not sparse_lens_fourth
