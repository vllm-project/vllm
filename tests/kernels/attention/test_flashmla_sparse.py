# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


def test_dsv41_decode_reuses_physical_selection_until_source_or_metadata_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Followers attend with the source's exact physical IDs and lengths;
    a new source or request metadata must publish a fresh selection."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.nvidia import flashmla

    mapping_calls = []
    attended = []

    def map_selection(
        topk, token_to_req, block_table, block_size, is_valid, *, output_buffers=None
    ):
        mapping_calls.append(output_buffers)
        physical, lengths = (
            output_buffers
            if output_buffers is not None
            else (torch.empty_like(topk), torch.empty(topk.shape[0], dtype=torch.int32))
        )
        local = topk.clamp(min=0)
        req = token_to_req[:, None]
        slot_ids = (
            block_table[req, local // block_size] * block_size + local % block_size
        )
        physical.copy_(torch.where(topk >= 0, slot_ids, -1))
        lengths.copy_((topk >= 0).sum(dim=1).to(torch.int32) * is_valid.to(torch.int32))
        return physical, lengths

    def attend(**kwargs):
        attended.append(
            (
                kwargs["extra_indices_in_kvcache"].clone(),
                kwargs["extra_topk_length"].clone(),
            )
        )
        return kwargs["out"], None

    monkeypatch.setattr(flashmla, "compute_global_topk_indices_and_lens", map_selection)
    monkeypatch.setattr(flashmla, "flash_mla_with_kvcache", attend)
    topk = torch.tensor([[0, 3, -1], [1, 2, -1]], dtype=torch.int32)
    buffers = (
        torch.full((3, 3), -99, dtype=torch.int32),
        torch.full((3,), -99, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        topk_indices_buffer=topk,
        physical_selection_buffers=buffers,
        index_source_layer_id=24,
        compress_ratio=1,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.empty(1, 2, 1)),
        scale=1.0,
        attn_sink=torch.empty(1),
    )
    source = SimpleNamespace(
        block_size=2,
        block_table=torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
        physical_selection_source_id=None,
        physical_selection_swa_metadata=None,
        physical_selection=None,
    )
    swa = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        max_decode_query_len=1,
        is_valid_token=torch.tensor([True, False]),
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        decode_swa_indices=torch.empty(2, 1, 1, dtype=torch.int32),
        decode_swa_lens=torch.empty(2, dtype=torch.int32),
        tile_sched_swaonly=object(),
        tile_sched_c1a=object(),
        tile_sched_c2a=object(),
    )
    q = torch.empty(2, 1, 1)
    output = torch.empty_like(q)

    def forward():
        flashmla.DeepseekV4FlashMLAAttention._forward_decode(
            layer, q, torch.empty(1, 2, 1), swa, source, False, output
        )

    for _ in range(4):
        forward()
    assert len(mapping_calls) == 1
    assert mapping_calls[0] is not None
    assert attended[-1][0][:, 0].tolist() == [[10, 13, -1], [15, 16, -1]]
    assert attended[-1][1].tolist() == [2, 0]

    layer.index_source_layer_id = 28
    topk[0, 0] = 2
    forward()
    assert len(mapping_calls) == 2
    assert attended[-1][0][0, 0, 0] == 12

    source = SimpleNamespace(**vars(source))
    source.block_table = torch.tensor([[9, 10], [11, 12]], dtype=torch.int32)
    source.physical_selection_source_id = None
    source.physical_selection_swa_metadata = None
    source.physical_selection = None
    swa.token_to_req_indices = torch.tensor([1, 0], dtype=torch.int32)
    forward()
    assert len(mapping_calls) == 3
    assert attended[-1][0][0, 0, 0] == 24

    swa = SimpleNamespace(**vars(swa))
    swa.is_valid_token = torch.tensor([False, True])
    forward()
    assert len(mapping_calls) == 4
    assert attended[-1][1].tolist() == [0, 2]

    swa.num_prefills = 1
    forward()
    assert len(mapping_calls) == 5
    assert mapping_calls[-1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph coverage")
def test_dsv41_physical_selection_buffers_refresh_on_graph_replay() -> None:
    """Captured physicalization reads current logical IDs and block tables."""
    from vllm.models.deepseek_v41.common.ops import (
        compute_global_topk_indices_and_lens,
    )

    topk = torch.tensor([[0, 3, -1]], dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[5, 7]], dtype=torch.int32, device="cuda")
    is_valid = torch.ones(1, dtype=torch.bool, device="cuda")
    buffers = (torch.empty_like(topk), torch.empty(1, dtype=torch.int32, device="cuda"))
    args = (topk, token_to_req, block_table, 2, is_valid)

    compute_global_topk_indices_and_lens(*args, output_buffers=buffers)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute_global_topk_indices_and_lens(*args, output_buffers=buffers)

    topk[0, 0] = 2
    block_table[0, 0] = 11
    block_table[0, 1] = 13
    graph.replay()
    expected = compute_global_topk_indices_and_lens(*args)
    torch.testing.assert_close(buffers[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(buffers[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("sm120", [False, True])
def test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride(
    monkeypatch: pytest.MonkeyPatch,
    sm120: bool,
):
    from vllm.models.deepseek_v4 import sparse_mla
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(
        sparse_mla.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(12, 0) if sm120 else DeviceCapability(10, 0),
    )

    device = torch.device("cuda")
    capacity_width = 512
    global_decode_buffer = torch.empty(
        (2, capacity_width), dtype=torch.int32, device=device
    )
    prefill_buffer = torch.empty_like(global_decode_buffer)
    kwargs = dict(
        positions=torch.tensor([255, 511, 383, 639], device=device),
        compress_ratio=128,
        num_decode_tokens=2,
        token_to_req_indices=torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32, device=device
        ),
        block_table=torch.tensor([[3], [5]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=torch.empty(2, dtype=torch.int32, device=device),
        prefill_buffer=prefill_buffer,
    )
    captured_decode, _, captured_prefill = sparse_mla.build_c128a_topk_metadata(
        max_compressed_tokens=256,
        **kwargs,
    )
    # SM120 keeps the decode view contiguous across the full buffer width;
    # other backends get the active-width slice. The prefill view is always
    # narrowed.
    expected_decode_width = capacity_width if sm120 else 256
    assert captured_decode.shape == (2, expected_decode_width)
    assert captured_prefill.shape == (2, 256)
    assert captured_decode.stride(0) == captured_prefill.stride(0) == capacity_width
    assert captured_decode.is_contiguous() == sm120

    captured_rows = torch.empty((4, 4), dtype=torch.int32, device=device)
    captured_rows[:2].copy_(captured_decode[:, :4])
    captured_rows[2:].copy_(captured_prefill[:, :4])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_rows[:2].copy_(captured_decode[:, :4])
        captured_rows[2:].copy_(captured_prefill[:, :4])

    global_decode_buffer.fill_(-99)
    prefill_buffer.fill_(-99)
    sparse_mla.build_c128a_topk_metadata(
        max_compressed_tokens=128,
        **kwargs,
    )
    graph.replay()

    assert captured_rows.cpu().tolist() == [
        [1536, 1537, -1, -1],
        [2560, 2561, 2562, 2563],
        [0, 1, 2, -1],
        [0, 1, 2, 3],
    ]
    assert torch.all(global_decode_buffer[:, 128:] == -99)
    assert torch.all(prefill_buffer[:, 128:] == -99)


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
            decode_swa_width=4,
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


def test_flashinfer_sparse_index_preserves_logical_window(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    captured_shapes_and_windows: list[tuple[int, int]] = []

    def fake_build(*args, **kwargs):
        # window_size is the 12th positional arg of
        # build_flashinfer_mixed_sparse_indices.
        captured_shapes_and_windows.append((args[0].shape[-1], args[11]))
        num_tokens = args[0].shape[0] + args[3].shape[0]
        return (
            torch.zeros((num_tokens, 1), dtype=torch.int32),
            torch.zeros((num_tokens,), dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
    attn.compress_ratio = 1
    attn.window_size = 4
    attn.topk_indices_buffer = torch.zeros((4, 0), dtype=torch.int32)

    wide_width = 8
    wide_indices = torch.full((1, wide_width), -1, dtype=torch.int32)
    wide_indices[0, :2] = torch.tensor([5, 6], dtype=torch.int32)
    wide_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_size=64,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        decode_swa_indices=wide_indices,
        decode_swa_lens=torch.tensor([2], dtype=torch.int32),
        decode_swa_width=wide_width,
        is_valid_token=torch.tensor([True], dtype=torch.bool),
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
    )
    attn._build_sparse_index_metadata(
        kv_cache=None,
        swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
        swa_metadata=wide_metadata,
        attn_metadata=None,
        swa_only=True,
    )
    assert captured_shapes_and_windows == [(wide_width, attn.window_size)]

    empty_width = 8
    empty_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_size=64,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32),
        decode_swa_indices=torch.empty((0, 1, empty_width), dtype=torch.int32),
        decode_swa_lens=torch.empty((0,), dtype=torch.int32),
        decode_swa_width=empty_width,
        is_valid_token=torch.tensor([True, True], dtype=torch.bool),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        num_prefill_tokens=2,
    )
    attn._build_sparse_index_metadata(
        kv_cache=None,
        swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
        swa_metadata=empty_metadata,
        attn_metadata=None,
        swa_only=True,
    )
    assert captured_shapes_and_windows == [
        (wide_width, attn.window_size),
        (empty_width, attn.window_size),
    ]


def test_flashinfer_mixed_sparse_indices_separates_window_and_padded_width():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
    )

    device = torch.device("cuda")
    padded_width = 8
    logical_window = 4
    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=torch.empty(
            (0, padded_width), dtype=torch.int32, device=device
        ),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((1, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([logical_window], dtype=torch.int32, device=device),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32, device=device),
        swa_block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        swa_block_size=64,
        compressed_block_table=None,
        compressed_block_size=64,
        window_size=logical_window,
        compress_ratio=1,
        topk=0,
    )

    assert sparse_indices.shape == (1, padded_width)
    assert sparse_indices[0].cpu().tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert sparse_lens.cpu().tolist() == [padded_width]
