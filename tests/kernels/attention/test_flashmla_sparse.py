# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch


def test_compute_global_topk_indices_and_lens_allows_inplace_output():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton sparse index kernel")

    from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens

    device = torch.device("cuda")
    local_indices = torch.tensor(
        [[0, 1, -1, 5], [2, -1, 3, 4]],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor([True, True], dtype=torch.bool, device=device)

    expected_indices, expected_lens = compute_global_topk_indices_and_lens(
        local_indices.clone(),
        token_to_req_indices,
        block_table,
        block_size=4,
        is_valid_token=is_valid_token,
    )

    lens = torch.empty((local_indices.shape[0],), dtype=torch.int32, device=device)
    actual_indices, actual_lens = compute_global_topk_indices_and_lens(
        local_indices,
        token_to_req_indices,
        block_table,
        block_size=4,
        is_valid_token=is_valid_token,
        output_buffers=(local_indices, lens),
    )

    assert actual_indices.data_ptr() == local_indices.data_ptr()
    assert actual_lens.data_ptr() == lens.data_ptr()
    torch.testing.assert_close(actual_indices.cpu(), expected_indices.cpu())
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())


def test_compute_global_topk_indices_and_lens_bounds_block_table_gather():
    """Out-of-range top-k indices must fall out as -1, not gather past the row.

    The indices are data: they come from the indexer's top-k, which writes into a
    ``torch.empty`` buffer shared by every layer, so an unwritten or corrupted slot
    can hold a large positive value. Checking only ``>= 0`` made that an illegal
    access on every TP rank (the failure upstream hit on SM12x once its logits
    kernel started emitting NaN bit patterns as indices).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton sparse index kernel")

    from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens

    device = torch.device("cuda")
    block_size = 4
    # Two blocks per request, so block_index 2 and above are out of range.
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device=device)
    # 0x7FC00000 is the NaN bit pattern upstream observed being written as an index.
    local_indices = torch.tensor(
        [[0, 8, 2143289344, 5], [2, -1, 3, 4]],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor([True, True], dtype=torch.bool, device=device)

    indices, lens = compute_global_topk_indices_and_lens(
        local_indices,
        token_to_req_indices,
        block_table,
        block_size=block_size,
        is_valid_token=is_valid_token,
    )

    expected = torch.tensor(
        [
            [10 * block_size + 0, -1, -1, 11 * block_size + 1],
            [20 * block_size + 2, -1, 20 * block_size + 3, 21 * block_size + 0],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(indices.cpu(), expected)
    # Rejected entries must not be counted, or downstream reads padding as real.
    torch.testing.assert_close(lens.cpu(), torch.tensor([2, 3], dtype=torch.int32))


def test_compute_global_topk_indices_and_lens_masks_padding_tokens():
    """A padding row must emit -1 slots, not real ones.

    ``d64074e6f0`` folded ``is_valid_token`` into the per-entry predicate
    alongside the block-table bound. Both existing tests pass all-True masks, so
    that half of the change had no coverage: a regression that only zeroed
    ``topk_lens`` while still writing real slot ids would pass them, and the
    stale slots would then be read by whatever consumes the buffer without
    re-checking the length.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton sparse index kernel")

    from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens

    device = torch.device("cuda")
    block_size = 4
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device=device)
    local_indices = torch.tensor(
        [[0, 1, 2, 3], [2, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # Row 1 is a padding token: every one of its indices is individually valid
    # and in range, so only the mask can reject them.
    is_valid_token = torch.tensor([True, False], dtype=torch.bool, device=device)

    indices, lens = compute_global_topk_indices_and_lens(
        local_indices,
        token_to_req_indices,
        block_table,
        block_size=block_size,
        is_valid_token=is_valid_token,
    )

    assert lens.cpu().tolist() == [4, 0]
    padded_row = indices.cpu()[1]
    assert (padded_row == -1).all(), (
        f"padding row must be all -1, got {padded_row.tolist()}"
    )


def test_fused_qnorm_rope_kv_insert_out_is_defunctionalized():
    """The op DSv4 attention actually calls must be in the defunctionalize list.

    Upstream #49236 split this op into an allocating ``..._insert`` and a
    caller-buffered ``..._insert_out``; ``DeepseekV4Attention`` calls the ``_out``
    form. Registering only the old name still imports, still serves and still
    produces correct numbers -- it just silently loses the copy elision this pass
    exists to provide, which no other test would notice.
    """
    import torch as _torch

    if not hasattr(
        _torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out"
    ):
        pytest.skip("vllm._C was not built with the DSv4 fused insert op")

    import inspect

    from vllm.compilation.passes.utility import fix_functionalization
    from vllm.models.deepseek_v4 import attention as dsv4_attention

    called = {
        name
        for name in (
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out",
        )
        if name in inspect.getsource(dsv4_attention)
    }
    assert "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out" in called

    registered = inspect.getsource(fix_functionalization)
    for name in called:
        assert f'"{name}"' in registered, (
            f"{name} is called by DSv4 attention but never appended to "
            "fused_deepseek_v4_mla_targets"
        )


def test_deepseek_v4_c128a_dynamic_topk_packed_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton C128A metadata kernel")

    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    capacity_width = 256
    active_width = 128
    # One backing matrix for both decode and prefill rows (they partition the
    # same per-step token batch), packed at the active width.
    topk_buffer = torch.empty((2, capacity_width), dtype=torch.int32, device=device)
    decode_lens_buffer = torch.empty(2, dtype=torch.int32, device=device)

    global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
        positions=torch.tensor([255, 511], dtype=torch.int64, device=device),
        compress_ratio=128,
        num_decode_tokens=1,
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=device),
        topk_buffer=topk_buffer,
        decode_lens_buffer=decode_lens_buffer,
        max_compressed_tokens=active_width,
    )

    assert global_decode.shape == (1, active_width)
    assert prefill_local.shape == (1, active_width)
    assert global_decode.stride() == (active_width, 1)
    assert prefill_local.stride() == (active_width, 1)
    # The two views must not alias: prefill starts where decode ends.
    assert prefill_local.data_ptr() > global_decode.data_ptr()
    assert global_decode[0, :2].cpu().tolist() == [768, 769]
    assert decode_lens.cpu().tolist() == [2]
    assert prefill_local[0, :4].cpu().tolist() == list(range(4))
    assert torch.all(global_decode[0, 2:] == -1)
    assert torch.all(prefill_local[0, 4:] == -1)


def test_deepseek_v4_dspark_warmup_without_topk_buffer(monkeypatch):
    """SWA-only warmup must not require the sparse top-k buffer.

    During DSpark startup profile_run calls forward_mqa with no attention
    metadata. The SWA-only draft layer (compress_ratio <= 1) never allocates
    topk_indices_buffer, so a bare assert on it fails after the full target and
    draft model are already loaded (vllm-project/vllm#50615).

    Adapted from vllm-project/vllm#50693: that PR asserts the exact workspace
    tuple produced by upstream's inlined reservation, which we do not use -- our
    reservation goes through _prefill_workspace_reservation_specs and is shared
    with the warmup module. Assert the contract instead: no crash, output zeroed,
    and a workspace still requested.
    """
    from vllm.models.deepseek_v4.nvidia import flashmla as flashmla_mod

    workspace_calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            workspace_calls.append(shapes_and_dtypes)
            return tuple(torch.empty(0) for _ in shapes_and_dtypes)

    monkeypatch.setattr(
        flashmla_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        flashmla_mod,
        "current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    cls = flashmla_mod.DeepseekV4FlashMLAAttention
    attn = SimpleNamespace(
        compress_ratio=1,
        max_model_len=1024,
        window_size=128,
        max_num_batched_tokens=16,
        topk_indices_buffer=None,
        PREFILL_CHUNK_SIZE=4,
        # _prefill_workspace_reservation_specs needs these too; upstream's
        # inlined version did not.
        head_dim=16,
        n_local_heads=2,
        _reserve_prefill_workspace=cls._reserve_prefill_workspace,
    )
    q = torch.ones((2, 1, 16), dtype=torch.bfloat16)
    output = torch.ones_like(q)

    cls.forward_mqa(attn, q, torch.empty_like(q), torch.arange(2), output)

    torch.testing.assert_close(output, torch.zeros_like(output))
    assert workspace_calls, "SWA-only warmup should still reserve a workspace"


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
