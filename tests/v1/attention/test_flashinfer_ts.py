# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLASHINFER_TS: the non-causal draft-block decode path.

Covers the two things that are easy to get wrong and silently produce plausible
output: the K/V plane layout the kernel validates, and the page-aligned window
restriction that stands in for a window the kernel cannot take with a dense
mask.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flashinfer_ts import (
    FlashInferTSMetadataBuilder,
)

HEAD_DIM = 128
PAGE = 128
WINDOW = 4096


def _sm100_family() -> bool:
    cap = current_platform.get_device_capability()
    return cap is not None and (cap.major, cap.minor) in ((10, 0), (10, 3), (10, 7))


requires_kernel = pytest.mark.skipif(
    not current_platform.is_cuda() or not _sm100_family(),
    reason="prims_ts decode runs on SM100a/SM103a/SM107a only",
)


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_reads_vllms_existing_interleaved_page(num_kv_heads):
    """No layout change: K/V are the halves of vLLM's [pages, Hkv, page, 2*D]
    page, whose token stride is 2*D. The kernel takes its TMA strides from the
    tensor, so this must stay a pure view with head_dim contiguous.
    """
    page = torch.zeros(8, num_kv_heads, PAGE, 2 * HEAD_DIM, dtype=torch.uint8)
    k, v = page[..., :HEAD_DIM], page[..., HEAD_DIM:]

    assert k.shape == (8, num_kv_heads, PAGE, HEAD_DIM)
    assert k.stride(-1) == 1 and v.stride(-1) == 1  # the one hard requirement
    assert k.stride(2) == 2 * HEAD_DIM  # interleaved, not compact
    assert k.data_ptr() == page.data_ptr()  # views, never a repack
    assert v.data_ptr() == page.data_ptr() + HEAD_DIM


@pytest.mark.parametrize("query_len", [1, 3, 7])
def test_window_restriction_is_a_superset_of_the_true_window(query_len):
    """The restricted page table must never drop a token the symmetric window
    can reach, and must overshoot by less than one page.
    """
    builder = FlashInferTSMetadataBuilder.__new__(FlashInferTSMetadataBuilder)
    builder.sliding_window = WINDOW
    builder.page_size = PAGE
    builder.query_len = query_len
    # The builder writes metadata into persistent buffers so a replayed CUDA
    # graph keeps reading the same addresses; the window math is exercised
    # through them, not around them.
    pages = builder._window_pages(query_len)
    builder._buf_seq_lens = torch.zeros(6, dtype=torch.int32)
    builder._buf_block_table = torch.zeros((6, pages), dtype=torch.int32)

    seq_lens = torch.tensor(
        [1, PAGE, WINDOW - 1, WINDOW + query_len, 8192, 120_000], dtype=torch.int32
    )
    block_table = torch.arange(
        seq_lens.numel() * (int(seq_lens.max()) // PAGE + 1), dtype=torch.int32
    ).view(seq_lens.numel(), -1)

    new_lens, new_bt = FlashInferTSMetadataBuilder._restrict_to_window(
        builder, seq_lens, block_table, query_len
    )

    span = WINDOW + query_len - 1
    reachable = torch.minimum(seq_lens, torch.full_like(seq_lens, span))
    assert torch.all(new_lens >= reachable), "window restriction dropped live tokens"
    assert int((new_lens - reachable).max()) < PAGE, "overshoot exceeds one page"
    # The kept pages must be the tail of each row, so the last token of the
    # sequence is still addressable.
    last_page = (seq_lens.long() - 1) // PAGE
    assert torch.all(new_bt.long().max(dim=1).values >= last_page)


def test_impl_accepts_the_arguments_the_attention_layer_passes():
    """Attention() calls impl_cls positionally and forwards extra_impl_args
    (``sinks`` among them); a signature that only matches AttentionImpl's
    abstract __init__ raises TypeError at engine init, not at import.
    """
    from vllm.v1.attention.backends.flashinfer_ts import FlashInferTSImpl

    impl = FlashInferTSImpl(
        16,
        HEAD_DIM,
        HEAD_DIM**-0.5,
        1,
        None,
        WINDOW,
        "fp8",
        None,
        AttentionType.DECODER,
        None,
        sinks=None,
    )
    # FP8 KV reads an FP8 query; the layer's own quant is skipped via
    # query_pre_quantized because the fused QK-norm/RoPE kernel emits it.
    assert impl.supports_quant_query_input

    with pytest.raises(NotImplementedError):
        FlashInferTSImpl(
            16,
            HEAD_DIM,
            HEAD_DIM**-0.5,
            1,
            None,
            WINDOW,
            "fp8",
            None,
            AttentionType.DECODER,
            None,
            sinks=torch.zeros(16),
        )


@requires_kernel
def test_dense_plan_matches_flash_attention_non_causal():
    """A dense prims_ts plan is the non-causal draft block FA4 produces."""
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    pytest.importorskip("flashinfer.attention.prims_ts")
    from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper

    torch.manual_seed(0)
    b, sq, hq, hkv, kv = 2, 3, 16, 1, PAGE
    dev = torch.device("cuda")
    scale = HEAD_DIM**-0.5
    pages = b * (kv // PAGE) + 1

    page = torch.randn(pages, 2 * hkv, PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    k_plane, v_plane = page[:, :hkv].contiguous(), page[:, hkv:].contiguous()
    interleaved = torch.cat([k_plane, v_plane], dim=-1).contiguous()
    k_fa, v_fa = interleaved.transpose(1, 2).split(HEAD_DIM, dim=-1)

    q = torch.randn(b * sq, hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = torch.arange(1, pages, dtype=torch.int32, device=dev).view(b, -1).contiguous()
    lens = torch.full((b,), kv, dtype=torch.int32, device=dev)
    cu_q = torch.arange(0, (b + 1) * sq, sq, dtype=torch.int32, device=dev)

    ref = torch.empty(b * sq, hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    _flash_attn_fwd(
        q,
        k_fa,
        v_fa,
        cu_seqlens_q=cu_q,
        seqused_k=lens,
        max_seqlen_q=sq,
        max_seqlen_k=kv,
        page_table=bt,
        softmax_scale=scale,
        causal=False,
        window_size_left=WINDOW - 1,
        window_size_right=WINDOW - 1,
        num_splits=0,
        out=ref,
    )

    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    wrapper.plan(
        device=dev,
        batch_size=b,
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim=HEAD_DIM,
        page_size=PAGE,
        max_kv_len=kv,
        max_seq_len_q=sq,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        mask_type="dense",
        window_left=-1,
    )
    out = torch.empty(b, sq, hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    wrapper.run(
        q=q.view(b, sq, hq, HEAD_DIM),
        paged_kv_cache=(k_plane, v_plane),
        seq_lens=lens,
        block_tables=bt,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        out=out,
    )

    # Within one bf16 ulp: the kernels round differently on some SM100 parts.
    torch.testing.assert_close(
        out.reshape(b * sq, hq, HEAD_DIM), ref, rtol=1e-2, atol=2e-3
    )


def test_metadata_tensors_keep_their_addresses_across_builds():
    """A captured CUDA graph replays the ADDRESS of seq_lens and the page table.

    Returning freshly allocated tensors per step leaves a replayed graph reading
    memory the allocator has since handed to someone else, which shows up as an
    engine that pins every GPU and never retires a step.
    """
    builder = FlashInferTSMetadataBuilder.__new__(FlashInferTSMetadataBuilder)
    builder.sliding_window = WINDOW
    builder.page_size = PAGE
    builder.query_len = 4
    pages = builder._window_pages(4)
    builder._buf_seq_lens = torch.zeros(4, dtype=torch.int32)
    builder._buf_block_table = torch.zeros((4, pages), dtype=torch.int32)

    block_table = torch.arange(4 * 1024, dtype=torch.int32).view(4, -1)
    ptrs = set()
    for step in range(3):
        seq_lens = torch.tensor([512, 8192, 40_000 + step, 120_000], dtype=torch.int32)
        lens, table = FlashInferTSMetadataBuilder._restrict_to_window(
            builder, seq_lens, block_table, 4
        )
        ptrs.add((lens.data_ptr(), table.data_ptr(), tuple(table.shape)))

    assert len(ptrs) == 1, f"metadata moved between builds: {ptrs}"
