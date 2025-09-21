import pytest
import torch


def _cuda_sm90_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


@pytest.mark.cuda
def test_sparse_flashmla_imports_and_flags():
    import vllm.attention.ops.flashmla as fm
    # Functions should exist
    assert hasattr(fm, "get_sparse_mla_metadata")
    assert hasattr(fm, "flash_mla_sparse_with_kvcache")
    assert hasattr(fm, "flash_mla_sparse_prefill")
    # Support check should return a (bool, reason)
    ok, reason = fm.is_flashmla_supported()
    assert isinstance(ok, bool)
    assert (reason is None) or isinstance(reason, str)


def test_sparse_flashmla_metadata_smoke():
    import vllm.attention.ops.flashmla as fm
    ok, reason = fm.is_flashmla_supported()
    if not ok or not _cuda_sm90_available():
        pytest.skip(reason or "SM90 not available")

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    q_heads_per_hk = num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_sparse_mla_metadata(cache_seqlens,
                                                     q_seq_per_hk,
                                                     num_heads_k,
                                                     topk,
                                                     q_heads_per_hk)
    assert tile_md.dtype == torch.int32
    assert num_splits.dtype == torch.int32


def test_sparse_flashmla_decode_smoke():
    import vllm.attention.ops.flashmla as fm
    ok, reason = fm.is_flashmla_supported()
    if not ok or not _cuda_sm90_available():
        pytest.skip(reason or "SM90 not available")

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 1
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_sparse_mla_metadata(cache_seqlens,
                                                     q_seq_per_hk,
                                                     num_heads_k,
                                                     topk,
                                                     q_heads_per_hk)

    # Inputs
    q = torch.zeros((batch_size, seqlen_q, num_heads_q, head_dim_k),
                    dtype=torch.bfloat16,
                    device=device)
    k_cache = torch.zeros((1, page_block_size, num_heads_k, bytes_per_token),
                          dtype=torch.uint8,
                          device=device)
    indices = torch.zeros((batch_size, seqlen_q, topk),
                          dtype=torch.int32,
                          device=device)

    out, lse = fm.flash_mla_sparse_with_kvcache(q, k_cache, cache_seqlens,
                                                head_dim_v, tile_md,
                                                num_splits, indices)
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


def test_sparse_flashmla_prefill_smoke():
    import vllm.attention.ops.flashmla as fm
    ok, reason = fm.is_flashmla_supported()
    if not ok or not _cuda_sm90_available():
        pytest.skip(reason or "SM90 not available")

    device = torch.device("cuda")
    s_q = 1
    s_kv = 1
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    q = torch.zeros((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.zeros((s_q, h_kv, topk), dtype=torch.int32, device=device)

    out, max_logits, lse = fm.flash_mla_sparse_prefill(q, kv, indices, 1.0, d_v)
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

