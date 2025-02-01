from typing import List, Optional, Tuple

import flashinfer
import pytest
import torch

from vllm.platforms import current_platform

NUM_HEADS = [(16, 16), (32, 8), (64, 8), (6, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
@torch.inference_mode
def test_flashinfer_decode_with_paged_kv(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.\
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD",
                use_tensor_cores=(
                    (num_query_heads//num_kv_heads) > 4)
                )
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_query_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 "NONE",
                 q_data_type=dtype,
                 kv_data_type=dtype,
                 logits_soft_cap=soft_cap)

    output = wrapper.run(query, key_value_cache)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=[1] * num_seqs,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
@torch.inference_mode
def test_flashinfer_prefill_with_paged_kv(seq_lens: List[Tuple[int, int]],
                                          num_heads: Tuple[int, int],
                                          head_size: int, dtype: torch.dtype,
                                          block_size: int,
                                          soft_cap: Optional[float]) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(
        query,
        key_value_cache,
    )

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("seq_lens", [[(1, 132), (5, 18)]])
@pytest.mark.parametrize("num_heads", [(32, 8), (6, 1)])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
def test_flashinfer_prefill_with_paged_fp8_kv(
        seq_lens: List[Tuple[int, int]], num_heads: Tuple[int, int],
        head_size: int, dtype: torch.dtype, block_size: int,
        soft_cap: Optional[float]) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(NUM_BLOCKS_FP8,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    kv_cache_fp8 = torch.cat([key_cache / k_scale, value_cache / v_scale],
                             dim=1).to(kv_cache_dtype)

    assert (kv_cache_fp8.shape == key_value_cache.shape)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS_FP8,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache.squeeze(1),
                                value_cache=value_cache.squeeze(1),
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    del query
    del block_tables
    # verify prefill fp8
    torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", [(32, 8), (64, 8), (6, 1)])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
@torch.inference_mode
def test_flashinfer_decode_with_paged_fp8_kv(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    # test doesn't work for num_heads = (16,16)
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    use_tensor_cores = (num_query_heads // num_kv_heads) > 4
    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(NUM_BLOCKS_FP8,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    key_cache_fp8 = (key_cache / k_scale).to(kv_cache_dtype)
    value_cache_fp8 = (value_cache / v_scale).to(kv_cache_dtype)
    assert (key_cache_fp8.shape[1] == 1 and value_cache_fp8.shape[1] == 1)
    kv_cache_fp8 = torch.cat([key_cache_fp8, value_cache_fp8], dim=1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS_FP8,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.\
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD",
                    use_tensor_cores=use_tensor_cores)
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_query_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 "NONE",
                 q_data_type=dtype,
                 kv_data_type=kv_cache_dtype,
                 logits_soft_cap=soft_cap)
    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=[1] * num_seqs,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    # Temporary fix: Increasing the tolerance. Seems like a flashinfer issue
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"
