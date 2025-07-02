from ast import Num
import math
import time
import random
import torch
import flashinfer
import pytest

from typing import Optional
from vllm.utils import get_max_shared_memory_bytes
from vllm.platforms import current_platform

# import os
# os.environ['FLASHINFER_WORKSPACE_BASE'] = "/workspace/scratch-pmaj-1/dl-vllm-vllm/new-cubins/"
# os.environ['FLASHINFER_CACHE_DIR'] = "/workspace/scratch-pmaj-1/dl-vllm-vllm/new-cubins/"
# os.environ['FLASHINFER_CUBIN_CHECKSUM_DISABLED'] = "1"
# os.environ['FLASHINFER_LOG_LEVEL'] = "5"
# # os.environ['FLASHINFER_JIT_VERBOSE'] = "1"

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# NUM_HEADS = [(16, 16), (32, 8), (64, 8), (6, 1)]
NUM_HEADS = [(16, 16)]
# HEAD_SIZES = [128, 256]
HEAD_SIZES = [128]
# BLOCK_SIZES = [16, 32]
BLOCK_SIZES = [16]
# DTYPES = [torch.float16, torch.bfloat16]
DTYPES = [torch.float16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
# SOFT_CAPS = [None, 30.0, 50.0]
SOFT_CAPS = [None]
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

@torch.no_grad()
def benchmark_decode(num_seqs, page_size=16, kv_layout="HND", num_kv_heads=8, kv_cache_dtype="auto", head_dim=128, warmup=10, trials=20):
    torch.set_default_device("cuda")
    device = "cuda"
    HEAD_GRP_SIZE = 8
    MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
    NUM_BLOCKS = 5423
    dtype = torch.bfloat16
    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8, device=device)
   
    torch.manual_seed(0)
    print(f"Running benchmark for num_seqs: {num_seqs}")
    # For decode, batch_size is num_decode_token
    num_qo_heads = num_kv_heads * HEAD_GRP_SIZE
    sm_scale  = float(1.0 / (head_dim**0.5))
    q = torch.randn(num_seqs, num_qo_heads, head_dim, device=device, dtype=dtype)
    kv_lens= [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    if num_seqs > 1:
        kv_lens[-1] = MAX_SEQ_LEN
    max_kv_len = max(kv_lens)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int, device=device)
    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
    
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
   
    kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, device=device,dtype=dtype)
    k_scale = v_scale = 1.0

    if kv_cache_dtype.startswith("fp8"):
        kv_cache, _ = to_float8(kv_cache)

    # Benchmark TRT decode
    def trt_decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                q,
                kv_cache,
                workspace_buffer,
                num_qo_heads,
                num_kv_heads,
                sm_scale,
                block_tables,
                kv_lens_tensor,
                page_size,
                MAX_SEQ_LEN,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        
    def time_fn(fn, warmup=10, trials=20, sync=True):
            if sync:
                torch.cuda.synchronize()
                print(f"Finished sync")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            print(f"Warming up")
            for i in range(warmup):
                fn()  # Warmup
                if i % 2 == 0:
                    print(f"Warmup {i} done")
            print(f"Benchmarking")
            for i in range(trials):
                start.record()
                fn()
                end.record()
                if sync:
                    torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms
                if i % 2 == 0:
                    print(f"Trial {i} done")
            return sum(times) / len(times), torch.std(torch.tensor(times))

    # trt_mean, trt_std = time_fn(trt_decode)
    print(f"Finished TRT decode")
    print(f"Running baseline decode")
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + page_size - 1) // page_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % page_size
        if kv_last_page_len == 0:
            kv_last_page_len = page_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    print(f"Creating baseline wrapper")
    wrapper = flashinfer.\
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD",
                use_tensor_cores=(
                    (num_qo_heads//num_kv_heads) > 4)
                )
    print(f"Planning baseline wrapper")
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_qo_heads,
                 num_kv_heads,
                 head_dim,
                 page_size,
                 "NONE",
                 q_data_type=dtype,
                 kv_data_type=torch.float8_e4m3fn if kv_cache_dtype.startswith("fp8") else dtype)

    def baseline_decode():
        return wrapper.run(q, kv_cache, sm_scale, k_scale, v_scale)
    print(f"Running baseline decode")
    output =  baseline_decode()
    torch.cuda.synchronize()
    print(f"Finished baseline decode: {output}")
    print(f"Benchmarking baseline decode")
    # baseline_mean, baseline_std = time_fn(baseline_decode, sync=False)
    print(f"Finished baseline decode")
    print(f"num_seqs: {num_seqs}, trt_decode_ms_avg: {trt_mean}, trt_decode_ms_std: {trt_std.item()}, baseline_decode_ms_avg: {baseline_mean}, baseline_decode_ms_std: {baseline_std.item()}")

@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@torch.inference_mode
def test_flashinfer_trtllm_decode_with_baseline(
    kv_lens: list[int],
    num_heads: tuple[int, int],
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
                                  num_kv_heads,
                                  block_size,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    k_scale = v_scale = 1.0
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
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND",
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

    output = wrapper.run(query, key_value_cache, scale)

    # TRTLLM Decode
    max_kv_len = max(kv_lens)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int, device=query.device)
    # key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    # trtllm_kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, page_size, head_dim)
    # trtllm_kv_cache = key_value_cache.permute(0,1,3,2,4).contiguous()
    output_trtllm = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query,
                key_value_cache,
                workspace_buffer,
                num_query_heads,
                num_kv_heads,
                scale,
                block_tables,
                kv_lens_tensor,
                block_size,
                max_kv_len,
                "auto",
                k_scale,
                v_scale,
            )
    print(f"output: {output.shape}, output_trtllm: {output_trtllm.shape}")
    print(f"output: {output}, output_trtllm: {output_trtllm}")
    
    torch.testing.assert_close(output, output_trtllm, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - output_trtllm))}"
  
    
       



if __name__ == "__main__":
    # num_seqs = [1, 2, 5, 128, 187, 200,]# 256, 512, 1024, 2048, 4096]
    num_seqs = [1, 2, 5,]
    for num_seqs in num_seqs:
        benchmark_decode(num_seqs)