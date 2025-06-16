from ast import Num
import math
import time
import random
import torch
import flashinfer
import pandas as pd

from vllm.utils import get_max_shared_memory_bytes
FLOAT32_BYTES = torch.finfo(torch.float).bits // 8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

@torch.no_grad()
def benchmark_decode(num_seqs, page_size=16, kv_layout="HND", num_kv_heads=8, kv_cache_dtype="fp8", head_dim=128, warmup=10, trials=20):
    torch.set_default_device("cuda")
    device = "cuda"
    HEAD_GRP_SIZE = 8
    MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
    NUM_BLOCKS = 5423
    dtype = torch.bfloat16
    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8, device=device)
   
    torch.manual_seed(0)
    print(f"num_seqs: {num_seqs}")
    # For decode, batch_size is num_decode_token
    num_qo_heads = num_kv_heads * HEAD_GRP_SIZE
    scale  = float(1.0 / (head_dim**0.5))
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
    kv_cache = torch.randn(size=kv_cache_shape, device=device).half()
    k_scale = v_scale = 1.0

    if kv_cache_dtype.startswith("fp8"):
        kv_cache, _ = to_float8(kv_cache)

    # Benchmark TRT decode
    def trt_decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                q.contiguous(),
                kv_cache,
                workspace_buffer,
                num_qo_heads,
                num_kv_heads,
                scale,
                block_tables,
                kv_lens_tensor,
                page_size,
                MAX_SEQ_LEN,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        
    def time_fn(fn):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(warmup): fn()  # Warmup
            for _ in range(trials):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms
            return sum(times) / len(times), torch.std(torch.tensor(times))

    trt_mean, trt_std = time_fn(trt_decode)
    print(f"num_seqs: {num_seqs}, trt_decode_ms_avg: {trt_mean}, trt_decode_ms_std: {trt_std.item()}")



if __name__ == "__main__":

    # num_seqs = [1, 2, 5, 128,  256, 512, 1024, 2048, 4096]
    num_seqs = [17]

    for num_seqs in num_seqs:
        benchmark_decode(num_seqs)