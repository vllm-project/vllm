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


"""
decode_query.shape
torch.Size([2, 64, 128])
kv_cache.shape
torch.Size([20111, 2, 16, 8, 128])
kv_cache.permute(0,1,3,2,4).contiguous().shape
torch.Size([20111, 2, 8, 16, 128])
attn_metadata.workspace_buffer
tensor([  0,   0,   0,  ...,  61, 157,  62], device='cuda:0',
       dtype=torch.uint8)
attn_metadata.workspace_buffer.shape
torch.Size([268435456])
self.num_heads
64
self.num_kv_heads
8
self.scale
0.08838834764831845
self.scale.dtype
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AttributeError: 'float' object has no attribute 'dtype'
attn_metadata.block_table_tensor[:num_decode_tokens].shape
torch.Size([2, 8192])
attn_metadata.block_table_tensor[:num_decode_tokens]
tensor([[1, 0, 0,  ..., 0, 0, 0],
        [2, 0, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.int32)
attn_metadata.seq_lens[:num_decode_tokens]
tensor([7, 9], device='cuda:0', dtype=torch.int32)
attn_metadata.page_size
16
attn_metadata.max_seq_len
9
"auto"
'auto'
layer._k_scale_float
1.0
layer._v_scale_float
1.0
output[:num_decode_tokens] = trtllm_batch_decode_with_kv_cache(
                query=decode_query.contiguous(),
                kv_cache=kv_cache.permute(0,1,3,2,4).contiguous(),
                workspace_buffer=attn_metadata.workspace_buffer,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=attn_metadata.block_table_tensor[:num_decode_tokens],
                seq_lens=attn_metadata.seq_lens[:num_decode_tokens],
                block_size=attn_metadata.page_size,
                max_seq_len=attn_metadata.max_seq_len,
                kv_cache_dtype="auto")
Traceback (most recent call last):
  File "<string>", line 1, in <module>
TypeError: trtllm_batch_decode_with_kv_cache() missing 2 required positional arguments: 'k_scale' and 'v_scale'
output[:num_decode_tokens] = trtllm_batch_decode_with_kv_cache(
                query=decode_query.contiguous(),
                kv_cache=kv_cache.permute(0,1,3,2,4).contiguous(),
                workspace_buffer=attn_metadata.workspace_buffer,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=attn_metadata.block_table_tensor[:num_decode_tokens],
                seq_lens=attn_metadata.seq_lens[:num_decode_tokens],
                block_size=attn_metadata.page_size,
                max_seq_len=attn_metadata.max_seq_len,
                kv_cache_dtype="auto",
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
            )
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.12/dist-packages/flashinfer/decode.py", line 1706, in trtllm_batch_decode_with_kv_cache
    run_func(
  File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Trtllm-gen kernels not found: qkvLayout=2, maskType=0, kernelType=2, tileScheduler=0, multiCtasKvMode=1, headDimPerCtaV=128, headDimQk=128, headDimV=128, tileSizeKv=128, numTokensPerPage=16, maxNumHeadsQPerKvInCta=8, reuseSmemKForV=0, uses2CtaMma=0

"""

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
    output_trtllm = trt_decode()
    print(output_trtllm.shape)
    # print(output_trtllm)
  
        # Prepare wrapper
    #    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    #    kv_indptr = torch.cat([
    #        torch.tensor([0], device=device),
    #        torch.cumsum(seq_lens_tensor // page_size, dim=0),
    #    ]).int()
    #    kv_indices = torch.arange(num_blocks, device=device).int()
    #    kv_last_page_len = torch.full((batch_size,), page_size, device=device).int()

    #    wrapper.plan(
    #        kv_indptr,
    #        kv_indices,
    #        kv_last_page_len,
    #        num_qo_heads,
    #        num_kv_heads,
    #        head_dim,
    #        page_size,
    #        pos_encoding_mode="NONE",
    #        data_type=torch.float16 if kv_cache_dtype == "auto" else torch.float8_e4m3fn,
    #        q_data_type=torch.float16,
    #    )

    #    def wrapper_decode():
    #        return wrapper.run(q.contiguous(), kv_cache)
    #    print("Running wrapper decode")
    #    out_decode_wrapper = wrapper_decode()
    #    print(out_decode_wrapper.shape)
    #    print(out_decode_wrapper)
    #    
    #    
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
    #    wrap_mean, wrap_std = time_fn(wrapper_decode)
    #    print(f"batch_size: {batch_size}, trt_decode_ms_avg: {trt_mean}, trt_decode_ms_std: {trt_std.item()}, wrapper_decode_ms_avg: {wrap_mean}, wrapper_decode_ms_std: {wrap_std.item()}")

   #     results.append({
   #         "batch_size": batch_size,
   #         "trt_decode_ms_avg": trt_mean,
   #         "trt_decode_ms_std": trt_std.item(),
   #         "wrapper_decode_ms_avg": wrap_mean,
   #         "wrapper_decode_ms_std": wrap_std.item(),
   #     })

    # return pd.DataFrame(results)


if __name__ == "__main__":
    import os
    os.environ['FLASHINFER_LOG_LEVEL'] = "5"
    os.environ['FLASHINFER_JIT_VERBOSE'] = "1"
    # num_seqs = [1,2,5] 
    num_seqs = [1, 2, 5, 128, 256, 512, 1024, 2048, 4096]

    for num_seqs in num_seqs:
        benchmark_decode(num_seqs)