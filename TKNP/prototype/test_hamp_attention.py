# torchrun --nproc_per_node=8 -m HAMP.test_hamp_attention --tensor_model_parallel_size 2 --pipeline_model_parallel_size 2 --batch_size 512 --hidden_size 8192 --check_accuracy --mode scatter_gather --print_ranks

import os
import time
import torch
import torch.distributed as dist
import numpy as np
import csv
from HAMP.hamp_attention import HAMP_Attention, HAMP_Params, ParallelMHA
from flash_attn.utils.generation import InferenceParams
from HybridTensor.utils.profiling import cuda_profiler

from megatron.core import parallel_state #im
import argparse

def initialize_distributed_environment():
    # Set environment variables for NCCL
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

    # Initialize the distributed process group
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set the device based on the rank of the current process
    device = f"cuda:{dist.get_rank()}"
    world_size = dist.get_world_size()

    # if device > num_gpus_per_node, then set device to device % num_gpus_per_node
    num_gpus_per_node = torch.cuda.device_count()
    rank = dist.get_rank()
    if rank >= num_gpus_per_node:
        rank_local = rank % num_gpus_per_node
        device = f"cuda:{rank_local}"

    # Set the current CUDA device to avoid operations being executed on the wrong GPU
    torch.cuda.set_device(device)

    # You can return device, world_size, and any other relevant information
    return device, world_size

def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]

def check_hamp_attention_accuracy(
    args,
    hamp_attention,
    output,
    input_x,
    kv,
    hidden_size,
    nheads,
    num_heads_kv,
    head_dim,
    softmax_scale,
    tp_process_group,
    dp_process_group,
    dp_ranks,
    dp_world_size,
    gpu_rank,
    device,
    dtype,
    batch_size,
):
    """Checks the accuracy of HAMP_Attention against a reference ParallelMHA implementation."""
    # Create a reference attention module
    ref_attention = ParallelMHA(
        embed_dim=hidden_size,
        num_heads=nheads,
        num_heads_kv=num_heads_kv,
        process_group=tp_process_group,
        qkv_proj_bias=False,
        out_proj_bias=False,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=0,
        rotary_emb_dim=128,
        use_flash_attn=True,
        sequence_parallel=False,
        device=device,
        dtype=dtype,
    )
    
    dp_root_rank = dp_ranks[0]

    if gpu_rank == dp_root_rank:
        if hasattr(hamp_attention, 'Wqkv') and hamp_attention.Wqkv is not None:
            ref_attention.Wqkv.weight.data.copy_(hamp_attention.Wqkv.weight.data)
        if hasattr(hamp_attention, 'out_proj') and hamp_attention.out_proj is not None:
            ref_attention.out_proj.weight.data.copy_(hamp_attention.out_proj.weight.data)
    
    # broadcast weights from dp root to all other ranks in the dp group
    dist.broadcast(ref_attention.Wqkv.weight, src=dp_root_rank, group=dp_process_group)
    dist.broadcast(ref_attention.out_proj.weight, src=dp_root_rank, group=dp_process_group)

    # run reference attention
    ref_inference_params = InferenceParams(max_seqlen=args.seq_len + 128, max_batch_size=batch_size)
    
    # Gather the local kv caches from all DP ranks
    gather_list = None
    if gpu_rank == dp_root_rank:
        gather_list = [torch.empty_like(kv) for _ in range(dp_world_size)]
    
    dist.gather(kv, gather_list=gather_list, dst=dp_root_rank, group=dp_process_group)
    
    # On the root rank, create the full KV cache and then broadcast it
    if gpu_rank == dp_root_rank:
        full_kv = torch.cat(gather_list, dim=0)
        _ = _update_kv_cache(full_kv, ref_inference_params, 0)
    else:
        # Other ranks need a placeholder in their inference_params to be overwritten by broadcast
        # We create a cache for the full batch size on all ranks.
        kv_cache = torch.empty(
            batch_size,
            args.seq_len,
            2,
            ref_attention.num_heads_kv_per_rank,
            head_dim,
            dtype=dtype,
            device=device,
        )
        _ = _update_kv_cache(kv_cache, ref_inference_params, 0)

    # Broadcast the updated cache from the root to all other DP ranks
    full_cache_tensor = ref_inference_params.key_value_memory_dict[0]
    dist.broadcast(full_cache_tensor, src=dp_root_rank, group=dp_process_group)
    
    ref_inference_params.seqlen_offset += args.seq_len - 1
    
    ref_output = ref_attention(input_x, inference_params=ref_inference_params)

    if gpu_rank == dp_root_rank:
        print(f"Checking accuracy on rank {gpu_rank}...")
        is_close = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
        print(f" Rank {gpu_rank} : Test passed: {is_close}")
        print(f" Rank {gpu_rank} : HAMP output: {output}")
        print(f" Rank {gpu_rank} : Ref output: {ref_output}")
        if not is_close:
            print("Output mismatch")
            print(f"HAMP output max diff: {torch.max(torch.abs(output - ref_output))}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=8192)
    parser.add_argument("--check_accuracy", action="store_true", default=False)
    parser.add_argument("--print_ranks", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--results_file", type=str, default="hamp_benchmark_results.csv", help="Path to CSV file for logging benchmark results")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes used in the job")
    
    return parser

def log_benchmark_results(results_file, world_size, num_nodes, tensor_model_parallel_size, pipeline_model_parallel_size, 
                         dp_world_size, hidden_size, batch_size, local_batch_size, seq_len, attention_time):
    """Log benchmark results to CSV file. Creates file with headers if it doesn't exist, otherwise appends."""
    fieldnames = ['world_size', 'num_nodes', 'tensor_model_parallel_size', 'pipeline_model_parallel_size', 
                  'dp_world_size', 'hidden_size', 'batch_size', 'local_batch_size', 'seq_len', 'attention_time(ms)']
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write headers only if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow({
            'world_size': world_size,
            'num_nodes': num_nodes,
            'tensor_model_parallel_size': tensor_model_parallel_size,
            'pipeline_model_parallel_size': pipeline_model_parallel_size,
            'dp_world_size': dp_world_size,
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'local_batch_size': local_batch_size,
            'seq_len': seq_len,
            'attention_time(ms)': attention_time
        })

if __name__ == "__main__":
    args = arg_parser().parse_args()
    device, world_size = initialize_distributed_environment()
    num_nodes = args.num_nodes
    dtype =  torch.float16
    order = "tp-pp-dp" # "tp-dp-pp"
    
    parallel_state.initialize_model_parallel(tensor_model_parallel_size =args.tensor_model_parallel_size,
                                            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                                            order=order)
    
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    gpu_rank = dist.get_rank()
    tp_process_group = parallel_state.get_tensor_model_parallel_group()
    pp_process_group = parallel_state.get_pipeline_model_parallel_group()
    dp_process_group = parallel_state.get_data_parallel_group()
    # print(f"Rank: {rank}, Process group: {process_group}")

    batch_size, hidden_size = args.batch_size, args.hidden_size
    max_seqlen = args.seq_len + 128
    head_dim = 128
    nheads = hidden_size // head_dim
    num_heads_kv = 8
    qkv_dim = head_dim * (nheads + 2 * num_heads_kv)
    softmax_scale = 1 / (head_dim ** 0.5)
    dp_ranks = parallel_state._DATA_PARALLEL_GLOBAL_RANKS
    dp_world_size = len(dp_ranks)

    # initialize HAMP Attention
    hamp_attention = HAMP_Attention(
            embed_dim = hidden_size,
            num_heads = nheads,
            num_heads_kv = num_heads_kv,
            process_group = tp_process_group,
            dp_group = dp_process_group,
            dp_ranks = dp_ranks,
            qkv_proj_bias=False,
            out_proj_bias=False,
            softmax_scale=softmax_scale,
            causal=True,
            layer_idx=0,
            rotary_emb_dim=128,
            use_flash_attn=True,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )

    # print ranks and HAMP Attention
    if args.print_ranks:
        time.sleep(gpu_rank * 1)
        print(f"Rank: {gpu_rank}, TP Ranks: {parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS}")
        print(f"Rank: {gpu_rank}, PP Ranks: {parallel_state._PIPELINE_GLOBAL_RANKS}")
        print(f"Rank: {gpu_rank}, DP Ranks: {parallel_state._DATA_PARALLEL_GLOBAL_RANKS}")
        print(f"Rank: {gpu_rank}, MP Ranks: {parallel_state._MODEL_PARALLEL_GLOBAL_RANKS}")
        print(f"Rank: {gpu_rank}, HAMP Attention: {hamp_attention}")

    dist.barrier()

    # simulate inference generation
    input_x = torch.randn(batch_size, 1, hidden_size, device=device, dtype=torch.float16, requires_grad=False)
    local_heads = hamp_attention.num_heads_per_rank if world_size > 1 else hamp_attention.num_heads
    local_kv_heads = hamp_attention.num_heads_kv_per_rank if world_size > 1 else hamp_attention.num_heads_kv
    head_dim = hamp_attention.head_dim
    
    inference_params = InferenceParams(max_seqlen=args.seq_len + 128, max_batch_size=batch_size)
    local_batch_size = batch_size // dp_world_size
    hamp_params = HAMP_Params(
        batch_size=batch_size,
        local_batch_size=local_batch_size,
        seqlen=args.seq_len,
        local_dp_rank=gpu_rank,
        dp_world_size=dp_world_size,
        dp_group=dp_process_group,
        dp_ranks=dp_ranks,
    )

    kv = torch.randn(local_batch_size, args.seq_len, 2, local_kv_heads, head_dim,  device=device, dtype=dtype, requires_grad=False)
    _ = _update_kv_cache(kv, inference_params, 0)
    inference_params.seqlen_offset += args.seq_len - 1

    # run HAMP Attention
    with torch.no_grad():
        output = hamp_attention(input_x, hamp_params=hamp_params, inference_params=inference_params)

        if args.check_accuracy:
            check_hamp_attention_accuracy(
                args,
                hamp_attention,
                output,
                input_x,
                kv,
                hidden_size,
                nheads,
                num_heads_kv,
                head_dim,
                softmax_scale,
                tp_process_group,
                dp_process_group,
                dp_ranks,
                dp_world_size,
                gpu_rank,
                device,
                dtype,
                batch_size,
            )

    
    if args.benchmark:
        # benchmark HAMP Attention
        with torch.no_grad():
            _, attn_time = cuda_profiler(hamp_attention, input_x, hamp_params=hamp_params, inference_params=inference_params, warmup_runs=10, timed_runs=100)

        if gpu_rank == 0:
            # print batch size, seq len
            print(f"Rank {gpu_rank} : Hidden size: {hidden_size}, Batch size: {batch_size}, Seq len: {args.seq_len}")
            print(f"Rank {gpu_rank} : World size: {world_size}, Nodes: {num_nodes}, Tensor model parallel size: {args.tensor_model_parallel_size}, Pipeline model parallel size: {args.pipeline_model_parallel_size}")
            print(f"Rank {gpu_rank} : HAMP Attention time: {attn_time} ms")
            
            # Log results to CSV file
            log_benchmark_results(
                args.results_file,
                world_size,
                num_nodes,
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                dp_world_size,
                hidden_size,
                batch_size,
                local_batch_size,
                args.seq_len,
                attn_time
            )
    
    # clean up
    dist.destroy_process_group()