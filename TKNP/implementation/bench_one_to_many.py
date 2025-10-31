#!/usr/bin/env python3

# torchrun --nproc_per_node=2 bench_one_to_many.py --feature-dim 4096 --base-size 32768 --jitter 0.3 --iters 20 --warmup 5
import argparse
import os
import time
from statistics import mean, median

import torch
import torch.distributed as dist

# --------------- Helpers ---------------

# def ddp_init(backend: str):
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))

#     use_cuda = torch.cuda.is_available() and backend == "nccl"
#     if use_cuda:
#         torch.cuda.set_device(local_rank)

#     dist.init_process_group(backend=backend)
#     return rank, world_size, use_cuda, local_rank

def ddp_init(backend: str):
    import os, torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    use_cuda = (backend == "nccl") and torch.cuda.is_available()
    if use_cuda:
        # Bind process to its GPU BEFORE init
        torch.cuda.set_device(local_rank)

    # Keep it simple & compatible: don't pass device_id
    dist.init_process_group(backend=backend)

    # Fail-fast sanity collective
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    t = torch.ones(1, device=device)
    dist.all_reduce(t)  # should succeed quickly

    return rank, world_size, use_cuda, local_rank

def barrier_sync(device_is_cuda: bool):
    # Make sure all device work is done before the barrier
    if device_is_cuda:
        torch.cuda.synchronize()
    dist.barrier()
    if device_is_cuda:
        torch.cuda.synchronize()


def sizes_with_variation(world_size, base_size, jitter_frac, min_size=0, seed=None):
    """
    Produce variable per-rank sizes for a single iteration.
    Rank 0 is root (sender), so sizes returned are what each rank receives.
    All ranks must generate the same sizes, so we use a shared seed.
    """
    import random
    if seed is not None:
        random.seed(seed)
    sizes = []
    for r in range(world_size):
        # Receiver size (root can send to itself as 0 by default)
        if r == 0:
            sizes.append(0)
            continue
        jitter = int(base_size * (random.uniform(-jitter_frac, jitter_frac)))
        sz = max(min_size, base_size + jitter)
        sizes.append(sz)
    return sizes


def tensor_on_device(shape, dtype, device, fill=None):
    t = torch.empty(shape, dtype=dtype, device=device)
    if fill is not None:
        t.fill_(fill)
    else:
        # Make it "touched" so kernels actually do the copies
        t.normal_()
    return t


def ms(t0, t1):  # seconds -> milliseconds
    return (t1 - t0) * 1000.0


def reduce_max_scalar(x: float, device):
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def summarize(name, times_ms, rank):
    if rank == 0 and times_ms:
        p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
        print(f"\n[{name}] results over {len(times_ms)} iters:")
        print(f"  mean  : {mean(times_ms):.3f} ms")
        print(f"  median: {median(times_ms):.3f} ms")
        print(f"  p95   : {p95:.3f} ms")
        print(f"  max   : {max(times_ms):.3f} ms")


# --------------- Approaches ---------------

def bench_all_to_all_single(
    rank, world_size, device, feat_dim, iters, warmup, base_size, jitter, dtype, verify_seed=None
):
    """
    Root has per-dst variable sizes. Others receive only from root.
    Implemented by calling all_to_all_single with:
      - Root: input_split_sizes = recv_sizes, output_split_sizes = only local recv size
      - Others: input_split_sizes = all zeros, output_split_sizes = [0.., recv_from_root, ..0]
    """
    name = "all_to_all_single"
    times = []
    printed_sizes = False
    verification_result = None

    for rep in range(iters + warmup):
        recv_sizes = sizes_with_variation(world_size, base_size, jitter, min_size=0, seed=rep)

        # Prepare send/recv buffers
        if rank == 0:
            total_send = sum(recv_sizes)
            send = tensor_on_device((total_send, feat_dim), dtype, device)
            # For verification: fill with known pattern
            if verify_seed is not None and rep == 0:
                send.fill_(float(rank + 1))
            # Split by ranks in order: [r0(=0), r1, r2, ...]
            input_split_sizes = recv_sizes[:]  # root sends to each rank its size
            # Only take what this rank receives (root receives its own chunk here, which is 0)
            output_split_sizes = [0] * world_size
            output_split_sizes[0] = recv_sizes[0]  # usually zero
            local_recv = tensor_on_device((recv_sizes[rank], feat_dim), dtype, device)
        else:
            # Non-root sends nothing
            send = tensor_on_device((0, feat_dim), dtype, device)
            input_split_sizes = [0] * world_size
            # Only receive from root (rank 0)
            output_split_sizes = [0] * world_size
            output_split_sizes[0] = recv_sizes[rank]
            local_recv = tensor_on_device((recv_sizes[rank], feat_dim), dtype, device)

        if not printed_sizes:
            if rank == 0:
                print(f"\n[{name}] Tensor sizes:")
                print(f"  Rank {rank} send tensor: shape={send.shape}, size={send.numel()} elements, memory={send.element_size() * send.numel() / 1024**2:.2f} MB")
                print(f"  Rank {rank} recv tensor: shape={local_recv.shape}, size={local_recv.numel()} elements, memory={local_recv.element_size() * local_recv.numel() / 1024**2:.2f} MB")
                print(f"  Per-rank row breakdown: {recv_sizes}")
            else:
                print(f"  Rank {rank} send tensor: shape={send.shape}, size={send.numel()} elements, memory={send.element_size() * send.numel() / 1024**2:.2f} MB")
                print(f"  Rank {rank} recv tensor: shape={local_recv.shape}, size={local_recv.numel()} elements, memory={local_recv.element_size() * local_recv.numel() / 1024**2:.2f} MB")
            printed_sizes = True

        barrier_sync(device.type == "cuda")

        t0 = time.perf_counter()
        dist.all_to_all_single(
            local_recv,
            send,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
        # Ensure device completion
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Use max across ranks as total latency reference
        elapsed = ms(t0, t1)
        max_elapsed = reduce_max_scalar(elapsed, device)
        barrier_sync(device.type == "cuda")

        if rep >= warmup and rank == 0:
            times.append(max_elapsed)
        
        # Store first iteration result for verification
        if verify_seed is not None and rep == 0:
            verification_result = local_recv.clone()

    summarize(name, times, rank)
    return name, times, verification_result


def bench_p2p_isend_irecv(
    rank, world_size, device, feat_dim, iters, warmup, base_size, jitter, dtype, verify_seed=None
):
    """
    Root -> others using sizes broadcast (vector of recv sizes).
    Then root posts isend to each dst; non-root posts irecv.
    """
    name = "p2p_isend_irecv"
    times = []
    printed_sizes = False
    verification_result = None

    for rep in range(iters + warmup):
        sizes = sizes_with_variation(world_size, base_size, jitter, min_size=0, seed=rep)
        sizes_t = torch.tensor(sizes, dtype=torch.int64, device=device)

        # Everyone learns sizes (1 tiny bcast)
        barrier_sync(device.type == "cuda")
        dist.broadcast(sizes_t, src=0)
        barrier_sync(device.type == "cuda")

        # Allocate buffers
        recv_n = int(sizes_t[rank].item())
        if rank == 0:
            # Make contiguous slices for each dst
            sends = []
            cursor = 0
            total = sum(sizes)
            big = tensor_on_device((total, feat_dim), dtype, device)
            # For verification: fill with known pattern
            if verify_seed is not None and rep == 0:
                big.fill_(float(rank + 1))
            for dst in range(world_size):
                n = sizes[dst]
                if n > 0:
                    sends.append(big[cursor: cursor + n])
                else:
                    sends.append(tensor_on_device((0, feat_dim), dtype, device))
                cursor += n
        else:
            recv = tensor_on_device((recv_n, feat_dim), dtype, device)

        if not printed_sizes:
            if rank == 0:
                print(f"\n[{name}] Tensor sizes:")
                print(f"  Rank {rank} send tensor (big): shape={big.shape}, size={big.numel()} elements, memory={big.element_size() * big.numel() / 1024**2:.2f} MB")
                print(f"  Per-rank row breakdown: {sizes}")
                for dst in range(1, world_size):
                    print(f"    -> Rank {dst}: {sends[dst].shape}, {sends[dst].numel()} elements")
            else:
                print(f"  Rank {rank} recv tensor: shape={recv.shape}, size={recv.numel()} elements, memory={recv.element_size() * recv.numel() / 1024**2:.2f} MB")
            printed_sizes = True

        barrier_sync(device.type == "cuda")

        t0 = time.perf_counter()

        reqs = []
        if rank == 0:
            for dst in range(1, world_size):
                n = sizes[dst]
                if n > 0:
                    reqs.append(dist.isend(tensor=sends[dst].view(-1), dst=dst))
        else:
            if recv_n > 0:
                reqs.append(dist.irecv(tensor=recv.view(-1), src=0))

        for r in reqs:
            r.wait()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = ms(t0, t1)
        max_elapsed = reduce_max_scalar(elapsed, device)
        barrier_sync(device.type == "cuda")

        if rep >= warmup and rank == 0:
            times.append(max_elapsed)
        
        # Store first iteration result for verification
        if verify_seed is not None and rep == 0:
            if rank != 0:
                verification_result = recv.clone()
            else:
                verification_result = None

    summarize(name, times, rank)
    return name, times, verification_result


def bench_pad_max_scatter(
    rank, world_size, device, feat_dim, iters, warmup, base_size, jitter, dtype, verify_seed=None
):
    """
    Pad each destination slice on root to a common max size, then use dist.scatter.
    NOTE: This is "unfair" if sizes vary a lot, but it's a realistic alternative
          people use to exploit tuned collectives.
    """
    name = "pad_to_max_scatter"
    times = []
    printed_sizes = False
    verification_result = None

    for rep in range(iters + warmup):
        sizes = sizes_with_variation(world_size, base_size, jitter, min_size=0, seed=rep)
        max_sz = max([0] + sizes)  # allow all zeros edge case
        pad = max_sz

        # Build per-rank tensors (padded to pad rows)
        if rank == 0:
            chunks = []
            for dst in range(world_size):
                n = sizes[dst]
                if n > 0:
                    chunk = tensor_on_device((pad, feat_dim), dtype, device)
                    # For verification: fill with known pattern
                    if verify_seed is not None and rep == 0:
                        chunk.fill_(float(rank + 1))
                    # Only first n rows relevant; rest are pad
                    # (We fill to avoid lazy allocation)
                else:
                    chunk = tensor_on_device((pad, feat_dim), dtype, device).zero_()
                chunks.append(chunk)
            recv = tensor_on_device((pad, feat_dim), dtype, device)
        else:
            recv = tensor_on_device((pad, feat_dim), dtype, device)

        if not printed_sizes:
            if rank == 0:
                print(f"\n[{name}] Tensor sizes:")
                print(f"  Rank {rank} recv tensor: shape={recv.shape}, size={recv.numel()} elements, memory={recv.element_size() * recv.numel() / 1024**2:.2f} MB")
                print(f"  Actual per-rank row sizes: {sizes}")
                print(f"  Padded to: {pad} rows per rank")
                for dst, chunk in enumerate(chunks):
                    print(f"    Chunk {dst}: shape={chunk.shape}, size={chunk.numel()} elements")
            else:
                print(f"  Rank {rank} recv tensor: shape={recv.shape}, size={recv.numel()} elements, memory={recv.element_size() * recv.numel() / 1024**2:.2f} MB")
                print(f"    Actual rows: {sizes[rank]}, Padded rows: {pad}")
            printed_sizes = True

        barrier_sync(device.type == "cuda")

        t0 = time.perf_counter()
        if rank == 0:
            dist.scatter(recv, scatter_list=chunks, src=0)
        else:
            dist.scatter(recv, scatter_list=[], src=0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = ms(t0, t1)
        max_elapsed = reduce_max_scalar(elapsed, device)
        barrier_sync(device.type == "cuda")

        if rep >= warmup and rank == 0:
            times.append(max_elapsed)
        
        # Store first iteration result for verification
        if verify_seed is not None and rep == 0:
            verification_result = recv.clone()

    summarize(name, times, rank)
    return name, times, verification_result


# --------------- Verification ---------------

def verify_tensors(results, rank, world_size, device):
    """
    Verify that all three approaches produced the same tensors.
    results: list of (name, times, tensor) tuples
    """
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    if rank == 0:
        print(f"Rank {rank}: Root doesn't receive data, skipping verification")
        # Still participate in global check
        match_flag = torch.tensor([1.0], device=device)
        dist.all_reduce(match_flag, op=dist.ReduceOp.MIN)
        if match_flag.item() > 0.5:
            print("\n✓ ALL RANKS: All methods produced identical tensors!")
        else:
            print("\n✗ VERIFICATION FAILED: Some methods produced different tensors")
        print("="*60)
        return
    
    # Extract tensors from results
    tensors = {}
    for name, times, tensor in results:
        if tensor is not None:
            tensors[name] = tensor
    
    if len(tensors) < 2:
        print(f"Rank {rank}: Not enough tensors to compare")
        # Report failure to global check
        match_flag = torch.tensor([0.0], device=device)
        dist.all_reduce(match_flag, op=dist.ReduceOp.MIN)
        return
    
    # Compare all pairs
    names = list(tensors.keys())
    all_match = True
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            t1, t2 = tensors[name1], tensors[name2]
            
            # Check shapes
            if t1.shape != t2.shape:
                print(f"Rank {rank}: Shape mismatch between {name1} and {name2}")
                print(f"  {name1}: {t1.shape}, {name2}: {t2.shape}")
                all_match = False
                continue
            
            # Check values
            if torch.allclose(t1, t2, rtol=1e-5, atol=1e-8):
                print(f"Rank {rank}: ✓ {name1} matches {name2}")
            else:
                print(f"Rank {rank}: ✗ {name1} does NOT match {name2}")
                max_diff = torch.max(torch.abs(t1 - t2)).item()
                mean_diff = torch.mean(torch.abs(t1 - t2)).item()
                print(f"  Max difference: {max_diff:.6e}, Mean difference: {mean_diff:.6e}")
                all_match = False
    
    # Global check across all ranks
    match_flag = torch.tensor([1.0 if all_match else 0.0], device=device)
    dist.all_reduce(match_flag, op=dist.ReduceOp.MIN)


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="One-to-many latency benchmark (root -> others)")
    parser.add_argument("--backend", type=str, default=("nccl" if torch.cuda.is_available() else "gloo"),
                        choices=["nccl", "gloo"], help="Process group backend")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"], help="Tensor dtype")
    parser.add_argument("--feature-dim", type=int, default=4096, help="Feature dimension per token/row")
    parser.add_argument("--base-size", type=int, default=32768, help="Baseline rows per destination")
    parser.add_argument("--jitter", type=float, default=0.3, help="Uniform +/- fraction for per-rank size")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed)")
    parser.add_argument("--verify", action="store_true", help="Verify that all methods produce identical tensors")
    args = parser.parse_args()

    rank, world_size, use_cuda, local_rank = ddp_init(args.backend)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if rank == 0:
        print(f"Backend={args.backend}  World={world_size}  Device={device}  DType={dtype}")
        print(f"feature_dim={args.feature_dim}  base_size={args.base_size}  jitter={args.jitter}")
        print(f"iters={args.iters} warmup={args.warmup}")
        if args.verify:
            print("Verification mode: ON")

    # For verification, use a fixed seed so all methods get the same data
    verify_seed = 42 if args.verify else None

    # Benchmarks
    results = []
    
    result1 = bench_all_to_all_single(rank, world_size, device, args.feature_dim,
                            args.iters, args.warmup, args.base_size, args.jitter, dtype, verify_seed)
    results.append(result1)

    result2 = bench_p2p_isend_irecv(rank, world_size, device, args.feature_dim,
                          args.iters, args.warmup, args.base_size, args.jitter, dtype, verify_seed)
    results.append(result2)

    result3 = bench_pad_max_scatter(rank, world_size, device, args.feature_dim,
                          args.iters, args.warmup, args.base_size, args.jitter, dtype, verify_seed)
    results.append(result3)

    # Verify tensors if requested
    if args.verify:
        dist.barrier()
        verify_tensors(results, rank, world_size, device)

    dist.barrier()
    if rank == 0:
        print("\nDone.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()