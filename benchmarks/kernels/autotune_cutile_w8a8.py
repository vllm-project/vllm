import argparse
import json
import multiprocessing as mp
import os
import time
import fcntl
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import cuda.tile as ct
from tqdm import tqdm

import vllm.kernels.cutile.cutile_w8a8
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from weight_shapes import WEIGHT_SHAPES
except ImportError:
    print("Error: weight_shapes.py not found. Please ensure it is in the same directory.")
    WEIGHT_SHAPES = {}

mp.set_start_method("spawn", force=True)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}

def get_configs_cutile():
    """Search space for cuTile Tile sizes and L2 grouping."""
    configs = []
    # Common tile sizes for FP8/Hopper/Ada optimization
    for tm in [128]:#[64, 128, 256]:
        for tn in [128]:#[64, 128, 256]:
            for tk in [128]:#[32, 64, 128]:
                for grp in [1]:#[1, 4, 8]:
                    configs.append({
                        "TILE_M": tm, "TILE_N": tn, "TILE_K": tk, "GROUP_SIZE_M": grp
                    })
    return configs

def benchmark_config(A, B, As, Bs, config, out_dtype=torch.float16, num_iters=10):
    M, K = A.shape
    _, N = B.shape
    tm, tn, tk, grp = config["TILE_M"], config["TILE_N"], config["TILE_K"], config["GROUP_SIZE_M"]
    
    C = torch.empty((M, N), dtype=out_dtype, device=A.device)
    grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream

    try:
        from vllm.kernels.cutile.cutile_w8a8 import matmul_kernel
        # Warmup
        for _ in range(3):
            ct.launch(stream, grid, matmul_kernel, (A, B, As, Bs, C, M, N, K, tm, tn, tk, grp))
    except Exception:
        return float('inf') 

    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(num_iters):
        start_event.record()
        ct.launch(stream, grid, matmul_kernel, (A, B, As, Bs, C, M, N, K, tm, tn, tk, grp))
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    
    return (sum(latencies) / num_iters) * 1000 # microseconds

def tune(M, N, K, out_dtype, search_space):
    A = torch.randn((M, K), device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn((K, N), device="cuda").to(torch.float8_e4m3fn)
    
    # Block scales (using vLLM default 128 block size)
    As = torch.rand((M, ct.cdiv(K, 128)), device="cuda", dtype=torch.float32)
    Bs = torch.rand((ct.cdiv(K, 128), ct.cdiv(N, 128)), device="cuda", dtype=torch.float32)

    best_config, best_time = None, float("inf")
    for config in tqdm(search_space, leave=False, desc=f"M={M}"):
        l = benchmark_config(A, B, As, Bs, config, out_dtype)
        if l < best_time:
            best_time, best_config = l, config
    return best_config

def save_to_master_registry(save_path, device_name, n, k, results):
    """Saves results into a single master JSON using file locking."""
    master_file = os.path.join(save_path, f"cutile_w8a8.json")
    
    # Open for reading and writing (a+ creates if missing)
    with open(master_file, "a+") as f:
        # Acquire an exclusive lock (wait if another GPU is writing)
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            # Initialize structure if file was empty
            full_data = json.loads(content) if content else {device_name: {}}
            
            if device_name not in full_data:
                full_data[device_name] = {}

            # Add entries for each M
            for m, config in results.items():
                shape_key = f"mperrank_{m}_n_{n}_k_{k}"
                full_data[device_name][shape_key] = {
                    "block_sizes": [
                        config["TILE_M"],
                        config["TILE_N"],
                        config["TILE_K"],
                        config["GROUP_SIZE_M"]
                    ],
                }
            # Overwrite the file with the new merged data
            f.seek(0)
            f.truncate()
            json.dump(full_data, f, indent=2)
            
            print(f" Registry updated successfully: {master_file}")
            print(f"    [Device: {device_name}] Added N={n}, K={k} for batch sizes: {list(results.keys())}")
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)

def get_weight_shapes(model_names: List[str], tp_sizes: List[int]) -> List[Tuple[int, int]]:
    """
    Extracts unique (N, K) shapes from the WEIGHT_SHAPES registry,
    applying TP splits as necessary.
    """
    unique_shapes = set()
    for model in model_names:
        if model not in WEIGHT_SHAPES:
            print(f"Warning: {model} not found in WEIGHT_SHAPES. Skipping.")
            continue
        
        for shape, tp_split_dim in WEIGHT_SHAPES[model]:
            # shape is [K, N] or [N, K] depending on your registry layout
            # We assume shape is [K, N] based on standard vLLM benchmark utils
            k_raw, n_raw = shape
            
            for tp_size in tp_sizes:
                # Apply the TP split to the dimension specified in the registry
                if tp_split_dim == 0: # Split K
                    unique_shapes.add((n_raw, k_raw // tp_size))
                else: # Split N
                    unique_shapes.add((n_raw // tp_size, k_raw))
                    
    return list(unique_shapes)

def tune_on_gpu(args_dict):
    gpu_id, batch_sizes, shapes, args = args_dict.values()
    torch.cuda.set_device(gpu_id)
    out_dtype = DTYPE_MAP[args.out_dtype]
    search_space = get_configs_cutile()

    for n, k in tqdm(shapes, desc=f"GPU {gpu_id} Shapes"):
        print(f"\n[GPU {gpu_id}] Tuning N={n}, K={k}")
        results = {}
        for m in batch_sizes:
            best_c = tune(m, n, k, out_dtype, search_space)
            if best_c is not None:
                results[m] = best_c
            else:
                print(f"Warning: No valid config found for M={m}, N={n}, K={k}. Skipping.")
        
        if results:
            device = current_platform.get_device_name().lower().replace(" ", "_")
            save_to_master_registry(args.save_path, device, n, k, results)

def main(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: 
        raise RuntimeError("No GPUs found.")

    shapes = get_weight_shapes(args.models, args.tp_sizes)
    if not shapes:
        raise ValueError("No shapes found for models provided.")

    print(f"Found {len(shapes)} unique weight shapes to tune.")

    if num_gpus == 1:
        print("Single GPU detected. Running tuning sequentially...")
        process_args = {
            "gpu_id": 0,
            "batch_sizes": args.batch_sizes, 
            "shapes": shapes, 
            "args": args
        }
        tune_on_gpu(process_args)
    else:
        num_workers = min(num_gpus, len(shapes))
        shapes_per_gpu = [shapes[i::num_workers] for i in range(num_workers)]
        process_args = [
            {
                "gpu_id": i % num_gpus,
                "batch_sizes": args.batch_sizes, 
                "shapes": shapes_per_gpu[i], 
                "args": args
            } for i in range(num_workers)
        ]
        with mp.Pool(num_workers) as pool:
            pool.map(tune_on_gpu, process_args)

if __name__ == "__main__":
    """
    python3 ./benchmarks/kernels/autotune_cutile_w8a8.py    --models\
      meta-llama/Llama-2-7b-hf     --tp-sizes 1     --batch-sizes 1
    """
    import vllm
    vllm_root = os.path.dirname(vllm.__file__)
    default_config_path = os.path.join(
        vllm_root, "model_executor", "layers", "quantization", "utils", "configs"
    )
    DEFAULT_MODELS = list(WEIGHT_SHAPES.keys()) if WEIGHT_SHAPES else []
    parser = FlexibleArgumentParser(description="cuTile Helion Autotuner")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to tune for")
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1], help="TP sizes to simulate")
    parser.add_argument("--batch-sizes", nargs="+", type=int, 
                        default=[1, 16, 32, 64, 128, 256, 512], help="Batch sizes to tune")
    parser.add_argument("--out-dtype", type=str, choices=DTYPE_MAP.keys(), default="float16")
    parser.add_argument("--save-path", type=str, default=default_config_path)
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)