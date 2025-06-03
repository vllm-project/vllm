import subprocess
import torch
import os
import glob
from typing import Dict, List, Tuple, Any
import numpy as np

def run_cache_layer(backend: str) -> str:
    """Run cache_layer.py with specified backend and return output directory."""
    cmd = ["python", "cache_layer.py", 
           "--backend", backend,
           "--model", "meta-llama/Llama-3.1-8B"]
    
    print(f"\nRunning cache_layer with backend: {backend}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running cache_layer with backend {backend}:")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        raise  # Re-raise the exception after printing
    return f"{backend}_attn_captures"

def load_tensors_from_pass(pass_dir: str) -> Dict[str, Any]:
    """Load all tensors from a specific pass directory."""
    tensors = {}
    
    # Load basic tensors
    for tensor_name in ['query', 'key', 'value']:
        tensor_path = os.path.join(pass_dir, f"{tensor_name}.pt")
        if os.path.exists(tensor_path):
            tensors[tensor_name] = torch.load(tensor_path)
    
    # Load KV cache
    kv_cache_path = os.path.join(pass_dir, "kv_cache.pt")
    if os.path.exists(kv_cache_path):
        tensors['kv_cache'] = torch.load(kv_cache_path)
    
    # Load virtual engine
    virtual_engine_path = os.path.join(pass_dir, "virtual_engine.pt")
    if os.path.exists(virtual_engine_path):
        tensors['virtual_engine'] = torch.load(virtual_engine_path)
    
    return tensors

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str) -> bool:
    """Compare two tensors and return if they're close enough."""
    if tensor1 is None and tensor2 is None:
        return True
    
    if isinstance(tensor1, int) or isinstance(tensor2, int):
        return tensor1 == tensor2
    
    if isinstance(tensor1, float) or isinstance(tensor2, float):
        return abs(tensor1 - tensor2) < 1e-5

    if (tensor1 is None) != (tensor2 is None):
        print(f"One tensor is None, the other is not for {name}")
        return False
    
    # Check shapes
    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch for {name}: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    # Consider them equal if difference is small enough
    is_close = torch.allclose(tensor1, tensor2, atol=1e-5)
    
    return is_close

def compare_outputs(dir1: str, dir2: str):
    """Compare outputs from two different runs."""
    # Get all pass directories from both runs
    passes1 = sorted(glob.glob(os.path.join(dir1, "pass_*")))
    passes2 = sorted(glob.glob(os.path.join(dir2, "pass_*")))
    
    if len(passes1) != len(passes2):
        print(f"Different number of passes: {len(passes1)} vs {len(passes2)}")
        return
    
    for pass_idx, (pass1_dir, pass2_dir) in enumerate(zip(passes1, passes2)):
        print(f"\nComparing pass {pass_idx + 1}:")
        
        # Load tensors from both passes
        tensors1 = load_tensors_from_pass(pass1_dir)
        tensors2 = load_tensors_from_pass(pass2_dir)
        
        # Compare each tensor
        for tensor_name in tensors1.keys():
            if tensor_name in tensors2:
                is_close = compare_tensors(
                    tensors1[tensor_name], 
                    tensors2[tensor_name],
                    tensor_name
                )
                status = "✓" if is_close else "✗"
                print(f"{status} {tensor_name}")
            else:
                print(f"✗ {tensor_name}: Missing in second run")

def main():
    # Run with two different backends
    backend1 = "FLASH_ATTN_VLLM_V1"
    backend2 = "FLASHINFER_VLLM_V1"
    
    output_dir1 = run_cache_layer(backend1)
    output_dir2 = run_cache_layer(backend2)
    
    print(f"\nComparing outputs between {backend1} and {backend2}")
    compare_outputs(output_dir1, output_dir2)

if __name__ == "__main__":
    main()
