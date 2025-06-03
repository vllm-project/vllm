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
        # print("stdout:\n", result.stdout)
        # print("stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running cache_layer with backend {backend}:")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        raise  # Re-raise the exception after printing
    return f"{backend}_attn_captures"

def load_tensors_from_pass(pass_dir: str) -> Dict[str, Any]:
    """Load all tensors from a specific pass directory."""
    tensors = {}
    pt_files = glob.glob(os.path.join(pass_dir, "*.pt"))
    
    for file_path in pt_files:
        tensor_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            tensors[tensor_name] = torch.load(file_path, map_location='cpu') # Added map_location for safety
        except Exception as e:
            print(f"Error loading tensor {tensor_name} from {file_path}: {e}")
            tensors[tensor_name] = None # Or handle error as appropriate
            
    return tensors

def compare_tensors(tensor1: Any, tensor2: Any, name: str) -> bool:
    """Compare two tensors/values and return if they're close enough."""
    if tensor1 is None and tensor2 is None:
        return True
    
    if (tensor1 is None) != (tensor2 is None):
        print(f"Mismatch: One tensor is None, the other is not for {name}. tensor1 is None: {tensor1 is None}, tensor2 is None: {tensor2 is None}")
        return False

    if isinstance(tensor1, (list, tuple)) and isinstance(tensor2, (list, tuple)):
        if len(tensor1) != len(tensor2):
            print(f"List/Tuple length mismatch for {name}: {len(tensor1)} vs {len(tensor2)}")
            return False
        for i, (t1_item, t2_item) in enumerate(zip(tensor1, tensor2)):
            if not compare_tensors(t1_item, t2_item, f"{name}[{i}]"):
                return False
        return True
    elif isinstance(tensor1, (list, tuple)) or isinstance(tensor2, (list, tuple)):
        print(f"Type mismatch for {name}: One is a list/tuple, the other is not.")
        return False

    if isinstance(tensor1, (int, float, bool, str)) and isinstance(tensor2, (int, float, bool, str)):
        if isinstance(tensor1, float) and isinstance(tensor2, float):
            return abs(tensor1 - tensor2) < 1e-5
        return tensor1 == tensor2

    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        print(f"Type mismatch for {name}: Expected torch.Tensor, got {type(tensor1)} and {type(tensor2)}")
        return False
    
    # Check shapes
    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch for {name}: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    # Consider them equal if difference is small enough
    # Handle boolean tensors separately as allclose might not be suitable
    if tensor1.dtype == torch.bool and tensor2.dtype == torch.bool:
        return torch.equal(tensor1, tensor2)
        
    is_close = torch.allclose(tensor1.float(), tensor2.float(), atol=1e-5, rtol=1e-4) # Added rtol for float comparison
    
    # Debugging for a specific tensor if needed
    if name == "kv_cache_post" and False:
        print("shape", tensor1.shape)
        print(f"tensor1: {tensor1}")
        print(f"tensor2: {tensor2}")
    #     print(f"is_close: {is_close}")
    #     print(f"abs(tensor1 - tensor2): {torch.abs(tensor1 - tensor2)}")

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
    backend1 = "FLASH_ATTN"
    backend2 = "FLASH_ATTN_VLLM_V1"
    
    output_dir1 = run_cache_layer(backend1)
    output_dir2 = run_cache_layer(backend2)
    
    print(f"\nComparing outputs between {backend1} and {backend2}")
    compare_outputs(output_dir1, output_dir2)

if __name__ == "__main__":
    main()
