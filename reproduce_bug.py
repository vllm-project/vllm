
import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNormGated

def test_rmsnorm_gated():
    hidden_size = 64
    eps = 1e-5
    
    # Test initialization with default activation (silu)
    norm = RMSNormGated(hidden_size, eps=eps)
    print(f"Default activation: {norm.activation}")
    assert norm.activation == "silu"
    
    # Test initialization with sigmoid activation
    norm_sigmoid = RMSNormGated(hidden_size, eps=eps, activation="sigmoid")
    print(f"Sigmoid activation: {norm_sigmoid.activation}")
    assert norm_sigmoid.activation == "sigmoid"
    
    # Test forward_native with silu
    x = torch.randn(2, 4, hidden_size)
    z = torch.randn(2, 4, hidden_size)
    out_silu = norm.forward_native(x, z)
    print("Forward native (silu) successful")
    
    # Test forward_native with sigmoid
    out_sigmoid = norm_sigmoid.forward_native(x, z)
    print("Forward native (sigmoid) successful")
    
    # Check that outputs are different
    assert not torch.allclose(out_silu, out_sigmoid)
    print("Outputs for silu and sigmoid are different as expected")

if __name__ == "__main__":
    try:
        test_rmsnorm_gated()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
