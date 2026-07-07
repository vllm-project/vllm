import torch
import sys

from vllm.v1.attention.backends.kvcrush import H2OKVCrushCluster

# Create test tensors with specified shapes
bsz = 1
num_kv_heads = 8
num_query_heads = 24
seq_len = 50
head_dim = 128

key_states = torch.randn(bsz, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
query_states = torch.randn(bsz, num_query_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
value_states = torch.randn(bsz, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

print("Input shapes:")
print(f"key_states.shape: {key_states.shape}")
print(f"query_states.shape: {query_states.shape}")
print(f"value_states.shape: {value_states.shape}")
print()

# Initialize KVCrush
window_size = 16
max_capacity_prompt = 32
kvcrush_ratio = 0.25
page_size = 16  # vLLM block size

kvcrush = H2OKVCrushCluster(
    window_size=window_size,
    max_capacity_prompt=max_capacity_prompt,
    kvcrush_ratio=kvcrush_ratio
)

# Call update_kv
print("\nCalling update_kv...")
try:
    compressed_key, compressed_value = kvcrush.update_kv(
        key_states=key_states,
        query_states=query_states,
        value_states=value_states,
        page_size=page_size
    )

    print("\nOutput shapes:")
    print(f"compressed_key.shape: {compressed_key.shape}")
    print(f"compressed_value.shape: {compressed_value.shape}")
    print("\nTest passed!")

except Exception as e:
    print(f"\nTest failed with error: {e}")
    import traceback
    traceback.print_exc()
