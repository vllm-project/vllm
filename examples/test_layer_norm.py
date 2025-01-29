import habana_frameworks.torch as htorch
import torch

# q_a_layernorm size: 1536
# kv_a_layernorm size: 512
# hidden_size 7168

hidden_size = 10
from vllm.model_executor.layers.layernorm import RMSNorm

# x = torch.randn(10, hidden_size, dtype=torch.bfloat16, device="hpu")
x = torch.zeros(10, hidden_size, dtype=torch.bfloat16, device="hpu")

layer_norm = RMSNorm(hidden_size)

print(layer_norm(x))

