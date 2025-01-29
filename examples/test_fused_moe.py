import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

dtype = torch.bfloat16
activation = "silu"
hidden_dim = 7168
ffn_dim = 2048
num_experts = 32
num_tokens = 163840
fused_weights = False
permuted_weights = True
k = 8
hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype)
score = torch.randn((num_tokens, num_experts), dtype=torch.float32)
routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
router_weights, expert_routing_table = torch.topk(routing_weights,
                                              k,
                                              dim=-1)
router_weights /= router_weights.sum(dim=-1, keepdim=True)
router_weights = router_weights.to(dtype=dtype)
# w12 = [torch.randn((hidden_dim, 2 * ffn_dim), dtype=dtype).to("hpu") for _ in range(num_experts)]
# w3 = [torch.randn((ffn_dim, hidden_dim), dtype=dtype).to("hpu") for _ in range(num_experts)]
w12 = [torch.randn((2 * ffn_dim, hidden_dim), dtype=dtype).to("hpu") for _ in range(num_experts)]
w3 = [torch.randn((hidden_dim, ffn_dim), dtype=dtype).to("hpu") for _ in range(num_experts)]

print(f"hidden_states.shape: {hidden_states.shape}, device: {hidden_states.device}, dtype: {hidden_states.dtype}")
print(f"expert_routing_table.shape: {expert_routing_table.shape}, device: {expert_routing_table.device}, dtype: {expert_routing_table.dtype}")
print(f"router_weights.shape: {router_weights.shape}, device: {router_weights.device}, dtype: {router_weights.dtype}")
print(f"w12[0].shape: {w12[0].shape}, device: {w12[0].device}, dtype: {w12[0].dtype}")
print(f"w3[0].shape: {w3[0].shape}, device: {w3[0].device}, dtype: {w3[0].dtype}")

result = torch.ops.hpu.mixture_of_experts(
    hidden_states.to("hpu"),
    expert_routing_table.to("hpu"),
    router_weights.to("hpu"),
    w12,
    w3,
    permuted_weights,
    activation,
    0,
    num_experts - 1,
)

print(f"result.shape: {result.shape}")

'''
hidden_states.shape: torch.Size([163840, 7168])
expert_routing_table.shape: torch.Size([163840, 8])
router_weights.shape: torch.Size([163840, 8])
w12[0].shape: torch.Size([4096, 7168])
w3[0].shape: torch.Size([7168, 2048])
result.shape: torch.Size([163840, 7168])
'''
