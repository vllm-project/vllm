# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils.cpp_extension import load

from vllm import _custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.quantization.gguf import _fuse_mul_mat

sources = ["./csrc/my_bindins.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
my_extension = load(
    name="my_extension",
    sources=sources,
    extra_cuda_cflags=["-arch=sm_80"],  # for CUDA 8.0 arch
    extra_include_paths=["./csrc"],
    verbose=True,
)


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    act,
) -> torch.Tensor:

    num_tokens, _ = x.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]
    out_hidden_states = torch.empty_like(x)
    # TODO get real block size
    BLOCK_SIZE = 4

    sorted_token_ids, expert_ids, _ = moe_align_block_size(
        topk_ids, BLOCK_SIZE, E)
    print(sorted_token_ids, sorted_token_ids.shape)
    print(expert_ids, expert_ids.shape)
    out = my_extension.ggmp_moe_a8(x, w1, sorted_token_ids, expert_ids,
                                   qweight_type, N, top_k, num_tokens)
    print("out 1 ", out, out.shape)
    out = act(out)
    print("out silu ", out, out.shape)
    out = my_extension.ggmp_moe_a8(out, w2, sorted_token_ids, expert_ids,
                                   qweight_type2, w2.shape[1], top_k,
                                   num_tokens)
    print("out 2 ", out, out.shape)
    ops.moe_sum(out.reshape(num_tokens, top_k, w2.shape[1]), out_hidden_states)
    return out_hidden_states


# x = torch.randn(2048, 7168, device="cuda", dtype=torch.float16)
y = torch.arange(7168, device="cuda", dtype=torch.float16) * 0.01
print(y)
x = torch.vstack([y for _ in range(2048)])
print(x, x.shape)
# y = torch.arange(2048, device="cuda", dtype=float16) * 0.01
# x = torch.ones(2048, 7168, device="cuda", dtype=torch.float16)
act = SiluAndMul()

state = torch.load("state.pt")

w13_qweight = state["w13_qweight"].to("cuda")
w2_qweight = state["w2_qweight"].to("cuda")

w13_qweight_type = 10
w2_qweight_type = 11
topk_weights = torch.randn(2048, 8, device="cuda", dtype=torch.float16)
# topk_ids = torch.randint(0, 256, (2048, 8), device="cuda")
topk_ids = torch.ones(2048, 8, device="cuda", dtype=torch.int64) * 255
print(topk_ids)

final_hidden_states = torch.empty_like(x)
final_hidden_states_kern = _fused_moe_gguf(x, w13_qweight, w2_qweight,
                                           topk_weights, topk_ids,
                                           w13_qweight_type, w2_qweight_type,
                                           act)
out_toks = []
for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
    inp = x[tok].reshape((1, ) + x.shape[1:])
    current_hidden_state = None
    for ww, ii in zip(w, idx):
        expert_up = w13_qweight[ii]

        out = _fuse_mul_mat(inp, expert_up, w13_qweight_type)
        if tok == 0:
            print(out[0, :10], out.shape)
        out = act(out)

        expert_down = w2_qweight[ii]
        current_state = _fuse_mul_mat(out, expert_down,
                                      w2_qweight_type).mul_(ww)
        if current_hidden_state is None:
            current_hidden_state = current_state
        else:
            current_hidden_state.add_(current_state)
    final_hidden_states[tok] = current_hidden_state
print(out_toks)
print(final_hidden_states)
print(final_hidden_states_kern)
# assert torch.allclose(final_hidden_states, final_hidden_states_kernk)
