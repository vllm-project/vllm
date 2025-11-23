import torch
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import mxfp8_fake_w8a8_moe_quant_config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, dequant_mxfp8_to_bf16


def _cast_mxfp8_scales_to_fp32(scales: torch.Tensor) -> torch.Tensor:
    return (scales.to(torch.int32) << 23).view(torch.float32)


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def games():
    A = torch.cat([1024*torch.randn(2, 32, dtype=torch.bfloat16, device="cuda"),
                   512*torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
                   ], dim=-1)

    B = torch.cat([64*torch.randn(64, 32, dtype=torch.bfloat16, device="cuda"),
                   32*torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
                   ], dim=-1)

    C = torch.matmul(A, B)
    print(f"{C=}")

    print(f"{A.shape=}")
    print(f"{B.shape=}")
    print(f"{C.shape=}")

    A_q, A_scales = mxfp8_e4m3_quantize(A)
    B_q, B_scales = mxfp8_e4m3_quantize(B)

    print(f"{A_scales.shape=}")
    print(f"{B_scales.shape=}")

    A_scales_bf16 = _cast_mxfp8_scales_to_bf16(A_scales)
    B_scales_bf16 = _cast_mxfp8_scales_to_bf16(B_scales)
    print(f"{A_scales_bf16=}")
    print(f"{B_scales_bf16=}")

    # C_q = torch.matmul(A_q.to(torch.bfloat16), B_q.to(torch.bfloat16))
    # C_dq_1 = C_q * torch.matmul(A_scales_bf16, B_scales_bf16)
    # C_dq_1 = C_q * (A_scales_bf16.repeat_interleave(32, dim=-1) * B_scales_bf16.repeat_interleave(32, dim=-1))

    A_dq = A_q.to(torch.bfloat16) * A_scales_bf16.repeat_interleave(32, dim=-1)
    B_dq = B_q.to(torch.bfloat16) * B_scales_bf16.repeat_interleave(32, dim=-1)
    C_dq_2 = torch.matmul(A_dq, B_dq)

    # print(f"{C_dq_1=}")
    print(f"{C_dq_2=}")

    # print(f"{C_dq_1 - C_dq_2=}")

    print(f"{torch.max(torch.abs(A_dq - A))=}")
    print(f"{torch.max(torch.abs(B_dq - B))=}")
    print(f"{torch.max(torch.abs(C_dq_2 - C))=}")


def fake_mxfp8_matmul(A: torch.Tensor, B: torch.Tensor):
    A_q, A_scales = mxfp8_e4m3_quantize(A)
    A_dq = dequant_mxfp8_to_bf16(A_q, A_scales)
    
    B_q, B_scales = mxfp8_e4m3_quantize(B)
    B_dq = dequant_mxfp8_to_bf16(B_q, B_scales)
    return torch.matmul(A_dq, B_dq)


def fake_mxfp8_mlp(w1: torch.Tensor, w2: torch.Tensor, a: torch.Tensor):
    x = fake_mxfp8_matmul(a, w1.t().contiguous())
    x = torch.square(torch.nn.functional.relu(x))
    x = fake_mxfp8_matmul(x, w2.t().contiguous())
    return x


def ref_fake_mxfp8_moe(w1: torch.Tensor, w2: torch.Tensor, a: torch.Tensor, topk_weights, topk_ids):    
    E = w1.shape[0]

    final_hidden_states = torch.zeros_like(a, dtype=a.dtype)
    expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes=E)
    expert_mask = expert_mask.permute(2, 0, 1)

    for expert_idx in range(E):
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)

        if token_indices.numel() > 0:
            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = a[token_indices]
            expert_output = fake_mxfp8_mlp(w1[expert_idx], w2[expert_idx], expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)

    return final_hidden_states.type(a.dtype)


def moe():
    E = 16
    M = 2
    K = 512
    N = 2048
    topk = 2

    dtype = torch.bfloat16

    # Input tensor
    # M * K
    a = torch.randn((M, K), dtype=dtype, device="cuda") / 10
    a_q, a_scales = mxfp8_e4m3_quantize(a)
    a_dq = dequant_mxfp8_to_bf16(a_q, a_scales)

    # Generate MXFP8 weights
    w1 = (torch.rand((E, N, K), dtype=dtype, device="cuda") - 0.5) * 2
    w1_q, w1_scales = mxfp8_e4m3_quantize(w1)
    w1_dq = dequant_mxfp8_to_bf16(w1_q, w1_scales)

    w2 = (torch.rand((E, K, N), dtype=dtype, device="cuda") - 0.5) * 2
    w2_q, w2_scales = mxfp8_e4m3_quantize(w2)
    w2_dq = dequant_mxfp8_to_bf16(w2_q, w2_scales)

    # # Create a tensor of zeros
    score = torch.zeros((M, E), dtype=dtype, device="cuda")
    # For each row, randomly pick topk unique indices to be set to 1
    for m in range(M):
        topk_indices = torch.randperm(E, device="cuda")[:topk]
        score[m, topk_indices] = 1.0
    topk_weights, topk_ids = torch.topk(score, topk)
    print(f"{topk_ids=}")
    print(f"{topk_weights=}")

    quant_config = mxfp8_fake_w8a8_moe_quant_config(w1_scales, w2_scales)
    print(f"{quant_config.use_mxfp8_fake_w8a8=}")

    out = fused_experts(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        quant_config=quant_config,
        activation="relu2_no_mul"
    )

    ref_out = ref_fake_mxfp8_moe(w1_dq, w2_dq, a_dq, topk_weights, topk_ids)
    print(f"{out=}")
    print(f"{ref_out=}")

    print(f"{torch.max(torch.abs(out - ref_out))=}")


if __name__ == "__main__":
    # games()
    moe()