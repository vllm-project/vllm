import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton_scaled_matmul_tut import block_scaled_matmul_kernel
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import mxfp8_fake_w8a8_moe_quant_config, mxfp8_w8a8_moe_quant_config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, dequant_mxfp8_to_bf16
from flashinfer import mxfp8_quantize as flashinfer_mxfp8_e4m3_quantize

from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale


def _cast_mxfp8_scales_to_fp32(scales: torch.Tensor) -> torch.Tensor:
    return (scales.to(torch.int32) << 23).view(torch.float32)


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp8_to_fp32(
    x: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor to FP32.
    
    Args:
        x: FP8 E4M3 tensor to dequantize
        scales: uint8 tensor containing MXFP8 scales
        
    Returns:
        FP32 dequantized tensor
    """
    scales_fp32 = _cast_mxfp8_scales_to_fp32(scales)
    # Repeat scales along the last dimension to match the block size
    scales_expanded = scales_fp32.reshape(*x.shape[:-1], -1).repeat_interleave(32, dim=-1)
    return x.to(torch.float32) * scales_expanded


def block_scaled_matmul(a, a_scale, b, b_scale, dtype_dst, block_scale_type="mxfp8", is_swizzled=False):
    assert block_scale_type in ["mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"

    M = a.shape[0]
    N = b.shape[0]
    K = a.shape[1]
    assert b.shape[1] == K

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 32
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2
    NUM_STAGES = 4

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])
   
    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # Use 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements.
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    a_scale_shape = [1, M//128, K//VEC_SIZE//4, 2, 256]
    b_scale_shape = [1, N//128, K//VEC_SIZE//4, 2, 256]
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

    if is_swizzled:
        a_scale = a_scale.view(a_scale_shape)
        b_scale = b_scale.view(b_scale_shape)
    else:
        a_scale = swizzle_blockscale(a_scale).view(a_scale_shape)
        b_scale = swizzle_blockscale(b_scale).view(b_scale_shape)

    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.bfloat16:
        dtype_dst = 2
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        ELEM_PER_BYTE_A,
        ELEM_PER_BYTE_B,
        VEC_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        rep_m,
        rep_n,
        rep_k,
        NUM_STAGES,
    )

    return output



def games():
    M = 1024
    N = 2048
    K = 512

    A = torch.cat([torch.rand(1).item()*torch.rand(1,32, dtype=torch.bfloat16, device="cuda") for _ in range(M*K//32)]).reshape(M,K)
    B = torch.cat([torch.rand(1).item()*torch.rand(1,32, dtype=torch.bfloat16, device="cuda") for _ in range(N*K//32)]).reshape(N,K)

    C = torch.matmul(A, B.t())
    # print(f"{C=}")

    print(f"{A.shape=}")
    print(f"{B.shape=}")
    print(f"{C.shape=}")

    A_q, A_scales = mxfp8_e4m3_quantize(A)
    B_q, B_scales = mxfp8_e4m3_quantize(B)

    print(f"{A_scales.shape=}")
    print(f"{B_scales.shape=}")

    A_scales_bf16 = _cast_mxfp8_scales_to_bf16(A_scales)
    B_scales_bf16 = _cast_mxfp8_scales_to_bf16(B_scales)
    # print(f"{A_scales_bf16=}")
    # print(f"{B_scales_bf16=}")

    A_dq_bf16 = A_q.to(torch.bfloat16) * A_scales_bf16.repeat_interleave(32, dim=-1)
    B_dq_bf16 = B_q.to(torch.bfloat16) * B_scales_bf16.repeat_interleave(32, dim=-1)
    C_q_dq_bf16 = torch.matmul(A_dq_bf16, B_dq_bf16.t())

    A_dq_fp32 = dequant_mxfp8_to_fp32(A_q, A_scales)
    B_dq_fp32 = dequant_mxfp8_to_fp32(B_q, B_scales)
    C_q_dq_fp32 = torch.matmul(A_dq_fp32, B_dq_fp32.t())

    # print(f"{C_q_dq=}")

    print(f"{torch.max(torch.abs(A_dq_bf16 - A))=}")
    print(f"{torch.max(torch.abs(B_dq_bf16 - B))=}")
    print(f"{torch.max(torch.abs(C_q_dq_bf16 - C))=}")

    # A_q, A_scales_swz = flashinfer_mxfp8_e4m3_quantize(A, is_sf_swizzled_layout=True)
    # B_q, B_scales_swz = flashinfer_mxfp8_e4m3_quantize(B, is_sf_swizzled_layout=True)
    # C_mxfp8_fp32 = block_scaled_matmul(A_q, A_scales_swz, B_q, B_scales_swz, torch.float32, "mxfp8", is_swizzled=True)
    # C_mxfp8_bf16 = block_scaled_matmul(A_q, A_scales_swz, B_q, B_scales_swz, torch.bfloat16, "mxfp8", is_swizzled=True)
    C_mxfp8_fp32 = block_scaled_matmul(A_q, A_scales, B_q, B_scales, torch.float32, "mxfp8", is_swizzled=False)
    C_mxfp8_bf16 = block_scaled_matmul(A_q, A_scales, B_q, B_scales, torch.bfloat16, "mxfp8", is_swizzled=False)
    print(f"{torch.max(torch.abs(C_mxfp8_fp32 - C_q_dq_fp32))=}")
    print(f"{torch.max(torch.abs(C_mxfp8_bf16 - C_q_dq_bf16))=}")
    print(f"{torch.max(torch.abs(C_mxfp8_fp32.to(torch.bfloat16) - C_mxfp8_bf16))=}")


def fake_mxfp8_matmul(A: torch.Tensor, B: torch.Tensor):
    # MXFP8 quantizes along the last dimension, but we want to quantize along matmul inner dim.
    # So B is transposed relative to the matmul operation.
    # A: [M, K]
    # B: [N, K]
    A_q, A_scales = mxfp8_e4m3_quantize(A)
    A_dq = dequant_mxfp8_to_bf16(A_q, A_scales)
    
    B_q, B_scales = mxfp8_e4m3_quantize(B)
    B_dq = dequant_mxfp8_to_bf16(B_q, B_scales)
    return torch.matmul(A_dq, B_dq.t())


def fake_mxfp8_mlp(w1: torch.Tensor, w2: torch.Tensor, a: torch.Tensor):
    x = fake_mxfp8_matmul(a, w1)
    x = torch.square(torch.nn.functional.relu(x))
    x = fake_mxfp8_matmul(x, w2)
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
    M = 128
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

    quant_config_fake = mxfp8_fake_w8a8_moe_quant_config(w1_scales, w2_scales)
    print(f"{quant_config_fake.use_mxfp8_fake_w8a8=}")

    quant_config = mxfp8_w8a8_moe_quant_config(w1_scales, w2_scales)
    print(f"{quant_config.use_mxfp8_w8a8=}")

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