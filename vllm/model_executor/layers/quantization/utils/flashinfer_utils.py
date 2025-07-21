# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def calculate_tile_tokens_dim(num_tokens, top_k, num_experts):
    from flashinfer import next_positive_power_of_2

    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, 2, x.shape[-2] // 2,
                     x.shape[-1]).flip(dims=[1]).reshape(x.shape)


def rotate_flashinfer_fp8_moe_weights(gemm1_weights: torch.Tensor,
                                      gemm2_weights: torch.Tensor):
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a
    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i]))

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(
        gemm1_weights_fp8_interleaved).reshape(num_experts,
                                               2 * intermediate_size,
                                               hidden_size)

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8),
                epilogue_tile_m))

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8),
                             epilogue_tile_m))

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn)
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn)
