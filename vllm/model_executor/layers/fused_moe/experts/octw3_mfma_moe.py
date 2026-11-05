# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Octave-W3 experts on CDNA3 (gfx942): rotate + MFMA GEMM on the 3-bit codes.

The kernels live in csrc/rocm/moe_octw3_mfma_cdna.cu and are JIT-built on
first use. Weights use the RDNA3 kernel layout (``octw3_repack``).
"""

import functools
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
    Rdna3OctaveW3Experts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

_SRC = Path(__file__).resolve().parents[5] / "csrc/rocm/moe_octw3_mfma_cdna.cu"
_DECL = """
void octw3_rotate_cdna(torch::Tensor x, torch::Tensor y, bool silu_mul,
                       int64_t had);
void moe_octw3_mfma_gemm_cdna(torch::Tensor a, torch::Tensor c, torch::Tensor w,
                              torch::Tensor scales, torch::Tensor topk_weights,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor expert_ids,
                              torch::Tensor num_tokens_post_padded,
                              int64_t a_div, int64_t output_topk,
                              int64_t block_size_m);
void moe_octw3_mfma_gemm_cdna_cfg(torch::Tensor a, torch::Tensor c,
                                  torch::Tensor w, torch::Tensor scales,
                                  torch::Tensor topk_weights,
                                  torch::Tensor sorted_token_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_padded,
                                  int64_t a_div, int64_t output_topk,
                                  int64_t cfg);
int64_t moe_octw3_mfma_cfg_block_m(int64_t cfg);
void moe_octw3_pk_gemm_cdna(torch::Tensor a, torch::Tensor c, torch::Tensor w,
                            torch::Tensor topk_weights,
                            torch::Tensor sorted_token_ids,
                            torch::Tensor expert_ids,
                            torch::Tensor num_tokens_post_padded,
                            int64_t a_div, int64_t output_topk,
                            int64_t block_size_m);
"""


@functools.cache
def octw3_cdna_ops():
    import os

    from torch.utils.cpp_extension import load_inline

    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
    return load_inline(
        name="octw3_mfma_cdna",
        cpp_sources=_DECL,
        cuda_sources=_SRC.read_text(),
        functions=[
            "octw3_rotate_cdna",
            "moe_octw3_mfma_gemm_cdna",
            "moe_octw3_mfma_gemm_cdna_cfg",
            "moe_octw3_mfma_cfg_block_m",
            "moe_octw3_pk_gemm_cdna",
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def octw3_dec_pack(w: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[E, K/32, 3, N] codes + [E, K/32, N] fp16 scales -> the decode layout
    [E, N/16, K/128, 4, 16, 4] int32 (3 code words + scale per column and
    32-k group)."""
    E, G, _, N = w.shape
    codes = w.view(E, G // 4, 4, 3, N // 16, 16).permute(0, 4, 1, 2, 5, 3)
    sc = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    sc = sc.view(E, G // 4, 4, N // 16, 16).permute(0, 3, 1, 2, 4).unsqueeze(-1)
    return torch.cat([codes, sc], -1).contiguous()


def on_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx942" in torch.cuda.get_device_properties(0).gcnArchName


# Block size by token count, measured on the Flash-Next layer (E=384, top-10):
# 16-row decode kernel up to 512 tokens, then the LDS-tiled GEMM.
def _block_size_m(num_tokens: int) -> int:
    if num_tokens <= 512:
        return 16
    return 32 if num_tokens < 2048 else 64


class CdnaOctaveW3Experts(Rdna3OctaveW3Experts):
    """Same workspaces and routing as the RDNA3 WMMA path; the weights are in
    the decode layout (octw3_dec_pack), with the scales inside."""

    @staticmethod
    def _supports_current_device() -> bool:
        return on_gfx942() and _SRC.exists()

    @staticmethod
    def pack_weights(
        w: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E, G, _, N = w.shape
        return octw3_dec_pack(w, scales).view(E, G, 4, N), scales.new_empty(0)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ) -> None:
        ops = octw3_cdna_ops()
        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E
        rows = M * top_k
        block_size_m = _block_size_m(M)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts, expert_map
        )
        topk_weights_f32 = topk_weights.reshape(-1).float()
        no_topk_weights = torch.empty(0, dtype=torch.float32, device=output.device)

        # workspace13 may alias `output`; it is only zeroed after GEMM1 has
        # consumed the rotated input.
        x_rot = workspace13.view(-1)[: M * K].view(M, K)
        ws2 = workspace2.view(-1)
        gate_up_out = ws2[: rows * 2 * N].view(rows, 2 * N)
        act = ws2[rows * 2 * N : rows * 3 * N].view(rows, N)

        ops.octw3_rotate_cdna(hidden_states, x_rot, False, self.had_w13)
        ops.moe_octw3_pk_gemm_cdna(
            x_rot,
            gate_up_out,
            w1,
            topk_weights_f32 if apply_router_weight_on_input else no_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            0,
            block_size_m,
        )
        ops.octw3_rotate_cdna(gate_up_out, act, True, self.had_w2)
        output.zero_()
        ops.moe_octw3_pk_gemm_cdna(
            act,
            output,
            w2,
            no_topk_weights if apply_router_weight_on_input else topk_weights_f32,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            top_k,
            block_size_m,
        )
