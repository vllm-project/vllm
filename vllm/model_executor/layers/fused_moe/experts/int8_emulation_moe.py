# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int8 weight-only quantization emulation for MoE.

Weights are dequantized from packed int8 to BF16 once at load time;
the forward pass then runs plain TritonExperts in BF16.

Weight layout (GPTQ int8, pack_factor=4):
  w13_int32: [E, K//4, 2*N]  int32  packed LSB-first along K (gate+up stacked on dim 2)
  w2_int32:  [E, N//4, K]    int32  packed LSB-first along N
  w13_scale: [E, K//group_size, 2*N]  float16 | bfloat16
  w2_scale:  [E, N//group_size, K]    float16 | bfloat16
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _int8_dequant_kernel(
    w_ptr,  # [E, K_packed, N] int32
    s_ptr,  # [E, n_groups, N] fp16/bf16
    out_ptr,  # [E, N, K] or [E, K, N] fp16/bf16
    w_stride_e,
    w_stride_k,
    w_stride_n,
    s_stride_e,
    s_stride_g,
    s_stride_n,
    out_stride_e,
    out_stride_row,
    out_stride_col,
    E,
    K_packed,
    N,
    group_size,
    TRANSPOSE_OUTPUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """
    Grid: (E, cdiv(K_packed, BLOCK_K), cdiv(N, BLOCK_N))
    Each program handles one [BLOCK_K*4, BLOCK_N] dequantized tile
    for one expert.
    """
    pid_e = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = k_offs < K_packed
    n_mask = n_offs < N

    w_ptrs = (
        w_ptr
        + pid_e * w_stride_e
        + k_offs[:, None] * w_stride_k
        + n_offs[None, :] * w_stride_n
    )
    w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

    for b in tl.static_range(4):
        byte_val = (w_tile >> (b * 8)) & 0xFF
        w_fp = (byte_val - 128).to(tl.float32)

        k_rows = k_offs * 4 + b
        k_row_mask = k_mask & (k_rows < K_packed * 4)
        # Clamp g_idx so out-of-bounds k_offs don't produce a pointer that
        # escapes the scale tensor (Triton computes the address speculatively
        # before applying the mask, which causes an illegal memory access).
        n_groups = K_packed * 4 // group_size
        g_idx = tl.minimum(k_rows // group_size, n_groups - 1)

        s_ptrs = (
            s_ptr
            + pid_e * s_stride_e
            + g_idx[:, None] * s_stride_g
            + n_offs[None, :] * s_stride_n
        )
        scale = tl.load(s_ptrs, mask=k_row_mask[:, None] & n_mask[None, :], other=0.0)

        dequant = (w_fp * scale.to(tl.float32)).to(OUTPUT_DTYPE)

        if TRANSPOSE_OUTPUT:
            out_ptrs = (
                out_ptr
                + pid_e * out_stride_e
                + n_offs[:, None] * out_stride_row
                + k_rows[None, :] * out_stride_col
            )
            tl.store(
                out_ptrs, tl.trans(dequant), mask=n_mask[:, None] & k_row_mask[None, :]
            )
        else:
            out_ptrs = (
                out_ptr
                + pid_e * out_stride_e
                + k_rows[:, None] * out_stride_row
                + n_offs[None, :] * out_stride_col
            )
            tl.store(out_ptrs, dequant, mask=k_row_mask[:, None] & n_mask[None, :])


def triton_unpack_and_dequant_int8_gptq(
    w_int32: torch.Tensor,
    scale: torch.Tensor,
    transpose_output: bool,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Triton kernel: unpack GPTQ int8 weights and dequantize.

    Args:
        w_int32: [E, K_packed, N] int32, 4 int8 per int32, LSB-first along K.
        scale:   [E, K//group_size, N] float16 or bfloat16.
        transpose_output: True -> [E, N, K]; False -> [E, K, N].
        output_dtype: torch.float16 or torch.bfloat16.
    """
    E, K_packed, N = w_int32.shape
    K = K_packed * 4
    group_size = K // scale.shape[1]

    out_shape = (E, N, K) if transpose_output else (E, K, N)
    out = torch.empty(out_shape, dtype=output_dtype, device=w_int32.device)

    tl_dtype = tl.float16 if output_dtype == torch.float16 else tl.bfloat16
    BLOCK_K, BLOCK_N = 32, 32

    _int8_dequant_kernel[(E, triton.cdiv(K_packed, BLOCK_K), triton.cdiv(N, BLOCK_N))](
        w_int32,
        scale,
        out,
        w_int32.stride(0),
        w_int32.stride(1),
        w_int32.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        E,
        K_packed,
        N,
        group_size,
        TRANSPOSE_OUTPUT=transpose_output,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        OUTPUT_DTYPE=tl_dtype,
    )
    return out


class Int8EmulationTritonExperts(TritonExperts):
    """Int8 W-only MoE that dequantizes weights to BF16 at load time.

    Weights arrive already dequantized (convert_to_wna16_moe_kernel_format
    does the unpacking); apply() simply forwards to TritonExperts.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Int8EmulationTritonExperts MoE backend. Int8 weights are "
            "dequantized to BF16 at load time; this uses more memory than a "
            "native int8 kernel and is intended for devices or configurations "
            "where a native W8A16 kernel is unavailable."
        )
        # Weights are dequantized to BF16 before apply() is called, so
        # TritonExperts must see them as plain float -- clear the int8 dtype
        # and scales so the hidden-size assertion and kernel dispatch treat
        # them as unquantized.
        self.quant_config._w1.dtype = None
        self.quant_config._w2.dtype = None
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kInt8Static and activation_key is None

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return None

    @property
    def block_shape(self) -> list[int] | None:
        return None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if w1.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                "Int8EmulationTritonExperts.apply() received non-float weights "
                f"(dtype={w1.dtype}). Weights must be dequantized to BF16 before "
                "the forward pass via convert_to_wna16_moe_kernel_format."
            )
        return super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
