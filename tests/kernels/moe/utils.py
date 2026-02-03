# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import per_block_cast_to_int8
from tests.kernels.quantization.nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
    NaiveBatchedExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.utils.math_utils import round_up


def make_dummy_moe_config(
    num_experts: int = 1,
    experts_per_token: int = 1,
    hidden_dim: int = 1,
    intermediate_size_per_partition: int = 1,
    in_dtype: torch.dtype = torch.bfloat16,
) -> FusedMoEConfig:
    """
    This is a dummy config for the mk constructor interface
    as most kernels like DeepGEMM, CUTLASSFp4, Triton, MARLIN
    do not actually use this config.

    CUTLASSFp8 needs to set some params for workshapes.
    """
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size_per_partition,
        num_local_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation="silu",
        in_dtype=in_dtype,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
    )


def triton_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant=False,
    block_shape: list[int] | None = None,
) -> torch.Tensor:
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )

    return fused_experts(a, w1, w2, topk_weight, topk_ids, quant_config=quant_config)


def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(
            max_num_tokens, num_dispatchers=1, num_local_experts=w1.shape[0], rank=0
        ),
        BatchedTritonExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=1,
            quant_config=quant_config,
            moe_config=make_dummy_moe_config(),
        ),
    )

    return fused_experts(a, w1, w2, topk_weight, topk_ids)


def naive_batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(
            max_num_tokens, num_dispatchers=1, num_local_experts=w1.shape[0], rank=0
        ),
        NaiveBatchedExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=1,
            quant_config=quant_config,
            moe_config=make_dummy_moe_config(),
        ),
    )

    return fused_experts(a, w1, w2, topk_weight, topk_ids)


def chunk_scales(
    scales: torch.Tensor | None, start: int, end: int
) -> torch.Tensor | None:
    if scales is not None:
        if scales.numel() == 1:
            return scales
        else:
            return scales[start:end]
    return None


def make_quantized_test_activations(
    E: int,
    m: int,
    k: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | None = None,
    block_shape: list[int] | None = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    a = torch.randn((E, m, k), device="cuda", dtype=in_dtype) / 10
    a_q = a
    a_scale = None

    if quant_dtype is not None:
        assert quant_dtype == torch.float8_e4m3fn or quant_dtype == torch.int8, (
            "only fp8/int8 supported"
        )
        a_q = torch.zeros_like(a, dtype=quant_dtype)
        a_scale_l = [None] * E
        for e in range(E):
            a_q[e], a_scale_l[e] = moe_kernel_quantize_input(
                a[e], None, quant_dtype, per_act_token_quant, block_shape
            )
        a_scale = torch.stack(a_scale_l)

        if not per_act_token_quant and block_shape is None:
            a_scale = a_scale.view(E, 1, 1)

    return a, a_q, a_scale


def moe_quantize_weights(
    w: torch.Tensor,
    w_s: torch.Tensor | None,
    quant_dtype: torch.dtype | str | None,
    per_token_quant: bool,
    block_shape: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    assert (
        quant_dtype == torch.float8_e4m3fn
        or quant_dtype == torch.int8
        or quant_dtype == "nvfp4"
    ), "only fp8/int8/nvfp4 supported"

    w_gs = None

    if block_shape is not None:
        assert not per_token_quant
        if quant_dtype == torch.int8:
            w, w_s = per_block_cast_to_int8(w, block_shape)
        elif quant_dtype == torch.float8_e4m3fn:
            w, w_s = per_block_cast_to_fp8(w, block_shape)
        elif quant_dtype == "nvfp4":
            raise RuntimeError("blocked quantization not supported for nvfp4")
        else:
            raise RuntimeError(f"Unsupported quant type {quant_dtype}")
    else:
        if quant_dtype == torch.int8:
            w, w_s = ops.scaled_int8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant
            )
        elif quant_dtype == torch.float8_e4m3fn:
            w, w_s = ops.scaled_fp8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant
            )
        elif quant_dtype == "nvfp4":
            assert not per_token_quant
            w_amax = torch.abs(w).max().to(torch.float32)
            w_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w_amax
            w, w_s = ops.scaled_fp4_quant(w, w_gs)
        else:
            raise RuntimeError(f"Unsupported quant type {quant_dtype}")

    return w, w_s, w_gs


def make_test_weight(
    e: int,
    rows: int,
    cols: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: torch.dtype | str | None = None,
    block_shape: list[int] | None = None,
    per_out_ch_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    w_16 = torch.randn((e, rows, cols), device="cuda", dtype=in_dtype) / 15
    w_gs = None

    if quant_dtype is not None:
        w_l = [None] * e
        w_s_l = [None] * e
        w_gs_l = [None] * e
        for idx in range(e):
            w_l[idx], w_s_l[idx], w_gs_l[idx] = moe_quantize_weights(
                w_16[idx], None, quant_dtype, per_out_ch_quant, block_shape
            )

        w = torch.stack(w_l)
        w_s = torch.stack(w_s_l)
        if e > 0 and w_gs_l[0] is not None:
            w_gs = torch.stack(w_gs_l)
        if w_s.ndim == 2:
            assert w_s.shape[-1] == 1
            w_s = w_s.view(-1, 1, 1)

        if block_shape is not None:
            block_n, block_k = block_shape
            n_tiles = (rows + block_n - 1) // block_n
            k_tiles = (cols + block_k - 1) // block_k
            assert w_s.shape == (e, n_tiles, k_tiles)
    else:
        w = w_16
        w_s = None
        w_gs = None

    return w_16, w, w_s, w_gs


def make_test_weights(
    e: int,
    n: int,
    k: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: torch.dtype | str | None = None,
    block_shape: list[int] | None = None,
    per_out_ch_quant: bool = False,
    make_gate: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None],
]:
    return (
        make_test_weight(
            e,
            (2 if make_gate else 1) * n,
            k,
            in_dtype,
            quant_dtype,
            block_shape,
            per_out_ch_quant,
        ),
        make_test_weight(e, k, n, in_dtype, quant_dtype, block_shape, per_out_ch_quant),
    )


def per_token_cast_to_fp8(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (block_size - (n % block_size)) % block_size
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, block_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def make_test_quant_config(
    e: int,
    n: int,
    k: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | str | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
    make_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, FusedMoEQuantConfig]:
    (_, w1, w1_s, w1_gs), (_, w2, w2_s, w2_gs) = make_test_weights(
        e,
        n,
        k,
        in_dtype,
        quant_dtype,
        per_out_ch_quant=per_act_token_quant,
        block_shape=block_shape,
        make_gate=make_gate,
    )

    # Hacky/trivial scales for nvfp4.
    a1_gscale: torch.Tensor | None = None
    a2_gscale: torch.Tensor | None = None
    if quant_dtype == "nvfp4":
        a1_gscale = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gscale = torch.ones((e,), device="cuda", dtype=torch.float32)
        a1_scale = a1_gscale
        a2_scale = a2_gscale
    else:
        a1_scale = None
        a2_scale = None

    return (
        w1,
        w2,
        FusedMoEQuantConfig.make(
            quant_dtype,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
            w1_scale=w1_s,
            w2_scale=w2_s,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            # TODO: make sure this is handled properly
            g1_alphas=(1 / w1_gs) if w1_gs is not None else None,
            g2_alphas=(1 / w2_gs) if w2_gs is not None else None,
        ),
    )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    quant_config: FusedMoEQuantConfig | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score.float(), topk, renormalize
    )
    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        quant_config=quant_config,
    )


# CustomOp?
class BaselineMM(torch.nn.Module):
    def __init__(
        self,
        b: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        super().__init__()
        self.b = b.to(dtype=torch.float32)
        self.out_dtype = out_dtype

    def forward(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.mm(a.to(dtype=torch.float32), self.b).to(self.out_dtype), None


class TestMLP(torch.nn.Module):
    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        super().__init__()
        self.gate_up_proj = BaselineMM(w1, out_dtype)
        self.down_proj = BaselineMM(w2, out_dtype)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


def make_naive_shared_experts(
    N: int,
    K: int,
    in_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    w1 = torch.randn((K, N * 2), device="cuda", dtype=in_dtype) / 15
    w2 = torch.randn((N, K), device="cuda", dtype=in_dtype) / 15
    return TestMLP(w1, w2, out_dtype=in_dtype)


class RealMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        hidden_act: str = "silu",
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
        w1_s: torch.Tensor | None = None,
        w2_s: torch.Tensor | None = None,
    ) -> None:
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.gate_up_proj.register_parameter(
            "weight", torch.nn.Parameter(w1, requires_grad=False)
        )
        self.gate_up_proj.register_parameter(
            "weight_scale", torch.nn.Parameter(w1_s, requires_grad=False)
        )
        self.gate_up_proj.register_parameter(
            "input_scale", None
        )  # torch.nn.Parameter(None, requires_grad=False))
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.down_proj.register_parameter(
            "weight", torch.nn.Parameter(w2, requires_grad=False)
        )
        self.down_proj.register_parameter(
            "weight_scale", torch.nn.Parameter(w2_s, requires_grad=False)
        )
        self.down_proj.register_parameter(
            "input_scale", None
        )  # torch.nn.Parameter(None, requires_grad=False))
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def make_shared_experts(
    N: int,
    K: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: torch.dtype | str | None = None,
) -> torch.nn.Module:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    (_, w1, w1_s, _), (_, w2, w2_s, _) = make_test_weights(
        1,
        N,
        K,
        in_dtype=in_dtype,
        quant_dtype=quant_dtype,
    )
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(in_dtype)
        if quant_dtype == torch.float8_e4m3fn:
            w1 = w1[0].transpose(0, 1)
            w2 = w2[0].transpose(0, 1)
            w1_s = w1_s[0].transpose(0, 1) if w1_s is not None else None
            w2_s = w2_s[0].transpose(0, 1) if w2_s is not None else None
            quant_config = Fp8Config(True)
        else:
            w1 = w1[0]
            w2 = w2[0]
            w1_s = None
            w2_s = None
            quant_config = None

        return RealMLP(K, N, w1, w2, "silu", quant_config, w1_s=w1_s, w2_s=w2_s)
    finally:
        torch.set_default_dtype(old_dtype)
