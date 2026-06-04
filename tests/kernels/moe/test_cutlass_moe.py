# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import dataclasses
from math import prod

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsFp4,
    CutlassExpertsFp8,
    CutlassExpertsW4A8Fp8,
    run_cutlass_moe_fp8,
    run_cutlass_moe_w4a8_fp8,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (7, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (224, 1024, 1024),
    (224, 3072, 1024),
    (224, 3072, 1536),
    (32768, 1024, 1024),
    # These sizes trigger wrong answers.
    # (7232, 2048, 5120),
    # (40000, 2048, 5120),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def test_cutlass_moe_supports_gelu_tanh_activation_metadata():
    assert CutlassExpertsFp8._supports_activation(MoEActivation.GELU_TANH)
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH)
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH_NO_MUL)


@dataclasses.dataclass
class MOETensors:
    a: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    ab_strides1: torch.Tensor
    c_strides1: torch.Tensor
    ab_strides2: torch.Tensor
    c_strides2: torch.Tensor

    @staticmethod
    def make_moe_tensors(
        m: int, k: int, n: int, e: int, dtype: torch.dtype
    ) -> "MOETensors":
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        ab_strides1 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        return MOETensors(
            a=a,
            w1=w1,
            w2=w2,
            ab_strides1=ab_strides1,
            c_strides1=c_strides1,
            ab_strides2=ab_strides2,
            c_strides2=c_strides2,
        )


@dataclasses.dataclass
class MOETensors8Bit(MOETensors):
    # quantized
    a_q: torch.Tensor | None = None  # a -> a_q
    w1_q: torch.Tensor | None = None  # w1 -> w1_q
    w2_q: torch.Tensor | None = None  # w2 -> w2_q
    a_scale: torch.Tensor | None = None
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    # dequantized
    a_d: torch.Tensor | None = None  # a -> a_q -> a_d
    w1_d: torch.Tensor | None = None  # w1 -> w1_q -> w1_d
    w2_d: torch.Tensor | None = None  # w2 -> w2_q -> w2_d

    @staticmethod
    def make_moe_tensors_8bit(
        m: int, k: int, n: int, e: int, per_act_token: bool, per_out_channel: bool
    ) -> "MOETensors8Bit":
        dtype = torch.half
        q_dtype = torch.float8_e4m3fn

        moe_tensors_fp16 = MOETensors.make_moe_tensors(m, k, n, e, dtype)

        # a -> a_q, w1 -> w1_q, w2 -> w2_q
        n_b_scales = 2 * n if per_out_channel else 1
        k_b_scales = k if per_out_channel else 1
        # Get the right scale for tests.
        a_q, a_scale = ops.scaled_fp8_quant(
            moe_tensors_fp16.a, None, use_per_token_if_dynamic=per_act_token
        )

        w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

        w1_scale = torch.empty((e, n_b_scales, 1), device="cuda", dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1), device="cuda", dtype=torch.float32)
        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w1[expert], use_per_token_if_dynamic=per_out_channel
            )
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w2[expert], use_per_token_if_dynamic=per_out_channel
            )

        # a_q -> a_d, w1_q -> w1_d, w2_q -> w2_d
        a_d = a_q.float().mul(a_scale).to(dtype)
        w1_d = torch.empty_like(moe_tensors_fp16.w1)
        w2_d = torch.empty_like(moe_tensors_fp16.w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()

        return MOETensors8Bit(
            a=moe_tensors_fp16.a,
            w1=moe_tensors_fp16.w1,
            w2=moe_tensors_fp16.w2,
            ab_strides1=moe_tensors_fp16.ab_strides1,
            c_strides1=moe_tensors_fp16.c_strides1,
            ab_strides2=moe_tensors_fp16.ab_strides2,
            c_strides2=moe_tensors_fp16.c_strides2,
            a_q=a_q,
            w1_q=w1_q,
            w2_q=w2_q,
            a_scale=a_scale,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a_d=a_d,
            w1_d=w1_d,
            w2_d=w2_d,
        )


def run_with_expert_maps(
    num_experts: int,
    num_local_experts: int,
    quant_config: FusedMoEQuantConfig,
    **cutlass_moe_kwargs,
):
    def slice_experts():
        slice_params = [
            "w1",
            "w2",
        ]
        full_tensors = {
            k: v
            for k, v in cutlass_moe_kwargs.items()
            if k in slice_params and k in cutlass_moe_kwargs
        }

        for i in range(0, num_experts, num_local_experts):
            s, e = i, i + num_local_experts

            # make expert map
            expert_map = [-1] * num_experts
            expert_map[s:e] = list(range(num_local_experts))
            expert_map = torch.tensor(expert_map, dtype=torch.int32, device="cuda")

            # update cutlass moe arg with expert_map
            cutlass_moe_kwargs["expert_map"] = expert_map
            # update cutlass moe arg tensors
            for k, t in full_tensors.items():
                cutlass_moe_kwargs[k] = t[s:e]

            new_quant_config = copy.deepcopy(quant_config)
            new_quant_config._w1.scale = quant_config.w1_scale[s:e]
            new_quant_config._w2.scale = quant_config.w2_scale[s:e]

            yield cutlass_moe_kwargs, new_quant_config

    out_tensor = torch.zeros_like(cutlass_moe_kwargs["hidden_states"])
    for kwargs, new_quant_config in slice_experts():
        w2 = kwargs["w2"]
        a = kwargs["hidden_states"]
        kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            CutlassExpertsFp8(
                moe_config=make_dummy_moe_config(
                    num_experts=w2.shape[0],
                    hidden_dim=w2.shape[1],
                    intermediate_size_per_partition=w2.shape[2],
                    in_dtype=a.dtype,
                ),
                quant_config=new_quant_config,
            ),
        )
        out_tensor = out_tensor + kernel.apply(
            **kwargs,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            apply_router_weight_on_input=False,
        )

    return out_tensor


def run_8_bit(
    moe_tensors: MOETensors8Bit,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
    num_local_experts: int | None = None,
) -> torch.Tensor:
    assert not any(
        [
            t is None
            for t in [
                moe_tensors.w1_q,
                moe_tensors.w2_q,
                moe_tensors.w1_scale,
                moe_tensors.w2_scale,
                moe_tensors.a_scale,
            ]
        ]
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=moe_tensors.w1_scale,
        w2_scale=moe_tensors.w2_scale,
        per_act_token_quant=per_act_token,
        per_out_ch_quant=per_out_ch,
        # Set to moe_tensors.a_scale iff static scales + per tensor.
        # This is not currently being tested.
        a1_scale=None,
    )

    kwargs = {
        "hidden_states": moe_tensors.a,
        "w1": moe_tensors.w1_q,  # type: ignore[union-attr]
        "w2": moe_tensors.w2_q,  # type: ignore[union-attr]
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }

    num_experts = moe_tensors.w1.size(0)  # type: ignore[attr-defined]
    with_ep = num_local_experts is not None or num_local_experts == num_experts
    if not with_ep:
        kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            CutlassExpertsFp8(
                moe_config=make_dummy_moe_config(
                    num_experts=moe_tensors.w2_q.shape[0],  # type: ignore[union-attr]
                    hidden_dim=moe_tensors.w2_q.shape[1],  # type: ignore[union-attr]
                    intermediate_size_per_partition=moe_tensors.w2_q.shape[2],  # type: ignore[union-attr]
                    in_dtype=moe_tensors.a.dtype,
                ),
                quant_config=quant_config,
            ),
        )
        return kernel.apply(
            **kwargs,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    assert num_local_experts is not None
    return run_with_expert_maps(
        num_experts,
        num_local_experts,  # type: ignore[arg-type]
        quant_config,
        **kwargs,
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    monkeypatch,
    workspace_init,
    ep_size: int | None = None,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.

        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        triton_output = fused_experts(
            mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids, quant_config=quant_config
        )

        if ep_size is not None:
            assert e % ep_size == 0, "Cannot distribute experts evenly"
            number_local_experts = e // ep_size
        else:
            number_local_experts = None

        cutlass_output = run_8_bit(
            mt, topk_weights, topk_ids, per_act_token, per_out_ch, number_local_experts
        )

        # Note 5.5 only needed for larger problem sizes, 5 works ok for
        # the rest.
        torch.testing.assert_close(
            triton_output, cutlass_output, atol=5.5e-2, rtol=1e-2
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_cuda_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        dtype = torch.half

        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        triton_output = fused_experts(
            mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids, quant_config=quant_config
        )

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run_8_bit(
                mt, topk_weights, topk_ids, per_act_token, per_out_ch
            )

        torch.accelerator.synchronize()
        graph.replay()
        torch.accelerator.synchronize()

        torch.testing.assert_close(triton_output, cutlass_output, atol=9e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [64])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("per_act_token", [True])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8, 16])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_EP(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    monkeypatch,
    workspace_init,
):
    test_cutlass_moe_8_bit_no_graph(
        m,
        n,
        k,
        e,
        topk,
        per_act_token,
        per_out_channel,
        monkeypatch,
        workspace_init,
        ep_size,
    )


LARGE_MNK_FACTORS = [
    (1, 8192, 5120, 31),
    (32768, 1024, 1024, 16),
    (65536, 512, 1024, 16),
]


@pytest.mark.parametrize("m,n,k,topk", LARGE_MNK_FACTORS)
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("per_act_token", [False])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [8])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_EP_large(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    monkeypatch,
    workspace_init,
):
    test_cutlass_moe_8_bit_no_graph(
        m,
        n,
        k,
        e,
        topk,
        per_act_token,
        per_out_channel,
        monkeypatch,
        workspace_init,
        ep_size,
    )


@pytest.mark.parametrize("m,n,k,topk", [(1, 8192, 5120, 31)])
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("per_act_token", [False])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [8])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_run_cutlass_moe_fp8(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(
            m, k, n, e, per_act_token, per_out_channel
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)
        # we want to make sure there is at least one token that's generated in
        # this expert shard and at least one token that's NOT generated in this
        # expert shard
        topk_ids[0][0] = -1
        topk_ids[0][1] = 1

        workspace13_shape = (m * topk, max(2 * n, k))
        workspace2_shape = (m * topk, max(n, k))
        output_shape = (m, k)

        workspace13 = torch.empty(
            prod(workspace13_shape), device="cuda", dtype=mt.a.dtype
        )
        workspace2 = torch.empty(
            prod(workspace2_shape), device="cuda", dtype=mt.a.dtype
        )

        num_local_experts = e // ep_size
        start, end = 0, num_local_experts
        expert_map = [-1] * e
        expert_map[start:end] = list(range(num_local_experts))
        expert_map = torch.tensor(expert_map, dtype=torch.int32, device="cuda")

        ab_strides1 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device="cuda", dtype=torch.int64)

        activation = MoEActivation.SILU
        a1q, a1q_scale = moe_kernel_quantize_input(
            mt.a, mt.a_scale, torch.float8_e4m3fn, per_act_token
        )
        global_num_experts = -1 if mt.w1_q is None else mt.w1_q.size(0)
        func = lambda output: run_cutlass_moe_fp8(
            output,
            a1q,
            mt.w1_q,
            mt.w2_q,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            mt.w1_scale,
            mt.w2_scale,
            a1q_scale,
            None,
            ab_strides1,
            ab_strides2,
            c_strides1,
            c_strides2,
            workspace13,
            workspace2,
            None,
            mt.a.dtype,
            per_act_token,
            per_out_channel,
            False,
            topk_weights,
        )

        workspace13.random_()
        output_random_workspace = torch.empty(
            output_shape, device="cuda", dtype=mt.a.dtype
        )
        func(output_random_workspace)

        workspace13.fill_(0)
        output_zero_workspace = torch.zeros(
            output_shape, device="cuda", dtype=mt.a.dtype
        )
        func(output_zero_workspace)

        torch.testing.assert_close(
            output_random_workspace, output_zero_workspace, atol=5e-3, rtol=1e-3
        )


# --- run_cutlass_moe_w4a8_fp8 tests ---
GROUP_SIZE_W4A8 = 128

IS_W4A8_SUPPORTED = (
    current_platform.is_cuda()
    and current_platform.get_device_capability() is not None
    and current_platform.get_device_capability()[0] >= 9
)


def _to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return tensor.clamp(min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)


@dataclasses.dataclass
class W4A8MoETensors:
    """Tensors for testing run_cutlass_moe_w4a8_fp8."""

    hidden_states: torch.Tensor  # bf16, shape (M, K) — reference activations
    a1q: torch.Tensor  # fp8, shape (M, K) — quantized activations fed to the kernel
    a1q_scale: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    # Dequantized reference weights (bf16) used to build a numerical reference
    # via torch_experts. These carry the same int4-group quantization error as
    # the packed kernel inputs, so tolerance can stay tight on weights.
    w1_ref: torch.Tensor  # shape (E, 2*N, K)
    w2_ref: torch.Tensor  # shape (E, K, N)
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    w1_chan_scale: torch.Tensor
    w2_chan_scale: torch.Tensor
    a_strides1: torch.Tensor
    a_strides2: torch.Tensor
    b_strides1: torch.Tensor
    b_strides2: torch.Tensor
    c_strides1: torch.Tensor
    c_strides2: torch.Tensor
    s_strides1: torch.Tensor
    s_strides2: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    M: int
    K: int
    N: int
    E: int
    topk: int


def make_w4a8_moe_tensors(
    M: int,
    K: int,
    N: int,
    E: int,
    topk: int,
    device: str = "cuda",
    seed: int = 42,
) -> W4A8MoETensors:
    """Build tensors for run_cutlass_moe_w4a8_fp8 (group_size=128, bf16 input)."""
    set_random_seed(seed)
    assert K % GROUP_SIZE_W4A8 == 0 and N % GROUP_SIZE_W4A8 == 0

    # Hidden states (bf16) plus fp8-quantized copy + per-token scale for the
    # kernel. The kernel consumes already-quantized activations; the bf16 copy
    # is retained so tests can build a float reference via torch_experts.
    hidden_states = torch.randn((M, K), device=device, dtype=torch.bfloat16) / 8
    a1q, a1q_scale = ops.scaled_fp8_quant(
        hidden_states, None, use_per_token_if_dynamic=True
    )
    if a1q_scale.dim() == 0 or (a1q_scale.numel() != M and a1q_scale.numel() != M * 1):
        a1q_scale = a1q_scale.expand(M, 1)

    # Router scores and top-k
    score = torch.randn((M, E), device=device, dtype=torch.bfloat16)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )
    # Ensure at least one token uses an expert in this shard and one does not
    topk_ids[0, 0] = -1
    if topk > 1:
        topk_ids[0, 1] = min(1, E - 1)

    wtype = scalar_types.int4

    def quantize_and_pack_rows(
        w_float: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """w_float (size_k, size_n).

        Returns (w_q_packed, w_s, w_ref):
          w_q_packed: (size_n, size_k//8) int32, packed for cutlass encode.
          w_s:        group-wise scale from quantize_weights.
          w_ref:      (size_k, size_n) dequantized float — same quantization
                      error as w_q_packed; used to build a numerical reference.
        """
        w_ref, w_q, w_s, _ = quantize_weights(
            w_float, wtype, group_size=GROUP_SIZE_W4A8, zero_points=False
        )
        w_q = pack_rows(w_q & 0x0F, 4, w_q.size(0), w_q.size(1))
        w_q = w_q.t().contiguous()
        return w_q, w_s, w_ref

    # W1: logical (E, 2*N, K) -> per expert (2*N, K) -> packed (2*N//8, K)
    w1_qs: list[torch.Tensor] = []
    w1_ss: list[torch.Tensor] = []
    w1_refs: list[torch.Tensor] = []
    for _ in range(E):
        w1_float = torch.randn((2 * N, K), device=device, dtype=torch.float16) / 8
        w1_q, w1_s, w1_ref = quantize_and_pack_rows(w1_float)
        w1_qs.append(w1_q)
        w1_ss.append(w1_s)
        w1_refs.append(w1_ref)

    w1_stacked = torch.stack(w1_qs)
    w1_packed, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(w1_stacked)

    # W1 scales: quantize_weights gives (2*N/128, K) per expert;
    # kernel wants (E, 2*N, K//128)
    n_groups_k = K // GROUP_SIZE_W4A8
    w1_s_list = [s.reshape(2 * N, n_groups_k) for s in w1_ss]
    w1_s_stack = torch.stack(w1_s_list)
    w1_scale_fp8 = _to_fp8(w1_s_stack)
    w1_scale_packed = ops.cutlass_pack_scale_fp8(
        w1_scale_fp8.permute(0, 2, 1).contiguous()
    )
    w1_chan_scale = torch.ones((E, 2 * N), device=device, dtype=torch.float32)

    # W2: logical (E, K, N) -> per expert (K, N) -> packed (N//8, K) for encode
    w2_qs = []
    w2_ss = []
    w2_refs: list[torch.Tensor] = []
    for _ in range(E):
        w2_float = torch.randn((K, N), device=device, dtype=torch.float16) / 8
        w2_q, w2_s, w2_ref = quantize_and_pack_rows(w2_float)
        w2_qs.append(w2_q)
        w2_ss.append(w2_s)
        w2_refs.append(w2_ref)

    w2_stacked = torch.stack(w2_qs)
    w2_packed, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(w2_stacked)

    # W2 scales: quantize_weights gives (K/128, N) per expert;
    # kernel wants (E, N, K//128)
    w2_s_list = [s.reshape(N, n_groups_k) for s in w2_ss]
    w2_s_stack = torch.stack(w2_s_list)
    w2_scale_fp8 = _to_fp8(w2_s_stack)
    w2_scale_packed = ops.cutlass_pack_scale_fp8(
        w2_scale_fp8.permute(0, 2, 1).contiguous()
    )
    w2_chan_scale = torch.ones((E, N), device=device, dtype=torch.float32)

    # Strides (a, c, s)
    a_strides1 = torch.full((E,), K, device=device, dtype=torch.int64)
    a_strides2 = torch.full((E,), N, device=device, dtype=torch.int64)
    c_strides1 = torch.full((E,), 2 * N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((E,), K, device=device, dtype=torch.int64)
    s_strides1 = torch.zeros((E, 2), device=device, dtype=torch.int64)
    s_strides1[:, 0] = 2 * N
    s_strides2 = torch.zeros((E, 2), device=device, dtype=torch.int64)
    s_strides2[:, 0] = K

    w1_ref_stack = torch.stack([w.to(torch.bfloat16) for w in w1_refs])
    w2_ref_stack = torch.stack([w.to(torch.bfloat16) for w in w2_refs])

    return W4A8MoETensors(
        hidden_states=hidden_states,
        a1q=a1q,
        a1q_scale=a1q_scale,
        w1=w1_packed,
        w2=w2_packed,
        w1_ref=w1_ref_stack,
        w2_ref=w2_ref_stack,
        w1_scale=w1_scale_packed,
        w2_scale=w2_scale_packed,
        w1_chan_scale=w1_chan_scale,
        w2_chan_scale=w2_chan_scale,
        a_strides1=a_strides1,
        a_strides2=a_strides2,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        c_strides1=c_strides1,
        c_strides2=c_strides2,
        s_strides1=s_strides1,
        s_strides2=s_strides2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        M=M,
        K=K,
        N=N,
        E=E,
        topk=topk,
    )


@pytest.mark.skipif(
    not IS_W4A8_SUPPORTED,
    reason="W4A8 CUTLASS MoE is not supported on this GPU.",
)
@pytest.mark.parametrize(
    "m,k,n,e,topk",
    [
        (64, 256, 256, 4, 2),
        (128, 512, 512, 8, 2),
    ],
)
def test_run_cutlass_moe_w4a8_fp8_no_graph(m, k, n, e, topk):
    """run_cutlass_moe_w4a8_fp8: output matches a torch_experts reference and
    does not depend on workspace contents."""
    set_random_seed(7)
    tensors = make_w4a8_moe_tensors(M=m, K=k, N=n, E=e, topk=topk)

    workspace13_shape = (tensors.M * tensors.topk, max(2 * tensors.N, tensors.K))
    workspace2_shape = (tensors.M * tensors.topk, max(tensors.N, tensors.K))
    workspace13 = torch.empty(
        prod(workspace13_shape), device="cuda", dtype=torch.bfloat16
    )
    workspace2 = torch.empty(
        prod(workspace2_shape), device="cuda", dtype=torch.bfloat16
    )
    output_shape = (tensors.M, tensors.K)

    kernel_kwargs = dict(
        hidden_states=tensors.a1q,
        w1=tensors.w1,
        w2=tensors.w2,
        topk_ids=tensors.topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=tensors.E,
        expert_map=None,
        w1_scale=tensors.w1_scale,
        w2_scale=tensors.w2_scale,
        a1q_scale=tensors.a1q_scale,
        a2_scale=None,
        w1_chan_scale=tensors.w1_chan_scale,
        w2_chan_scale=tensors.w2_chan_scale,
        a_strides1=tensors.a_strides1,
        a_strides2=tensors.a_strides2,
        b_strides1=tensors.b_strides1,
        b_strides2=tensors.b_strides2,
        c_strides1=tensors.c_strides1,
        c_strides2=tensors.c_strides2,
        s_strides1=tensors.s_strides1,
        s_strides2=tensors.s_strides2,
        expert_num_tokens=None,
        out_dtype=torch.bfloat16,
        per_act_token=True,
        per_out_ch=True,
        use_batched_format=False,
        topk_weights=tensors.topk_weights,
        group_size=GROUP_SIZE_W4A8,
    )

    workspace13.random_()
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    run_cutlass_moe_w4a8_fp8(
        output, workspace13=workspace13, workspace2=workspace2, **kernel_kwargs
    )

    assert output.shape == output_shape
    assert output.dtype == torch.bfloat16

    # Workspace-independent: zeroed workspace yields the same result.
    workspace13.zero_()
    workspace2.zero_()
    output_zero = torch.zeros(output_shape, device="cuda", dtype=torch.bfloat16)
    run_cutlass_moe_w4a8_fp8(
        output_zero, workspace13=workspace13, workspace2=workspace2, **kernel_kwargs
    )
    torch.testing.assert_close(output, output_zero, atol=5e-3, rtol=1e-2)

    # Numerical reference: run torch_experts on dequantized (bf16) weights.
    # w_ref carries the same int4-group quantization error as the kernel's
    # packed weights, so weight error is captured. The remaining discrepancy
    # is dominated by fp8 activation quantization, hence the 1e-1 tolerance
    # (matches the nvfp4 moe test pattern).
    reference = torch_experts(
        tensors.hidden_states,
        tensors.w1_ref,
        tensors.w2_ref,
        tensors.topk_weights,
        tensors.topk_ids,
        global_num_experts=tensors.E,
        activation=MoEActivation.SILU,
    )
    torch.testing.assert_close(output, reference, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(
    not IS_W4A8_SUPPORTED,
    reason="W4A8 CUTLASS MoE is not supported on this GPU.",
)
def test_run_cutlass_moe_w4a8_fp8_with_expert_map():
    """Test run_cutlass_moe_w4a8_fp8 with expert_map (e.g. expert parallelism)."""
    set_random_seed(11)
    E_global = 8
    num_local = 4
    tensors = make_w4a8_moe_tensors(M=64, K=256, N=256, E=num_local, topk=2, seed=11)
    # This rank holds global experts 4..7 as local 0..3
    expert_map = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 2, 3],
        dtype=torch.int32,
        device="cuda",
    )
    # Route tokens to global experts 4..7 so they hit local experts
    tensors.topk_ids = torch.clamp(tensors.topk_ids + 4, 4, E_global - 1)

    workspace13 = torch.empty(
        tensors.M * tensors.topk * max(2 * tensors.N, tensors.K),
        device="cuda",
        dtype=torch.bfloat16,
    )
    workspace2 = torch.empty(
        tensors.M * tensors.topk * max(tensors.N, tensors.K),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty((tensors.M, tensors.K), device="cuda", dtype=torch.bfloat16)

    run_cutlass_moe_w4a8_fp8(
        output,
        tensors.a1q,
        tensors.w1,
        tensors.w2,
        tensors.topk_ids,
        MoEActivation.SILU,
        global_num_experts=E_global,
        expert_map=expert_map,
        w1_scale=tensors.w1_scale,
        w2_scale=tensors.w2_scale,
        a1q_scale=tensors.a1q_scale,
        a2_scale=None,
        w1_chan_scale=tensors.w1_chan_scale,
        w2_chan_scale=tensors.w2_chan_scale,
        a_strides1=tensors.a_strides1,
        a_strides2=tensors.a_strides2,
        b_strides1=tensors.b_strides1,
        b_strides2=tensors.b_strides2,
        c_strides1=tensors.c_strides1,
        c_strides2=tensors.c_strides2,
        s_strides1=tensors.s_strides1,
        s_strides2=tensors.s_strides2,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_num_tokens=None,
        out_dtype=torch.bfloat16,
        per_act_token=True,
        per_out_ch=True,
        use_batched_format=False,
        topk_weights=tensors.topk_weights,
        group_size=GROUP_SIZE_W4A8,
    )

    assert output.shape == (tensors.M, tensors.K)

    # Numerical reference: every token lands on a local expert (topk_ids was
    # clamped into 4..7 and expert_map sends 4..7 -> local 0..3), so the EP
    # output must equal torch_experts over the local weights with topk_ids
    # shifted back into the local index space.
    local_topk_ids = tensors.topk_ids - 4
    reference = torch_experts(
        tensors.hidden_states,
        tensors.w1_ref,
        tensors.w2_ref,
        tensors.topk_weights,
        local_topk_ids,
        global_num_experts=num_local,
        activation=MoEActivation.SILU,
    )
    torch.testing.assert_close(output, reference, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(
    not IS_W4A8_SUPPORTED,
    reason="W4A8 CUTLASS MoE is not supported on this GPU.",
)
@pytest.mark.parametrize(
    "m,k,n,e,topk",
    [
        (64, 256, 256, 4, 2),
        (128, 512, 512, 8, 2),
    ],
)
def test_cutlass_moe_w4a8_fp8_modular(m, k, n, e, topk):
    """Drive CutlassExpertsW4A8Fp8 through FusedMoEKernel — the production
    wiring path — and check the output matches a torch_experts reference.

    The direct-kernel tests above call run_cutlass_moe_w4a8_fp8 with hand-built
    strides and pre-quantized activations. Real callers go through the modular
    kernel, which does the fp8 activation quantization in prepare_finalize and
    wires up the expert class; this test exercises that wiring.
    """
    set_random_seed(9)
    tensors = make_w4a8_moe_tensors(M=m, K=k, N=n, E=e, topk=topk)

    quant_config = int4_w4afp8_moe_quant_config(
        w1_scale=tensors.w1_scale,
        w2_scale=tensors.w2_scale,
        g1_alphas=tensors.w1_chan_scale,
        g2_alphas=tensors.w2_chan_scale,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )

    moe_config = make_dummy_moe_config(
        num_experts=tensors.E,
        experts_per_token=tensors.topk,
        hidden_dim=tensors.K,
        intermediate_size_per_partition=tensors.N,
        in_dtype=torch.bfloat16,
    )

    experts = CutlassExpertsW4A8Fp8(
        out_dtype=torch.bfloat16,
        a_strides1=tensors.a_strides1,
        a_strides2=tensors.a_strides2,
        b_strides1=tensors.b_strides1,
        b_strides2=tensors.b_strides2,
        c_strides1=tensors.c_strides1,
        c_strides2=tensors.c_strides2,
        s_strides1=tensors.s_strides1,
        s_strides2=tensors.s_strides2,
        moe_config=moe_config,
        quant_config=quant_config,
        group_size=GROUP_SIZE_W4A8,
    )

    kernel = mk.FusedMoEKernel(
        make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
        experts,
        inplace=False,
    )

    with set_current_vllm_config(vllm_config):
        output = kernel.apply(
            hidden_states=tensors.hidden_states,
            w1=tensors.w1,
            w2=tensors.w2,
            topk_weights=tensors.topk_weights,
            topk_ids=tensors.topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=tensors.E,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    assert output.shape == (tensors.M, tensors.K)
    assert output.dtype == torch.bfloat16

    reference = torch_experts(
        tensors.hidden_states,
        tensors.w1_ref,
        tensors.w2_ref,
        tensors.topk_weights,
        tensors.topk_ids,
        global_num_experts=tensors.E,
        activation=MoEActivation.SILU,
    )
    torch.testing.assert_close(output, reference, atol=1e-1, rtol=1e-1)
