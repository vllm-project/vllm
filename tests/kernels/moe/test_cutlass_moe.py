# SPDX-License-Identifier: Apache-2.0
import dataclasses
from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk)
from vllm.platforms import current_platform

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 3072, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
    (224, 3072, 1024),
    (224, 3072, 1536),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(
    pipeline_parallel_size=1))
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


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
    def make_moe_tensors(m: int, k: int, n: int, e: int,
                         dtype: torch.dtype) -> "MOETensors":
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        return MOETensors(a=a,
                          w1=w1,
                          w2=w2,
                          ab_strides1=ab_strides1,
                          c_strides1=c_strides1,
                          ab_strides2=ab_strides2,
                          c_strides2=c_strides2)


@dataclasses.dataclass
class MOETensors8Bit(MOETensors):
    # quantized
    a_q: Optional[torch.Tensor] = None  # a -> a_q
    w1_q: Optional[torch.Tensor] = None  # w1 -> w1_q
    w2_q: Optional[torch.Tensor] = None  # w2 -> w2_q
    a_scale: Optional[torch.Tensor] = None
    w1_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    # dequantized
    a_d: Optional[torch.Tensor] = None  # a -> a_q -> a_d
    w1_d: Optional[torch.Tensor] = None  # w1 -> w1_q -> w1_d
    w2_d: Optional[torch.Tensor] = None  # w2 -> w2_q -> w2_d

    @staticmethod
    def make_moe_tensors_8bit(m: int, k: int, n: int, e: int,
                              per_act_token: bool,
                              per_out_channel: bool) -> "MOETensors8Bit":
        dtype = torch.half
        q_dtype = torch.float8_e4m3fn

        moe_tensors_fp16 = MOETensors.make_moe_tensors(m, k, n, e, dtype)

        # a -> a_q, w1 -> w1_q, w2 -> w2_q
        n_b_scales = 2 * n if per_out_channel else 1
        k_b_scales = k if per_out_channel else 1
        # Get the right scale for tests.
        _, a_scale = ops.scaled_fp8_quant(
            moe_tensors_fp16.a, use_per_token_if_dynamic=per_act_token)
        a_q, _ = ops.scaled_fp8_quant(moe_tensors_fp16.a,
                                      a_scale,
                                      use_per_token_if_dynamic=per_act_token)
        w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w1[expert],
                use_per_token_if_dynamic=per_out_channel)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w2[expert],
                use_per_token_if_dynamic=per_out_channel)

        # a_q -> a_d, w1_q -> w1_d, w2_q -> w2_d
        a_d = a_q.float().mul(a_scale).to(dtype)
        w1_d = torch.empty_like(moe_tensors_fp16.w1)
        w2_d = torch.empty_like(moe_tensors_fp16.w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()

        return MOETensors8Bit(a=moe_tensors_fp16.a,
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
                              w2_d=w2_d)


def run_with_expert_maps(num_experts: int, num_local_experts: int,
                         **cutlass_moe_kwargs):

    def slice_experts():
        slice_params = [
            "w1_q", "w2_q", "ab_strides1", "ab_strides2", "c_strides1",
            "c_strides2", "w1_scale", "w2_scale"
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
            expert_map = torch.tensor(expert_map,
                                      dtype=torch.int32,
                                      device="cuda")

            # update cutlass moe arg with expert_map
            cutlass_moe_kwargs["expert_map"] = expert_map
            # update cutlass moe arg tensors
            for k, t in full_tensors.items():
                cutlass_moe_kwargs[k] = t[s:e]

            yield cutlass_moe_kwargs

    out_tensor = torch.zeros_like(cutlass_moe_kwargs["a"])
    for kwargs in slice_experts():
        out_tensor = out_tensor + cutlass_moe_fp8(**kwargs)

    return out_tensor


def run_8_bit(moe_tensors: MOETensors8Bit,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              num_local_experts: Optional[int] = None) -> torch.Tensor:
    assert not any([
        t is None for t in [
            moe_tensors.w1_q, moe_tensors.w2_q, moe_tensors.w1_scale,
            moe_tensors.w2_scale, moe_tensors.a_scale
        ]
    ])

    kwargs = {
        'a': moe_tensors.a,
        'w1_q': moe_tensors.w1_q.transpose(1, 2),  # type: ignore[union-attr]
        'w2_q': moe_tensors.w2_q.transpose(1, 2),  # type: ignore[union-attr]
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'ab_strides1': moe_tensors.ab_strides1,
        'c_strides1': moe_tensors.c_strides1,
        'ab_strides2': moe_tensors.ab_strides2,
        'c_strides2': moe_tensors.c_strides2,
        'w1_scale': moe_tensors.w1_scale,
        'w2_scale': moe_tensors.w2_scale,
        'a1_scale': moe_tensors.a_scale
    }

    num_experts = moe_tensors.w1.size(0)
    with_ep = num_local_experts is not None or num_local_experts == num_experts
    if not with_ep:
        return cutlass_moe_fp8(**kwargs)

    assert num_local_experts is not None
    return run_with_expert_maps(
        num_experts,
        num_local_experts,  # type: ignore[arg-type]
        **kwargs)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_8_bit_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token,
                                                  per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a,
                                               score,
                                               topk,
                                               renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.
        triton_output = fused_experts(mt.a_d, mt.w1_d, mt.w2_d, topk_weights,
                                      topk_ids)

        cutlass_output = run_8_bit(mt, topk_weights, topk_ids)

        torch.testing.assert_close(triton_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_8_bit_cuda_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(vllm_config):
        dtype = torch.half

        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token,
                                                  per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(mt.a,
                                               score,
                                               topk,
                                               renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.
        triton_output = fused_experts(mt.a_d, mt.w1_d, mt.w2_d, topk_weights,
                                      topk_ids)

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run_8_bit(mt, topk_weights, topk_ids)

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(triton_output,
                                   cutlass_output,
                                   atol=9e-2,
                                   rtol=1e-2)


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
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_8_bit_EP(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token,
                                                  per_out_channel)

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a,
                                               score,
                                               topk,
                                               renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.
        triton_output = fused_experts(mt.a_d, mt.w1_d, mt.w2_d, topk_weights,
                                      topk_ids)

        assert e % ep_size == 0, "Cannot distribute experts evenly"
        cutlass_output = run_8_bit(mt,
                                   topk_weights,
                                   topk_ids,
                                   num_local_experts=e // ep_size)

        torch.testing.assert_close(triton_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)
