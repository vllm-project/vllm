# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import dataclasses
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.ibm_fused_moe.persistent_gg_bf16 import grouped_gemm_persistent as ibm_gg_bf16
from vllm.model_executor.layers.fused_moe.ibm_fused_moe.persistent_gg_bf16_tma import (grouped_gemm_persistent as ibm_gg_bf16_tma, is_kernel_supported as is_ibm_gg_bf16_tma_supported)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (_moe_permute)


@dataclasses.dataclass
class TestConfig:
    m: int
    n: int
    k: int
    e: int
    num_topk: int
    dtype: torch.dtype
    group_size: int

@dataclasses.dataclass
class TestTensors:
    a: torch.Tensor
    w: torch.Tensor
    topk_ids: torch.Tensor

    def __repr__(self):

        def describe_t(t: torch.Tensor, name: str) -> str:
            return  f"  - {name} : {t.shape} {t.dtype} {t.device}\n"

        s = ""
        s += "Test Tensors :\n"
        s += describe_t(self.a, "a") 
        s += describe_t(self.w, "w") 
        s += describe_t(self.topk_ids, "topk_ids") 
        return s

    @staticmethod
    def make(cfg: TestConfig) -> "TestTensors":
        m, n, k, e, num_topk, dtype = (cfg.m, cfg.n, cfg.k, cfg.e, cfg.num_topk, cfg.dtype)
        device = "cuda"
        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        w = torch.randn((e, n, k), device=device, dtype=dtype) / 10
        topk_ids = torch.randint(low=0, high=e, size=(m, num_topk), device = device)
        return TestTensors(a = a,
                           w = w,
                           topk_ids = topk_ids)

@dataclasses.dataclass
class GroupedTestTensors:
    a_grouped: torch.Tensor
    w: torch.Tensor
    expert_ids_grouped: torch.Tensor
    inv_perm : torch.Tensor

    group_size : int

    @staticmethod
    def make(tt: TestTensors,
             tc: TestConfig) -> "GroupedTestTensors":
        a_grouped, _, _, expert_ids, inv_perm  = _moe_permute(curr_hidden_states=tt.a,
                            a1q_scale = None,
                            curr_topk_ids = tt.topk_ids,
                            global_num_experts = tc.e,
                            expert_map = None,
                            block_m = tc.group_size)
                    
        return GroupedTestTensors(a_grouped, tt.w, expert_ids, inv_perm,
                                  group_size = tc.group_size)

def torch_gg(gtt: GroupedTestTensors) -> torch.Tensor:
    group_size = gtt.group_size
    m = gtt.a_grouped.size(0)
    n = gtt.w.size(1)
    torch_out = torch.empty((m, n), dtype=gtt.a_grouped.dtype, device="cuda")
    expert_ids_cpu = gtt.expert_ids_grouped.to(device="cpu") 
    num_groups = m // group_size 

    for g in range(num_groups):
        s = g * group_size
        e = s + group_size   
        ei = expert_ids_cpu[s]
        o = torch_out[s:e] 
        a = gtt.a_grouped[s:e] 
        torch.mm(a, gtt.w[ei].t(), out=o)

    return torch_out

Ms = [1, 33, 64, 222, 256]
Ns = [128, 1024, 2048]
Ks = [128, 511, 1024]
Es = [8, 64]
TOPKs = [2, 6]

IMPLs = [ibm_gg_bf16_tma, ibm_gg_bf16]
@pytest.mark.parametrize("M", Ms)
@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("K", Ks)
@pytest.mark.parametrize("E", Es)
@pytest.mark.parametrize("TOPK", TOPKs)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("impl", IMPLs)
def test_ibm_bf16(M, N, K, E, TOPK, dtype, impl):

    config: TestConfig = TestConfig(m = M, n = N, k = K, e = E, num_topk = TOPK, dtype = dtype, group_size = 128)
    tt: TestTensors = TestTensors.make(config)
    gtt: GroupedTestTensors = GroupedTestTensors.make(tt, config) 

    if impl == ibm_gg_bf16_tma and not is_ibm_gg_bf16_tma_supported(gtt.a_grouped, gtt.w):
        pytest.skip("ibm_gg_bf16_tma doesn't support this combination")

    print (f"M {M} N {N} K {K} E {E} TOPK {TOPK} | A {gtt.a_grouped.shape}  ...", flush=True)

    ref_output = torch_gg(gtt) 
    impl_output = impl(gtt.a_grouped,
                       gtt.w,
                       gtt.expert_ids_grouped)

    torch.testing.assert_close(ref_output, impl_output)
