# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import dataclasses
import pytest
from typing import Optional, Union
import math
from itertools import product
from dataclasses import dataclass
from vllm.config import current_platform

# from enum import Enum
import torch
from all2all_moe_utils import (
    MODELS,
    ModelArgs,
    QuantArgs,
    make_block_quant_fp8_weights,
    make_non_quant_weights,
)
from .utils import ProcessGroupInfo, parallel_launch

from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (
    BatchedTritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
    DeepEPLLPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedExperts,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    MoEConfig,
    TritonExperts,
)
from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
    PplxPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.utils import FlexibleArgumentParser

PREPARE_FINALIZE_TYPES = [
    PplxPrepareAndFinalize,
    DeepEPLLPrepareAndFinalize,
    DeepEPHTPrepareAndFinalize,
]
FUSED_EXPERT_TYPES = [
    BatchedDeepGemmExperts,
    BatchedTritonExperts,
    BatchedExperts,
    BatchedTritonOrDeepGemmExperts,
    CutlassExpertsFp8,
    DeepGemmExperts,
    TritonOrDeepGemmExperts,
    TritonExperts,
]


def desc_tensor(t: Optional[torch.Tensor], name: str) -> str:
    if t is not None:
        return f"   {name} : {t.shape} {t.dtype} {t.device}\n"
    else:
        return f"   {name} : {t}\n"


@dataclasses.dataclass
class BenchmarkConfig:
    M: Union[int, list[int]]  # num_tokens per rank
    K: int  # model hidden size
    N: int  # moe intermediate size
    topk: int  # num experts per token
    global_num_experts: int  # Total num experts
    local_num_experts: int  # local num experts
    prepare_finalize: mk.FusedMoEPrepareAndFinalize
    fused_experts: mk.FusedMoEPermuteExpertsUnpermute
    dtype: torch.dtype
    vllm_all2all_backend: str
    vllm_use_deep_gemm: bool
    quant_args: QuantArgs
    world_size: int

    def __repr__(self):
        s = ""
        s += "== Benchmark Config ====\n"
        s += f"  M                      : {self.M} \n"
        s += f"  K                      : {self.K} \n"
        s += f"  N                      : {self.N} \n"
        s += f"  topk                   : {self.topk} \n"
        s += f"  global_num_experts     : {self.global_num_experts} \n"
        s += f"  local_num_experts      : {self.local_num_experts} \n"
        s += f"  prepare_finalize       : {self.prepare_finalize} \n"
        s += f"  fused_experts          : {self.fused_experts} \n"
        s += f"  dtype                  : {self.dtype} \n"
        s += f"  vllm_all2all_backend   : {self.vllm_all2all_backend} \n"
        s += f"  vllm_use_deep_gemm     : {self.vllm_use_deep_gemm} \n"
        s += f"  world_size             : {self.world_size} \n"
        s += "  == Quant Args : ==== \n"
        s += f"     quant_dtype     : {self.quant_args.quant_dtype} \n"
        s += f"     per_act_token   : {self.quant_args.per_act_token} \n"
        s += f"     block_size      : {self.quant_args.block_size} \n"
        return s

    @staticmethod
    def with_m(obj: "BenchmarkConfig", m: int) -> "BenchmarkConfig":
        copy_obj = copy.deepcopy(obj)
        copy_obj.M = m
        return copy_obj

    @staticmethod
    def from_args(args: argparse.Namespace) -> "BenchmarkConfig":
        model_args: ModelArgs = MODELS[args.model]

        vllm_all2all_backend = "naive"
        if args.pf_type == PplxPrepareAndFinalize:
            vllm_all2all_backend = "pplx"
        elif args.pf_type == DeepEPHTPrepareAndFinalize:
            vllm_all2all_backend = "deepep_high_throughput"
        elif args.pf_type == DeepEPLLPrepareAndFinalize:
            vllm_all2all_backend = "deepep_low_latency"

        vllm_use_deep_gemm = False
        if args.experts_type in [
            DeepGemmExperts,
            BatchedDeepGemmExperts,
            TritonOrDeepGemmExperts,
            BatchedTritonOrDeepGemmExperts,
        ]:
            vllm_use_deep_gemm = True

        global_num_experts = args.experts_per_rank * args.world_size
        local_num_experts = args.experts_per_rank

        print(
            f"Model num_experts is {model_args.num_experts} | "
            f"but using global_num_experts {global_num_experts} "
            "to respect args.experts_per_rank"
        )

        return BenchmarkConfig(
            M=args.m_per_rank,
            K=model_args.K,
            N=model_args.N,
            topk=model_args.topk,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            prepare_finalize=args.pf_type,
            fused_experts=args.experts_type,
            dtype=torch.bfloat16,
            vllm_all2all_backend=vllm_all2all_backend,
            vllm_use_deep_gemm=vllm_use_deep_gemm,
            world_size=args.world_size,
            quant_args=model_args.quant_args,
        )


@dataclasses.dataclass
class BenchmarkWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: Optional[torch.Tensor]
    w2_scale: Optional[torch.Tensor]

    def __repr__(self):
        s = ""
        s += "==== BenchmarkWeights == \n"
        s += desc_tensor(self.w1, "w1")
        s += desc_tensor(self.w2, "w2")
        if self.w1_scale is not None:
            s += desc_tensor(self.w1_scale, "w1_scale")
            s += desc_tensor(self.w2_scale, "w2_scale")
        return s

    def to_current_device(self):
        self.w1 = self.w1.to(device=torch.cuda.current_device())
        self.w2 = self.w2.to(device=torch.cuda.current_device())
        self.w1_scale = self.w1_scale.to(device=torch.cuda.current_device())
        self.w2_scale = self.w2_scale.to(device=torch.cuda.current_device())

    def slice_weights(self, rank: int, num_local_experts: int):
        s = rank * num_local_experts
        e = s + num_local_experts
        self.w1 = self.w1[s:e, :, :]
        self.w2 = self.w2[s:e, :, :]
        self.w1_scale = self.w1_scale[s:e, :, :]
        self.w2_scale = self.w2_scale[s:e, :, :]

    def make(bc: BenchmarkConfig) -> "BenchmarkWeights":
        if (
            bc.quant_args.quant_dtype == torch.float8_e4m3fn
            and bc.quant_args.block_size is not None
        ):
            w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
                e=bc.global_num_experts,
                n=bc.N,
                k=bc.K,
                block_size=bc.quant_args.block_size,
            )
            return BenchmarkWeights(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale)

        assert bc.quant_args.quant_dtype is None
        # just make normal dtype weights
        w1, w2 = make_non_quant_weights(
            e=bc.global_num_experts, n=bc.N, k=bc.K, dtype=bc.dtype
        )
        return BenchmarkWeights(w1=w1, w2=w2, w1_scale=None, w2_scale=None)


@dataclasses.dataclass
class BenchmarkInputs:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: torch.Tensor

    def __repr__(self):
        s = ""
        s += "== Benchmark Inputs ==== \n"
        s += desc_tensor(self.hidden_states, "hidden_states")
        s += desc_tensor(self.topk_weights, "topk_weights")
        s += desc_tensor(self.topk_ids, "topk_ids")
        s += desc_tensor(self.expert_map, "expert_map")
        s += f" expert_map    : {self.expert_map} \n"
        return s

    @staticmethod
    def make(bc: BenchmarkConfig, pgi: ProcessGroupInfo):
        dtype = bc.dtype
        topk, m, k = (bc.topk, bc.M, bc.K)
        hidden_states = (
            torch.randn((m, k), device=torch.cuda.current_device(), dtype=dtype) / 10.0
        )

        topk_ids_dtype = None
        if bc.prepare_finalize == PplxPrepareAndFinalize:
            topk_ids_dtype = torch.uint32
        else:
            assert bc.prepare_finalize in [
                DeepEPHTPrepareAndFinalize,
                DeepEPLLPrepareAndFinalize,
            ]
            topk_ids_dtype = torch.int64

        # distribute topk_ids evenly
        topk_ids = torch.empty((m, topk), device="cpu", dtype=topk_ids_dtype)
        for mi in range(m):
            topk_ids[mi] = torch.randperm(bc.global_num_experts)[:topk]
        topk_ids = topk_ids.to(device=torch.cuda.current_device())

        topk_weights = torch.randn(
            topk_ids.shape, dtype=torch.float32, device=torch.cuda.current_device()
        )

        num_local_experts = bc.local_num_experts
        expert_map = torch.full(
            (bc.global_num_experts,), fill_value=-1, dtype=torch.int32
        )
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        expert_map = expert_map.to(
            device=torch.cuda.current_device(), dtype=torch.int32
        )

        return BenchmarkInputs(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )


def describe_benchmark(bc: BenchmarkConfig, bi: BenchmarkInputs):
    all2all_manager = get_ep_group().device_communicator.all2all_manager

    _, num_tokens_per_expert = torch.unique(
        torch.flatten(bi.topk_ids), return_counts=True
    )
    num_tokens_per_expert = num_tokens_per_expert * bc.world_size

    num_tokens_per_dp_rank = [0] * bc.world_size
    for ri in range(bc.world_size):
        s = ri * bc.local_num_experts
        e = s + bc.local_num_experts
        num_tokens_per_dp_rank[ri] = sum(num_tokens_per_expert[s:e])

    s = "== Benchmark Description ====\n"
    s += f"   Total DP Ranks / World Size : {bc.world_size} \n"
    s += (
        "   all2all type : "
        f"{'Internode' if all2all_manager.internode else 'Intranode'} \n"
    )
    s += f"   Tokens Per Rank : {bc.M} | Topk {bc.topk} \n"
    s += f"   Num Tokens per Expert : {num_tokens_per_expert} \n"
    s += "   Num Tokens per Rank (all experts) : \n"
    for ri in range(bc.world_size):
        s += f"     Rank {ri} : {num_tokens_per_dp_rank[ri]} \n"
    print(s)


def is_batched_prepare_finalize(type):
    return type in [PplxPrepareAndFinalize, DeepEPLLPrepareAndFinalize]


def is_batched_fused_experts(type):
    return type in [
        BatchedDeepGemmExperts,
        BatchedTritonExperts,
        BatchedTritonExperts,
        BatchedTritonOrDeepGemmExperts,
        CutlassExpertsFp8,
    ]


def make_fused_experts(bc: BenchmarkConfig, moe: MoEConfig):
    all2all_manager = get_ep_group().device_communicator.all2all_manager

    if (
        is_batched_prepare_finalize(bc.prepare_finalize)
        and not is_batched_fused_experts(bc.fused_experts)
    ) or (
        not is_batched_prepare_finalize(bc.prepare_finalize)
        and is_batched_fused_experts(bc.fused_experts)
    ):
        raise ValueError(f"Incompatible {bc.prepare_finalize} and {bc.fused_experts}")

    if bc.fused_experts == CutlassExpertsFp8:
        raise NotImplementedError("CutlassExpertsFP8 not yet supported")
    if bc.fused_experts == BatchedExperts:
        raise NotImplementedError("naive BatchedExperts not yet supported")

    from vllm import envs

    fp8_dtype = torch.float8_e4m3fn
    use_fp8 = bc.quant_args.quant_dtype == fp8_dtype
    block_shape = bc.quant_args.block_size
    use_deep_gemm = envs.VLLM_USE_DEEP_GEMM

    if bc.fused_experts == BatchedDeepGemmExperts:
        kwargs = {
            "max_num_tokens": moe.max_num_tokens,
            "world_size": all2all_manager.world_size,
            "dp_size": moe.dp_size,
            "block_shape": block_shape,
        }
        print(f"Making BatchedDeepGemmExperts {kwargs} ...")
        experts = BatchedDeepGemmExperts(**kwargs)
    elif bc.fused_experts == BatchedTritonExperts:
        # TODO update args when BatchedTritonExperts supports fp8 block quant
        kwargs = {
            "max_num_tokens": moe.max_num_tokens,
            "world_size": all2all_manager.world_size,
            "dp_size": all2all_manager.tp_group.world_size,
            "use_fp8_w8a8": use_fp8,
            "use_int8_w8a8": False,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "block_shape": block_shape,
            "per_channel_quant": bc.quant_args.per_act_token,
        }
        print(f"Making BatchedTritonExperts {kwargs} ...")
        experts = BatchedTritonExperts(**kwargs)
    elif bc.fused_experts == BatchedTritonOrDeepGemmExperts:
        kwargs = {
            "max_num_tokens": moe.max_num_tokens,
            "world_size": all2all_manager.world_size,
            "dp_size": moe.moe_parallel_config.dp_size,
            "use_fp8_w8a8": use_fp8,
            "use_int8_w8a8": False,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "per_channel_quant": bc.quant_args.per_act_token,
            "block_shape": block_shape,
            "allow_deep_gemm": use_deep_gemm,
        }
        print(f"Making BatchedTritonOrDeepGemmExperts {kwargs} ...")
        experts = BatchedTritonOrDeepGemmExperts(**kwargs)
    elif bc.fused_experts == DeepGemmExperts:
        print("Making DeepGemmExperts () ...")
        experts = DeepGemmExperts()
    elif bc.fused_experts == TritonExperts:
        kwargs = {
            "use_fp8_w8a8": use_fp8,
            "use_int8_w8a8": False,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "block_shape": block_shape,
            "per_channel_quant": bc.quant_args.per_act_token,
        }
        print(f"Making TritonExperts {kwargs} ...")
        experts = TritonExperts(**kwargs)
    elif bc.fused_experts == TritonOrDeepGemmExperts:
        kwargs = {
            "use_fp8_w8a8": use_fp8,
            "block_shape": block_shape,
            "allow_deep_gemm": use_deep_gemm,
        }
        print(f"Making TritonOrDeepGemmExperts {kwargs} ...")
        experts = TritonOrDeepGemmExperts(**kwargs)

    return experts


def bench_modular_kernel(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    bc: BenchmarkConfig,
    bw: BenchmarkWeights,
) -> mk.FusedMoEModularKernel:
    # Transfer BenchmarkWeights to this Rank for good measure.
    bw.to_current_device()
    bw.slice_weights(pgi.rank, num_local_experts=bc.local_num_experts)

    moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        dp_size_=get_dp_group().world_size,
        vllm_parallel_config=vllm_config.parallel_config,
    )

    from vllm import envs

    assert len(bc.quant_args.block_size) == 2
    moe = MoEConfig(
        num_experts=bc.global_num_experts,
        experts_per_token=bc.topk,
        hidden_dim=bc.K,
        num_local_experts=bc.local_num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=bc.dtype,
        # quantization params
        quant_dtype=bc.quant_args.quant_dtype,
        per_act_token=bc.quant_args.per_act_token,
        block_shape=bc.quant_args.block_size,
        max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
    )
    print(f"MoE Config: {moe}")

    # make prepare finalize
    prepare_finalize = FusedMoEMethodBase.make_prepare_finalize(moe)
    print(f"Prepare Finalize : {prepare_finalize}")

    # make fused_experts
    fused_experts = make_fused_experts(bc, moe)

    modular_kernel = mk.FusedMoEModularKernel(
        prepare_finalize=prepare_finalize, fused_experts=fused_experts
    )

    for m in bc.M:
        print(f"Processing num_tokens per rank {m} ...")
        bc_m = BenchmarkConfig.with_m(bc, m)
        bi: BenchmarkInputs = BenchmarkInputs.make(bc_m, pgi)
        print(bi)
        print(bw)
        if pgi.rank == 0:
            describe_benchmark(bc_m, bi)

        # warmup runs
        print("warmup runs ...")
        for _ in range(5):
            modular_kernel.forward(
                hidden_states=bi.hidden_states,
                w1=bw.w1,
                w2=bw.w2,
                topk_weights=bi.topk_weights,
                topk_ids=bi.topk_ids,
                expert_map=bi.expert_map,
                w1_scale=bw.w1_scale,
                w2_scale=bw.w2_scale,
                global_num_experts=bc.global_num_experts,
            )

        print("profiler run ...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=True,
        ) as tprof:
            modular_kernel.forward(
                hidden_states=bi.hidden_states,
                w1=bw.w1,
                w2=bw.w2,
                topk_weights=bi.topk_weights,
                topk_ids=bi.topk_ids,
                expert_map=bi.expert_map,
                w1_scale=bw.w1_scale,
                w2_scale=bw.w2_scale,
                global_num_experts=bc.global_num_experts,
            )
            torch.cuda.synchronize(torch.cuda.current_device())

        tprof.export_chrome_trace(f"./trace_files/m{m}_{pgi.rank}_trace.json")


def main(world_size: int, bc: BenchmarkConfig):
    benchmark_weights: BenchmarkWeights = BenchmarkWeights.make(benchmark_config)
    print(benchmark_weights)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    env_dict = {
        "VLLM_ALL2ALL_BACKEND": bc.vllm_all2all_backend,
        "VLLM_USE_DEEP_GEMM": str(int(bc.vllm_use_deep_gemm)),
    }

    parallel_launch(
        world_size,
        bench_modular_kernel,
        vllm_config,
        env_dict,
        benchmark_config,
        benchmark_weights,
    )

def per_block_cast_to_fp8(
    x: torch.Tensor, block_size_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (
            int(math.ceil(m / 128)) * 128,
            int(math.ceil(n / block_size_n)) * block_size_n,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1q, w2q, w1_scale, w2_scale
    """
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16 = torch.randn((e, 2 * n, k), dtype=dtype, device=device) / 10
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    w2_bf16 = torch.randn((e, k, n), dtype=dtype, device=device) / 10
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn, device=device)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1), device=device, dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2), device=device, dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + 127) // 128, (k + 127) // 128)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    return w1, w2, w1_s, w2_s


def make_non_quant_weights(
    e: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2
    """
    device = torch.cuda.device()
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    return w1, w2

@dataclass
class Config:
    Ms: Union[list[int], int]
    K: int
    N: int
    E: int
    topks: Union[list[int], int]
    dtype: torch.dtype
    block_size: Optional[list[int]]

    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize
    fused_experts_type: mk.FusedMoEPermuteExpertsUnpermute

    world_size: int

    @property
    def M(self) -> int:
        assert isinstance(self.Ms, int)
        return self.Ms

    @property
    def topk(self) -> int:
        assert isinstance(self.topks, int)
        return self.topks

    @property
    def topk_ids_dtype(self) -> Optional[torch.dtype]:
        topk_ids_dtype = None
        if self.prepare_finalize_type == PplxPrepareAndFinalize:
            topk_ids_dtype = torch.uint32
        else:
            assert self.prepare_finalize_type in [
                DeepEPHTPrepareAndFinalize,
                DeepEPLLPrepareAndFinalize,
            ]
            topk_ids_dtype = torch.int64
        return topk_ids_dtype

    @property
    def num_local_experts(self) -> int:
        return self.E // self.world_size

    def is_fp8_block_quantized(self):
        return self.dtype == torch.float8_e4m3fn and self.block_size is not None

    def is_batched_prepare_finalize(self):
        return self.prepare_finalize_type in [PplxPrepareAndFinalize, DeepEPLLPrepareAndFinalize]

    def is_batched_fused_experts(self):
        return self.fused_experts_type in [BatchedDeepGemmExperts, BatchedTritonExperts, BatchedExperts,
                                           BatchedTritonOrDeepGemmExperts]

    def is_fe_fp8_supported(self):
        return self.fused_experts_type in [BatchedDeepGemmExperts,
                                           BatchedTritonExperts,
                                           BatchedTritonOrDeepGemmExperts,
                                           CutlassExpertsFp8,
                                           DeepGemmExperts,
                                           TritonExperts,
                                           TritonOrDeepGemmExperts]

    def is_fe_block_fp8_supported(self):
        return self.fused_experts_type in [BatchedDeepGemmExperts,
                                           BatchedTritonOrDeepGemmExperts,
                                           DeepGemmExperts,
                                           TritonExperts,
                                           TritonOrDeepGemmExperts]

    def needs_deep_gemm(self):
        return self.fused_experts_type in [
                BatchedDeepGemmExperts,
                BatchedTritonOrDeepGemmExperts,
                DeepGemmExperts,
                TritonOrDeepGemmExperts,
        ]

    def needs_pplx(self):
        return self.prepare_finalize_type in [PplxPrepareAndFinalize]

    def needs_deep_ep(self):
        return self.prepare_finalize_type in [DeepEPHTPrepareAndFinalize, DeepEPLLPrepareAndFinalize]

    def all2all_backend(self):
        if self.needs_pplx():
            return "pplx"
        if self.prepare_finalize_type == DeepEPHTPrepareAndFinalize:
            return "deepep_high_throughput"
        if self.prepare_finalize_type == DeepEPHTPrepareAndFinalize:
            return "deepep_low_latency"
        return "naive"

    def is_valid(self):
        # Check prepare-finalize and fused-experts compatibility
        batched_pf = is_batched_prepare_finalize(self.prepare_finalize_type)
        batched_fe = is_batched_fused_experts(self.fused_experts_type)
        # both or neither should be batched.
        if batched_pf and not batched_fe:
            return False
        if batched_fe and not batched_pf:
            return False

        # Check fp8 support
        is_fp8 = self.dtype == torch.float8_e4m3fn
        if is_fp8 and not self.is_fe_fp8_supported():
            return False

        # Check fp8 block quanization support
        is_block_quatized = self.block_size is not None
        if is_block_quatized and not is_fp8:
            return False
        if is_block_quatized and not self.is_fe_block_fp8_supported():
            return False

        # Check dependencies
        if self.needs_deep_ep() and not has_deep_ep():
            return False
        if self.needs_deep_gemm() and not has_deep_gemm():
            return False
        if self.needs_pplx() and not has_pplx():
            return False

        return True

@dataclasses.dataclass
class WeightTensors:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: Optional[torch.Tensor]
    w2_scale: Optional[torch.Tensor]

    def to_current_device(self):
        self.w1 = self.w1.to(device=torch.cuda.current_device())
        self.w2 = self.w2.to(device=torch.cuda.current_device())
        is_quantized = self.w1.dtype == torch.float8_e4m3fn
        if is_quantized:
            self.w1_scale = self.w1_scale.to(device=torch.cuda.current_device())
            self.w2_scale = self.w2_scale.to(device=torch.cuda.current_device())

    def slice_weights(self, rank: int, num_local_experts: int):
        s = rank * num_local_experts
        e = s + num_local_experts
        self.w1 = self.w1[s:e, :, :]
        self.w2 = self.w2[s:e, :, :]
        is_quantized = self.w1.dtype == torch.float8_e4m3fn
        if is_quantized:
            self.w1_scale = self.w1_scale[s:e, :, :]
            self.w2_scale = self.w2_scale[s:e, :, :]

    @staticmethod
    def make(config: Config) -> "WeightTensors":
        if config.is_fp8_block_quantized():
            w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
                e=config.E,
                n=config.N,
                k=config.K,
                block_size=config.block_size,
            )
            return WeightTensors(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale)

        assert config.block_size is None
        # just make normal dtype weights
        w1, w2 = make_non_quant_weights(
            e=config.E, n=config.N, k=config.K, dtype=config.dtype)
        return WeightTensors(w1=w1, w2=w2, w1_scale=None, w2_scale=None)

@dataclass
class RankTensors:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: torch.Tensor

    @staticmethod
    def make(config: Config,
             pgi: ProcessGroupInfo):

        dtype = config.dtype
        topk, m, k = (config.topks, config.M, config.K)
        hidden_states = (
            torch.randn((m, k), device=torch.cuda.current_device(), dtype=dtype) / 10.0
        )

        num_local_experts, global_num_experts = config.num_local_experts, config.E
        score = torch.randn((m, global_num_experts),
                            device="cuda",
                            dtype=dtype)
        topk_ids_dtype = config.topk_ids_dtype()
        topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk, False)
        topk_ids = topk_ids.to(topk_ids_dtype)

        # distribute topk_ids evenly
        for mi in range(m):
            topk_ids[mi] = torch.randperm(config.E)[:topk]
        topk_ids = topk_ids.to(device=torch.cuda.current_device())

        expert_map = torch.full(
            (global_num_experts,), fill_value=-1, dtype=torch.int32
        )
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        expert_map = expert_map.to(
            device=torch.cuda.current_device(), dtype=torch.int32
        )

        return RankTensors(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )

def rank_worker(
        pgi: ProcessGroupInfo,
        vllm_config: VllmConfig,
        cpu_group,
        config: Config,
        weights: WeightTensors,
):
    current_platform.seed_everything(pgi.rank)

    pass

PREPARE_FINALIZE_TYPES = [
    PplxPrepareAndFinalize,
    DeepEPLLPrepareAndFinalize,
    DeepEPHTPrepareAndFinalize,
]
FUSED_EXPERT_TYPES = [
    BatchedDeepGemmExperts,
    BatchedTritonExperts,
    BatchedExperts,
    BatchedTritonOrDeepGemmExperts,
    CutlassExpertsFp8,
    DeepGemmExperts,
    TritonOrDeepGemmExperts,
    TritonExperts,
]

Ms = [1, 8, 16]
Ks = [1024, 4096, 7168] # hidden sizes
Ns = [1024]
TOPKs = [1, 4, 8]
Es=[32]
DTYPEs = [torch.bfloat16, torch.float8_e4m3fn]
BLOCK_SIZEs = [None, [128, 128]]
COMBINATIONs = product(PREPARE_FINALIZE_TYPES, FUSED_EXPERT_TYPES)

@pytest.mark.parametrize("k", Ks)
@pytest.mark.parametrize("n", Ns)
@pytest.mark.parametrize("e", Es)
@pytest.mark.parametrize("dtype", DTYPEs)
@pytest.mark.parametrize("block_size", BLOCK_SIZEs)
@pytest.mark.parametrize("combination", COMBINATIONs)
@pytest.mark.parametrize("world_size", [2])
def test_modular_kernel_combinations(
    k: int,
    n: int,
    e: int,
    dtype : torch.dtype,
    block_size: list[int], 
    combination: tuple[mk.FusedMoEPrepareAndFinalize, mk.FusedMoEPermuteExpertsUnpermute],
    world_size: int):

    config = Config(
        Ms = Ms,
        K = k,
        N = n,
        E = e,
        topks = TOPKs,
        dtype = dtype,
        block_size = block_size,
        prepare_finalize_type = combination[0],
        fused_experts_type = combination[1],
        world_size = world_size,
    )

    if not config.is_valid():
        pytest.skip(f"Tests config {config} is not valid. Skipping ...")

    weights: WeightTensors = WeightTensors.make(config)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    env_dict = {
        "VLLM_ALL2ALL_BACKEND": config.all2all_backend(),
        "VLLM_USE_DEEP_GEMM": str(int(config.needs_deep_gemm())),
    }
    parallel_launch(
        world_size,
        rank_worker,
        vllm_config,
        env_dict,
        config,
        weights
    )

if __name__ == "__main__":

    def to_pf_class_type(s: str) -> mk.FusedMoEPrepareAndFinalize:
        for pf in PREPARE_FINALIZE_TYPES:
            if pf.__name__ == s:
                return pf
        raise ValueError(f"Cannot find a PrepareFinalize type that matches {s}")

    def to_experts_class_type(s: str) -> mk.FusedMoEPermuteExpertsUnpermute:
        for fe in FUSED_EXPERT_TYPES:
            if fe.__name__ == s:
                return fe
        raise ValueError(f"Cannot find a FusedExperts type that matches {s}")

    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1")

    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Number of ranks that participate in all2all",
    )

    parser.add_argument(
        "--experts-per-rank", type=int, required=True, help="number of experts per rank"
    )

    parser.add_argument(
        "--pf-type",
        type=to_pf_class_type,
        required=True,
        help=(
            "Choose a PrepareFinalize Type : "
            f"{[x.__name__ for x in PREPARE_FINALIZE_TYPES]}"
        ),
    )

    parser.add_argument(
        "--experts-type",
        type=to_experts_class_type,
        required=True,
        help=(
            f"Choose a FusedExpert type : {[x.__name__ for x in FUSED_EXPERT_TYPES]}"
        ),
    )

    parser.add_argument(
        "--m-per-rank",
        nargs="+",
        type=int,
        default=[16],
        help="num tokens per rank",
    )

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    benchmark_config: BenchmarkConfig = BenchmarkConfig.from_args(args)
    print(benchmark_config)

    main(
        args.world_size,
        benchmark_config,
    )