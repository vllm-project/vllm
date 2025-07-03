# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import dataclasses
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Optional, Union

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, current_platform
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
# Fused experts imports
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (  # noqa: E501
    BatchedTritonOrDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
    DeepEPHTPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
    DeepEPLLPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts, NaiveBatchedExperts)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.layer import (FusedMoEMethodBase,
                                                        TritonExperts)
from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
    PplxPrepareAndFinalize)
# PrepareFinalize imports
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts)
from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx

from .parallel_utils import ProcessGroupInfo, parallel_launch_with_config

# TODO (varun): These requirements are very strict and could be relaxed.
meets_package_requirements = pytest.mark.skipif(
    not (has_deep_ep() and has_pplx() and has_deep_gemm()),
    reason="Requires deep_ep & deep_gemm & pplx packages",
)


def per_token_cast_to_fp8(
        x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (block_size - (n % block_size)) % block_size
    x = torch.nn.functional.pad(x,
                                (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, block_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(
        x: torch.Tensor, block_size_k: int,
        block_size_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (
            int(math.ceil(m / block_size_k)) * block_size_k,
            int(math.ceil(n / block_size_n)) * block_size_n,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_size_k,
                           x_padded.size(1) // block_size_k, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def make_non_quant_weights(
    e: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2
    """
    device = torch.cuda.current_device()
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
    return w1, w2


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1_scale, w2_scale
    """
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16, w2_bf16 = make_non_quant_weights(e, n, k, dtype)
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn, device=device)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1),
                       device=device,
                       dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2),
                       device=device,
                       dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + (block_n - 1)) // block_n,
                          (k + (block_k - 1)) // block_k)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i],
                                               block_size_k=block_k,
                                               block_size_n=block_n)
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i],
                                               block_size_k=block_k,
                                               block_size_n=block_n)

    return w1, w2, w1_s, w2_s


def make_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    per_out_channel_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return w1, w2, w1_scale, w2_scale
    """
    q_dtype = torch.float8_e4m3fn

    w1, w2 = make_non_quant_weights(e, n, k, dtype=torch.bfloat16)

    # w1 -> w1_q, w2 -> w2_q
    w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
    w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

    n_b_scales = 2 * n if per_out_channel_quant else 1
    k_b_scales = k if per_out_channel_quant else 1
    w1_scale = torch.empty((e, n_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)

    for expert in range(e):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=per_out_channel_quant)
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=per_out_channel_quant)
    return w1_q, w2_q, w1_scale, w2_scale


@dataclass
class Config:
    Ms: Union[list[int], int]
    K: int
    N: int
    E: int
    topks: Union[list[int], int]
    dtype: torch.dtype
    quant_config: FusedMoEQuantConfig

    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize
    fused_experts_type: mk.FusedMoEPermuteExpertsUnpermute

    fused_moe_chunk_size: Optional[int]
    world_size: int

    torch_trace_dir_path: Optional[str] = None

    def describe(self) -> str:
        s = ""
        s += "== Config: \n"
        s += f" world_size={self.world_size} \n"
        s += f" PF={self.prepare_finalize_type.__name__} \n"
        s += f" FE={self.fused_experts_type.__name__} \n"
        s += f" topk={self.topks} \n"
        s += f" dtype={self.dtype} \n"
        s += " Quant: \n"
        s += f" fused_moe_chunk_size={self.fused_moe_chunk_size} \n "
        if self.quant_config is not None:
            s += f"     q_dtype={self.quant_dtype} \n"
            s += f"     q_block_shape={self.quant_block_shape} \n"
            s += f"     q_per_out_ch_quant={self.is_per_out_ch_quant} \n"
            s += f"     q_per_act_token={self.is_per_act_token_quant} \n"
        else:
            s += "     quant=None \n"
        return s

    @property
    def M(self) -> int:
        assert isinstance(self.Ms, int)
        return self.Ms

    @property
    def quant_dtype(self) -> Optional[torch.dtype]:
        if self.quant_config is None:
            return None
        return self.quant_config.quant_dtype

    @property
    def is_per_act_token_quant(self) -> bool:
        if self.quant_config is None:
            return False
        return self.quant_config.per_act_token_quant

    @property
    def is_per_out_ch_quant(self) -> bool:
        if self.quant_config is None:
            return False
        return self.quant_config.per_out_ch_quant

    @property
    def quant_block_shape(self) -> Optional[list[int]]:
        if self.quant_config is None:
            return None
        return self.quant_config.block_shape

    @property
    def topk(self) -> int:
        assert isinstance(self.topks, int)
        return self.topks

    @property
    def topk_ids_dtype(self) -> Optional[torch.dtype]:
        topk_ids_dtype = None
        if self.prepare_finalize_type == PplxPrepareAndFinalize:
            topk_ids_dtype = torch.uint32
        elif self.prepare_finalize_type in [
                DeepEPHTPrepareAndFinalize, DeepEPLLPrepareAndFinalize
        ]:
            topk_ids_dtype = torch.int64
        return topk_ids_dtype

    @property
    def num_local_experts(self) -> int:
        return self.E // self.world_size

    def is_fp8_block_quantized(self):
        return (self.quant_dtype == torch.float8_e4m3fn
                and self.quant_block_shape is not None)

    def is_batched_prepare_finalize(self):
        return self.prepare_finalize_type in [
            PplxPrepareAndFinalize, DeepEPLLPrepareAndFinalize
        ]

    def is_batched_fused_experts(self):
        return self.fused_experts_type in [
            CutlassExpertsFp8, BatchedDeepGemmExperts, BatchedTritonExperts,
            NaiveBatchedExperts, BatchedTritonOrDeepGemmExperts
        ]

    def is_standard_fused_experts(self):
        return self.fused_experts_type in [
            CutlassExpertsFp8, DeepGemmExperts, TritonOrDeepGemmExperts,
            TritonExperts
        ]

    def is_fe_16bit_supported(self):
        return self.fused_experts_type in [
            BatchedTritonExperts, NaiveBatchedExperts, TritonExperts
        ]

    def is_fe_fp8_supported(self):
        return self.fused_experts_type in [
            BatchedDeepGemmExperts, BatchedTritonExperts,
            BatchedTritonOrDeepGemmExperts, CutlassExpertsFp8, DeepGemmExperts,
            TritonExperts, TritonOrDeepGemmExperts
        ]

    def is_fe_block_fp8_supported(self):
        # TODO (varun) : Check if cutlass supports block-quantization
        return self.fused_experts_type in [
            BatchedDeepGemmExperts, BatchedTritonOrDeepGemmExperts,
            DeepGemmExperts, TritonExperts, TritonOrDeepGemmExperts
        ]

    def is_fe_supports_chunking(self):
        return self.fused_experts_type in [
            CutlassExpertsFp8, DeepGemmExperts, TritonOrDeepGemmExperts,
            TritonExperts
        ]

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
        return self.prepare_finalize_type in [
            DeepEPHTPrepareAndFinalize, DeepEPLLPrepareAndFinalize
        ]

    def all2all_backend(self):
        if self.needs_pplx():
            return "pplx"
        if self.prepare_finalize_type == DeepEPHTPrepareAndFinalize:
            return "deepep_high_throughput"
        if self.prepare_finalize_type == DeepEPLLPrepareAndFinalize:
            return "deepep_low_latency"
        return "naive"

    def needs_all2all(self):
        return self.prepare_finalize_type in [
            PplxPrepareAndFinalize, DeepEPHTPrepareAndFinalize,
            DeepEPLLPrepareAndFinalize
        ]

    def is_valid(self):
        # Check prepare-finalize and fused-experts compatibility
        if self.is_batched_prepare_finalize():
            if not self.is_batched_fused_experts():
                return False
        else:
            if not self.is_standard_fused_experts():
                return False

        # check bf16 / fp16 support
        is_16bit = (self.dtype.itemsize == 2 and self.quant_dtype is None)
        if is_16bit and not self.is_fe_16bit_supported():
            return False

        # Check fp8 support
        is_fp8 = self.quant_dtype == torch.float8_e4m3fn
        if is_fp8 and not self.is_fe_fp8_supported():
            return False

        # Check fp8 block quanization support
        is_block_quatized = self.quant_block_shape is not None
        if is_block_quatized and not is_fp8:
            return False
        if is_block_quatized and not self.is_fe_block_fp8_supported():
            return False

        # deep_gemm only works with block-quantized
        if self.needs_deep_gemm() and not is_block_quatized:
            return False

        # Check dependencies
        if self.needs_deep_ep() and not has_deep_ep():
            return False
        if self.needs_deep_gemm() and not has_deep_gemm():
            return False
        if self.needs_pplx() and not has_pplx():  # noqa: SIM103
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
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            self.w1_scale = self.w1_scale.to(
                device=torch.cuda.current_device())
            self.w2_scale = self.w2_scale.to(
                device=torch.cuda.current_device())

    def slice_weights(self, rank: int,
                      num_local_experts: int) -> "WeightTensors":
        s = rank * num_local_experts
        e = s + num_local_experts
        w1 = self.w1[s:e, :, :]
        w2 = self.w2[s:e, :, :]
        is_quantized = self.w1.dtype == torch.float8_e4m3fn
        w1_scale, w2_scale = (None, None)
        if is_quantized:
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            w1_scale = self.w1_scale[s:e, :, :]
            w2_scale = self.w2_scale[s:e, :, :]
        return WeightTensors(w1, w2, w1_scale, w2_scale)

    @staticmethod
    def make(config: Config) -> "WeightTensors":
        if config.quant_dtype is None:
            # just make normal dtype weights
            w1, w2 = make_non_quant_weights(e=config.E,
                                            n=config.N,
                                            k=config.K,
                                            dtype=config.dtype)
            return WeightTensors(w1=w1, w2=w2, w1_scale=None, w2_scale=None)

        assert config.quant_dtype == torch.float8_e4m3fn
        if not config.is_fp8_block_quantized():
            w1, w2, w1_scale, w2_scale = make_quant_fp8_weights(
                e=config.E,
                n=config.N,
                k=config.K,
                per_out_channel_quant=config.is_per_out_ch_quant,
            )
            return WeightTensors(w1=w1,
                                 w2=w2,
                                 w1_scale=w1_scale,
                                 w2_scale=w2_scale)

        assert config.quant_block_shape is not None
        w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
            e=config.E,
            n=config.N,
            k=config.K,
            block_size=config.quant_block_shape,
        )
        return WeightTensors(w1=w1,
                             w2=w2,
                             w1_scale=w1_scale,
                             w2_scale=w2_scale)


@dataclass
class RankTensors:
    hidden_states: torch.Tensor

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: torch.Tensor

    quant_config: FusedMoEQuantConfig

    def apply_topk_weights_to_input(self) -> torch.Tensor:
        num_topk = self.topk_ids.size(1)
        orig_dtype = self.hidden_states.dtype
        assert num_topk == 1
        a = self.hidden_states.clone()
        return a.float().mul(self.topk_weights.view(-1, 1)).to(orig_dtype)

    @staticmethod
    def make_hidden_states(config: Config) -> torch.Tensor:
        """
        Return hidden_states
        """
        m, k, dtype = (config.M, config.K, config.dtype)
        a = (torch.randn(
            (m, k), device=torch.cuda.current_device(), dtype=dtype) / 10.0)

        if config.quant_dtype is None:
            return a

        # We dequant and use that as hidden_states so the tests are stable.
        # quantizing and dequantizing yield slightly different results
        # depending on the hardware. Here we, quantize and dequantize
        # first - so further quantize and dequantize will yeild the same
        # values.
        a_q, a_scales = (None, None)
        if not config.is_fp8_block_quantized():
            a_q, a_scales = ops.scaled_fp8_quant(
                a, use_per_token_if_dynamic=config.is_per_act_token_quant)
            return a_q.float().mul(a_scales).to(dtype)

        assert config.quant_block_shape is not None
        block_k = config.quant_block_shape[1]
        a_q, a_scales = per_token_cast_to_fp8(a, block_size=block_k)
        return a_q.float().view(
            (-1, block_k)).mul(a_scales.view(-1, 1)).view(m, k).to(dtype)

    @staticmethod
    def make(config: Config, pgi: ProcessGroupInfo):

        dtype = config.dtype
        topk, m, _ = (config.topk, config.M, config.K)
        hidden_states = RankTensors.make_hidden_states(config)

        num_local_experts, global_num_experts = (config.num_local_experts,
                                                 config.E)
        score = torch.randn((m, global_num_experts),
                            device="cuda",
                            dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk,
                                               False)
        topk_ids = topk_ids.to(config.topk_ids_dtype)

        # distribute topk_ids evenly
        for mi in range(m):
            topk_ids[mi] = torch.randperm(config.E)[:topk]
        topk_ids = topk_ids.to(device=torch.cuda.current_device())

        expert_map = torch.full((global_num_experts, ),
                                fill_value=-1,
                                dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        expert_map = expert_map.to(device=torch.cuda.current_device(),
                                   dtype=torch.int32)

        return RankTensors(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            quant_config=config.quant_config,
        )


def make_fused_experts(
        config: Config,
        moe: FusedMoEConfig) -> mk.FusedMoEPermuteExpertsUnpermute:

    use_fp8 = config.quant_dtype == torch.float8_e4m3fn
    batch_kwargs = {
        "max_num_tokens": moe.max_num_tokens,
        "world_size": config.world_size,
        "dp_size": moe.dp_size,
    }
    quant_kwargs = {
        "use_fp8_w8a8": use_fp8,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "block_shape": config.quant_block_shape,
        "per_act_token_quant": config.is_per_act_token_quant,
    }
    deepgemm_kwargs = {"allow_deep_gemm": has_deep_gemm()}

    if config.fused_experts_type == BatchedDeepGemmExperts:
        kwargs = batch_kwargs | {
            "block_shape": config.quant_block_shape,
        }
        print(f"Making BatchedDeepGemmExperts {kwargs} ...")
        experts = BatchedDeepGemmExperts(**kwargs)
    elif config.fused_experts_type == BatchedTritonExperts:
        kwargs = batch_kwargs | quant_kwargs
        print(f"Making BatchedTritonExperts {kwargs} ...")
        experts = BatchedTritonExperts(**kwargs)
    elif config.fused_experts_type == BatchedTritonOrDeepGemmExperts:
        kwargs = batch_kwargs | quant_kwargs | deepgemm_kwargs
        print(f"Making BatchedTritonOrDeepGemmExperts {kwargs} ...")
        experts = BatchedTritonOrDeepGemmExperts(**kwargs)
    elif config.fused_experts_type == DeepGemmExperts:
        print("Making DeepGemmExperts () ...")
        experts = DeepGemmExperts()
    elif config.fused_experts_type == TritonExperts:
        kwargs = quant_kwargs
        print(f"Making TritonExperts {kwargs} ...")
        experts = TritonExperts(**kwargs)
    elif config.fused_experts_type == TritonOrDeepGemmExperts:
        kwargs = quant_kwargs | deepgemm_kwargs
        print(f"Making TritonOrDeepGemmExperts {kwargs} ...")
        experts = TritonOrDeepGemmExperts(**kwargs)
    elif config.fused_experts_type == NaiveBatchedExperts:
        kwargs = batch_kwargs | quant_kwargs
        print(f"Making NaiveBatchedExperts {kwargs} ...")
        experts = NaiveBatchedExperts(**kwargs)
    elif config.fused_experts_type == CutlassExpertsFp8:
        use_batched_format = config.is_batched_prepare_finalize()
        num_experts = (moe.num_local_experts
                       if use_batched_format else moe.num_experts)
        kwargs = {
            "max_experts_per_worker": num_experts,
            "out_dtype": moe.in_dtype,
            "per_act_token_quant": config.is_per_act_token_quant,
            "per_out_ch_quant": config.is_per_out_ch_quant,
            "block_shape": config.quant_block_shape,
            "use_batched_format": use_batched_format
        }
        print(f"Making CutlassExpertsFp8 {kwargs} ...")
        experts = CutlassExpertsFp8(**kwargs)

    return experts


def do_profile(fn: Callable,
               fn_kwargs: dict[Any, Any],
               pgi: ProcessGroupInfo,
               config: Config,
               num_warmups: int = 5):

    for _ in range(num_warmups):
        fn(**fn_kwargs)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=True,
    ) as tprof:
        fn(**fn_kwargs)
        torch.cuda.synchronize(torch.cuda.current_device())

    # TODO (varun): Add a descriptive trace file name
    tprof.export_chrome_trace(
        f"{config.torch_trace_dir_path}/m{config.M}_{pgi.rank}_trace.json")


def do_modular_kernel(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    config: Config,
    weights: WeightTensors,
    rank_tensors: RankTensors,
):
    assert isinstance(config.Ms, int)
    assert isinstance(config.topks, int)

    # make moe config
    moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        dp_size_=get_dp_group().world_size,
        world_size_=config.world_size,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    moe = FusedMoEConfig(
        num_experts=config.E,
        experts_per_token=config.topk,
        hidden_dim=config.K,
        num_local_experts=config.num_local_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=config.dtype,
        quant_config=config.quant_config,
        max_num_tokens=config.M,
    )

    # make modular kernel
    prepare_finalize = None
    if config.needs_all2all():
        prepare_finalize = FusedMoEMethodBase.maybe_make_prepare_finalize(moe)
        assert prepare_finalize is not None
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoEP()

    fused_experts = make_fused_experts(config, moe)

    modular_kernel = mk.FusedMoEModularKernel(
        prepare_finalize=prepare_finalize, fused_experts=fused_experts)

    apply_router_weight_on_input = config.topk == 1
    hidden_states = (rank_tensors.apply_topk_weights_to_input()
                     if apply_router_weight_on_input else
                     rank_tensors.hidden_states)

    mk_kwargs = {
        "hidden_states": hidden_states,
        "w1": weights.w1,
        "w2": weights.w2,
        "topk_weights": rank_tensors.topk_weights,
        "topk_ids": rank_tensors.topk_ids,
        "expert_map": rank_tensors.expert_map,
        "w1_scale": weights.w1_scale,
        "w2_scale": weights.w2_scale,
        "global_num_experts": config.E,
        "apply_router_weight_on_input": apply_router_weight_on_input,
    }
    modular_kernel.forward(**mk_kwargs)

    if config.torch_trace_dir_path is not None:
        do_profile(modular_kernel.forward, mk_kwargs, pgi, config)


def rank_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
):
    current_platform.seed_everything(pgi.rank)

    # sanity check
    from vllm import envs
    if config.fused_moe_chunk_size is not None:
        assert (config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE)

    # get weights to this device
    weights.to_current_device()

    Ms = config.Ms
    assert isinstance(Ms, list)
    TOPKs = config.topks
    assert isinstance(TOPKs, list)

    for m, topk in product(Ms, TOPKs):
        print(f"Running m={m}, topk={topk} ...")
        # override m and topk
        cfgx = copy.deepcopy(config)
        cfgx.Ms = m
        cfgx.topks = topk

        # inputs for rank
        ir = RankTensors.make(cfgx, pgi)
        # weights for rank
        wr = weights.slice_weights(pgi.rank, cfgx.num_local_experts)

        do_modular_kernel(pgi, vllm_config, cfgx, wr, ir)


def run(ms: list[int],
        k: int,
        n: int,
        e: int,
        topks: list[int],
        dtype: torch.dtype,
        quant_config: Optional[FusedMoEQuantConfig],
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int],
        world_size: int,
        torch_trace_dir_path: Optional[str] = None):

    config = Config(
        Ms=ms,
        K=k,
        N=n,
        E=e,
        topks=topks,
        dtype=dtype,
        quant_config=quant_config,
        prepare_finalize_type=combination[0],
        fused_experts_type=combination[1],
        fused_moe_chunk_size=fused_moe_chunk_size,
        world_size=world_size,
        torch_trace_dir_path=torch_trace_dir_path,
    )

    if not config.is_valid():
        pytest.skip(f"Tests config {config} is not valid. Skipping ...")

    print(f"Testing config \n{config.describe()} ...")

    weights: WeightTensors = WeightTensors.make(config)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    env_dict = {
        "VLLM_ALL2ALL_BACKEND": config.all2all_backend(),
        "VLLM_USE_DEEP_GEMM": str(int(config.needs_deep_gemm())),
    }

    if fused_moe_chunk_size is not None:
        env_dict.update(
            {"VLLM_FUSED_MOE_CHUNK_SIZE": str(fused_moe_chunk_size)})

    parallel_launch_with_config(world_size, rank_worker, vllm_config, env_dict,
                                config, weights)


PREPARE_FINALIZE_TYPES = [
    PplxPrepareAndFinalize,
    DeepEPLLPrepareAndFinalize,
    DeepEPHTPrepareAndFinalize,
]

FUSED_EXPERT_TYPES = [
    BatchedDeepGemmExperts,
    BatchedTritonExperts,
    NaiveBatchedExperts,
    BatchedTritonOrDeepGemmExperts,
    CutlassExpertsFp8,
    DeepGemmExperts,
    TritonOrDeepGemmExperts,
    TritonExperts,
]

Ms = [64]
Ks = [7168]  # hidden sizes
Ns = [2048]
TOPKs = [4, 1]
Es = [32]
DTYPEs = [torch.bfloat16]
BLOCK_SIZEs = [None, [128, 128]]
QUANTCONFIGs = [
    None,
    # per-channel / per-column weights and per-tensor activations
    FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn,
                        per_out_ch_quant=True,
                        per_act_token_quant=False,
                        block_shape=None),
    # per-channel / per-column weights and per-token activations
    FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn,
                        per_out_ch_quant=True,
                        per_act_token_quant=True,
                        block_shape=None),
    # per-tensor weights and per-tensor activations
    FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn,
                        per_out_ch_quant=False,
                        per_act_token_quant=False,
                        block_shape=None),
    # per-tensor weights and per-token activations
    FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn,
                        per_out_ch_quant=False,
                        per_act_token_quant=True,
                        block_shape=None),
    # block-quantized weights and 128 block per-token activations
    FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn,
                        per_out_ch_quant=False,
                        per_act_token_quant=False,
                        block_shape=[128, 128]),
    # TODO (varun) : Should we test the following combinations ?
    # block-quantized weights and per-token activations
    # block-quantized weights and per-tensor activations
]
FUSED_MOE_CHUNK_SIZEs = [None, 64]


@pytest.mark.parametrize("k", Ks)
@pytest.mark.parametrize("n", Ns)
@pytest.mark.parametrize("e", Es)
@pytest.mark.parametrize("dtype", DTYPEs)
@pytest.mark.parametrize("quant_config", QUANTCONFIGs)
@pytest.mark.parametrize("combination",
                         product(PREPARE_FINALIZE_TYPES, FUSED_EXPERT_TYPES))
@pytest.mark.parametrize("fused_moe_chunk_size", FUSED_MOE_CHUNK_SIZEs)
@pytest.mark.parametrize("world_size", [2])
@meets_package_requirements
def test_modular_kernel_combinations_multigpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int):
    run(Ms, k, n, e, TOPKs, dtype, quant_config, combination,
        fused_moe_chunk_size, world_size)


SINGLE_GPU_PREPARE_FINALIZE_TYPES = [MoEPrepareAndFinalizeNoEP]


@pytest.mark.parametrize("k", Ks)
@pytest.mark.parametrize("n", Ns)
@pytest.mark.parametrize("e", Es)
@pytest.mark.parametrize("dtype", DTYPEs)
@pytest.mark.parametrize("quant_config", QUANTCONFIGs)
@pytest.mark.parametrize(
    "combination",
    product(SINGLE_GPU_PREPARE_FINALIZE_TYPES, FUSED_EXPERT_TYPES))
@pytest.mark.parametrize("fused_moe_chunk_size", FUSED_MOE_CHUNK_SIZEs)
@pytest.mark.parametrize("world_size", [1])
@meets_package_requirements
def test_modular_kernel_combinations_singlegpu(
        k: int, n: int, e: int, dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        combination: tuple[mk.FusedMoEPrepareAndFinalize,
                           mk.FusedMoEPermuteExpertsUnpermute],
        fused_moe_chunk_size: Optional[int], world_size: int):
    run(Ms, k, n, e, TOPKs, dtype, quant_config, combination,
        fused_moe_chunk_size, world_size)


ALL_PREPARE_FINALIZE_TYPES = (PREPARE_FINALIZE_TYPES +
                              SINGLE_GPU_PREPARE_FINALIZE_TYPES)

if __name__ == '__main__':
    import argparse

    def to_pf_class_type(s: str) -> mk.FusedMoEPrepareAndFinalize:
        for pf in ALL_PREPARE_FINALIZE_TYPES:
            if pf.__name__ == s:
                return pf
        raise ValueError(
            f"Cannot find a PrepareFinalize type that matches {s}")

    def to_experts_class_type(s: str) -> mk.FusedMoEPermuteExpertsUnpermute:
        for fe in FUSED_EXPERT_TYPES:
            if fe.__name__ == s:
                return fe
        raise ValueError(f"Cannot find a FusedExperts type that matches {s}")

    parser = argparse.ArgumentParser(description=(
        "Run a single modular kernel combination \n"
        'Example : python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts'  # noqa :E501
    ))

    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of ranks that participate in all2all",
    )
    parser.add_argument(
        "--pf-type",
        type=to_pf_class_type,
        required=True,
        help=("Choose a PrepareFinalize Type : "
              f"{[x.__name__ for x in ALL_PREPARE_FINALIZE_TYPES]}"),
    )
    parser.add_argument(
        "--experts-type",
        type=to_experts_class_type,
        required=True,
        help=(f"Choose a FusedExpert type : "
              f"{[x.__name__ for x in FUSED_EXPERT_TYPES]}"),
    )
    parser.add_argument(
        "-m",
        nargs="+",
        type=int,
        default=[64],
        help="num tokens per rank",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=7168,
        help="hidden-size",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1024,
        help="N dimension of the first fused-moe matmul",
    )
    parser.add_argument("--num-experts",
                        type=int,
                        default=32,
                        help="Global num experts")
    parser.add_argument("--topk",
                        nargs="+",
                        type=int,
                        default=[4],
                        help="num topk")
    parser.add_argument(
        "--fused-moe-chunk-size",
        nargs="+",
        type=int,
        help="Fused moe chunk size used for the non-batched fused experts impl."
    )

    # Quant args
    parser.add_argument("--quant-dtype",
                        type=torch.dtype,
                        help="Quant datatype")
    parser.add_argument("--per-act-token-quant",
                        action='store_true',
                        help="Use per token act quant")
    parser.add_argument("--per-out-ch-quant",
                        action="store_true",
                        help="Per out channel quant for weight quantization.")
    parser.add_argument("--block-shape",
                        nargs="+",
                        type=int,
                        help="Quantization block shape")

    # Torch trace profile generation args
    parser.add_argument("--torch-trace-dir-path",
                        type=str,
                        default=None,
                        help="Get torch trace for single execution")

    args = parser.parse_args()

    quant_config = None
    if args.quant_dtype is not None:
        assert args.quant_dtype == torch.float8_e4m3fn
        if args.block_shape is not None:
            assert len(args.block_shape) == 2, (
                f"block shape must have 2 elements. got {args.block_shape}")
        quant_config = FusedMoEQuantConfig(
            quant_dtype=args.quant_dtype,
            per_act_token_quant=args.per_act_token_quant,
            per_out_ch_quant=args.per_out_ch_quant,
            block_shape=args.block_shape)

    if args.torch_trace_dir_path is not None:
        from pathlib import Path
        assert Path(args.torch_trace_dir_path).is_dir(), (
            f"Please create {args.torch_trace_dir_path}")

    run(
        ms=args.m,
        k=args.k,
        n=args.n,
        e=args.num_experts,
        topks=args.topk,
        dtype=torch.bfloat16,  # hard-code
        quant_config=quant_config,
        combination=(args.pf_type, args.experts_type),
        fused_moe_chunk_size=args.fused_moe_chunk_size,
        world_size=args.world_size,
        torch_trace_dir_path=args.torch_trace_dir_path)
