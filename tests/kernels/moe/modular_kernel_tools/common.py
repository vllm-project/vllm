# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_test_weights, per_token_cast_to_fp8
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx

from .mk_objects import expert_info, make_fused_experts, prepare_finalize_info
from .parallel_utils import ProcessGroupInfo


def _describe_tensor(t: Optional[torch.Tensor], name: str) -> str:
    if t is None:
        return f"{name} : None"
    else:
        return f"{name} : {t.shape} {t.dtype} {t.device}"


@dataclass
class Config:
    Ms: Union[list[int], int]
    K: int
    N: int
    E: int
    topks: Union[list[int], int]
    dtype: torch.dtype
    quant_config: Optional[FusedMoEQuantConfig]

    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize
    fused_experts_type: mk.FusedMoEPermuteExpertsUnpermute

    fused_moe_chunk_size: Optional[int]
    world_size: int

    torch_trace_dir_path: Optional[str] = None

    def __post_init__(self):
        if self.quant_config is None:
            self.quant_config = FusedMoEQuantConfig()

    def describe(self) -> str:
        s = ""
        s += "== Config: \n"
        s += f" world_size={self.world_size} \n"
        s += f" PF={self.prepare_finalize_type.__name__} \n"
        s += f" FE={self.fused_experts_type.__name__} \n"
        s += f" topk={self.topks} \n"
        s += f" dtype={self.dtype} \n"
        s += f" fused_moe_chunk_size={self.fused_moe_chunk_size} \n"
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
    def quant_dtype(self) -> Union[torch.dtype, str, None]:
        return self.quant_config.quant_dtype

    @property
    def is_per_act_token_quant(self) -> bool:
        return self.quant_config.per_act_token_quant

    @property
    def is_per_tensor_act_quant(self) -> bool:
        return (not self.is_per_act_token_quant
                and self.quant_block_shape is None)

    @property
    def is_per_out_ch_quant(self) -> bool:
        return self.quant_config.per_out_ch_quant

    @property
    def quant_block_shape(self) -> Optional[list[int]]:
        return self.quant_config.block_shape

    @property
    def topk(self) -> int:
        assert isinstance(self.topks, int)
        return self.topks

    @property
    def num_local_experts(self) -> int:
        return self.E // self.world_size

    def make_env_data(self) -> tuple[VllmConfig, dict[Any, Any]]:
        """
        make env data for vllm launch.
        """
        vllm_config = VllmConfig()
        vllm_config.parallel_config.data_parallel_size = self.world_size
        vllm_config.parallel_config.enable_expert_parallel = True

        env_dict = {
            "VLLM_USE_DEEP_GEMM": str(int(self.needs_deep_gemm())),
        }

        backend = self.all2all_backend()
        if backend is not None:
            env_dict.update({"VLLM_ALL2ALL_BACKEND": backend})

        if self.fused_moe_chunk_size is not None:
            env_dict.update(
                {"VLLM_FUSED_MOE_CHUNK_SIZE": str(self.fused_moe_chunk_size)})

        return vllm_config, env_dict

    def is_fp8_block_quantized(self):
        return (self.quant_dtype == torch.float8_e4m3fn
                and self.quant_block_shape is not None)

    def is_batched_prepare_finalize(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return (mk.FusedMoEActivationFormat.BatchedExperts ==
                info.activation_format)

    def is_batched_fused_experts(self):
        info = expert_info(self.fused_experts_type)
        return (mk.FusedMoEActivationFormat.BatchedExperts ==
                info.activation_format)

    def is_standard_fused_experts(self):
        info = expert_info(self.fused_experts_type)
        return mk.FusedMoEActivationFormat.Standard == info.activation_format

    def is_fe_16bit_supported(self):
        info = expert_info(self.fused_experts_type)
        return torch.bfloat16 in info.supported_dtypes

    def is_fe_fp8_supported(self):
        info = expert_info(self.fused_experts_type)
        return torch.float8_e4m3fn in info.supported_dtypes

    def is_fe_block_fp8_supported(self):
        info = expert_info(self.fused_experts_type)
        return (torch.float8_e4m3fn in info.supported_dtypes
                and info.blocked_quantization_support)

    def is_fe_supports_chunking(self):
        info = expert_info(self.fused_experts_type)
        return info.supports_chunking

    def supports_expert_map(self):
        info = expert_info(self.fused_experts_type)
        return info.supports_expert_map

    def needs_deep_gemm(self):
        info = expert_info(self.fused_experts_type)
        return info.needs_deep_gemm

    def needs_pplx(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend == "pplx"

    def needs_deep_ep(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return (info.backend == "deepep_high_throughput"
                or info.backend == "deepep_low_latency")

    def all2all_backend(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend

    def needs_all2all(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend is not None and info.backend != "naive"

    def is_valid(self):
        # Check prepare-finalize and fused-experts compatibility
        if self.is_batched_prepare_finalize():
            if not self.is_batched_fused_experts():
                return False
        else:
            if not self.is_standard_fused_experts():
                return False

        use_chunking = self.fused_moe_chunk_size is not None
        if use_chunking and not self.is_fe_supports_chunking():
            return False

        # Check quantization sanity
        if (int(self.is_per_act_token_quant) +
                int(self.is_per_tensor_act_quant) +
                int(self.quant_block_shape is not None)) > 1:
            # invalid quant config
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


@dataclass
class WeightTensors:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: Optional[torch.Tensor]
    w2_scale: Optional[torch.Tensor]

    def describe(self):
        s = ""
        s += "== Weight Tensors: \n"
        s += f' - {_describe_tensor(self.w1, "w1")} \n'
        s += f' - {_describe_tensor(self.w2, "w2")} \n'
        s += f' - {_describe_tensor(self.w1_scale, "w1_scale")} \n'
        s += f' - {_describe_tensor(self.w2_scale, "w2_scale")} \n'
        return s

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
        _, w1, w1_scale, _, w2, w2_scale = make_test_weights(
            e=config.E,
            n=config.N,
            k=config.K,
            in_dtype=config.dtype,
            quant_dtype=config.quant_dtype,
            block_shape=config.quant_block_shape,
            per_act_token_quant=config.is_per_out_ch_quant,
        )
        return WeightTensors(w1=w1,
                             w2=w2,
                             w1_scale=w1_scale,
                             w2_scale=w2_scale)


@dataclass
class RankTensors:
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: Optional[torch.Tensor]

    quant_config: Optional[FusedMoEQuantConfig]

    def describe(self):
        s = ""
        s += "== Rank Tensors: \n"
        s += f' - {_describe_tensor(self.hidden_states, "HS")} \n'
        s += f' - {_describe_tensor(self.hidden_states_scale, "HS_scale")} \n'
        s += f' - {_describe_tensor(self.topk_weights, "topk_weights")} \n'
        s += f' - {_describe_tensor(self.topk_ids, "topk_ids")} \n'
        s += f' - {_describe_tensor(self.expert_map, "expert_map")} \n'
        return s

    @staticmethod
    def make_hidden_states(
            config: Config) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return hidden_states
        """
        m, k, dtype = (config.M, config.K, config.dtype)
        a = (torch.randn(
            (m, k), device=torch.cuda.current_device(), dtype=dtype) / 15.0)

        if config.quant_dtype is None:
            return a, None

        # We dequant and use that as hidden_states so the tests are stable.
        # quantizing and dequantizing yield slightly different results
        # depending on the hardware. Here we, quantize and dequantize
        # first - so further quantize and dequantize will yield the same
        # values.
        if config.is_per_tensor_act_quant:
            a_q, a_scales = ops.scaled_fp8_quant(
                a, use_per_token_if_dynamic=False)
            return a_q.float().mul(a_scales).to(dtype), a_scales

        if config.is_per_act_token_quant:
            a_q, a_scales = ops.scaled_fp8_quant(a,
                                                 use_per_token_if_dynamic=True)
            return a_q.float().mul(a_scales).to(dtype), None

        assert config.quant_block_shape is not None
        block_k = config.quant_block_shape[1]
        a_q, a_scales = per_token_cast_to_fp8(a, block_size=block_k)
        return a_q.float().view(
            (-1, block_k)).mul(a_scales.view(-1, 1)).view(m, k).to(dtype), None

    @staticmethod
    def make(config: Config, pgi: ProcessGroupInfo):

        dtype = config.dtype
        topk, m, _ = (config.topk, config.M, config.K)
        hidden_states, hidden_states_scale = RankTensors.make_hidden_states(
            config)

        num_local_experts, global_num_experts = (config.num_local_experts,
                                                 config.E)
        score = torch.randn((m, global_num_experts),
                            device="cuda",
                            dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk,
                                               False)

        # distribute topk_ids evenly
        for mi in range(m):
            topk_ids[mi] = torch.randperm(config.E)[:topk]
        topk_ids = topk_ids.to(device=torch.cuda.current_device())

        expert_map = None
        if config.world_size > 1 and config.supports_expert_map():
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
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            quant_config=config.quant_config,
        )


def reference_moe_impl(config: Config, weights: WeightTensors,
                       rank_tensors: RankTensors) -> torch.Tensor:

    return torch_experts(a=rank_tensors.hidden_states,
                         w1=weights.w1,
                         w2=weights.w2,
                         topk_weight=rank_tensors.topk_weights,
                         topk_ids=rank_tensors.topk_ids,
                         global_num_experts=config.E,
                         expert_map=None,
                         w1_scale=weights.w1_scale,
                         w2_scale=weights.w2_scale,
                         a1_scale=rank_tensors.hidden_states_scale,
                         quant_dtype=config.quant_dtype,
                         per_act_token_quant=config.is_per_act_token_quant,
                         block_shape=config.quant_block_shape,
                         apply_router_weights_on_input=config.topk == 1)


def make_modular_kernel(config: Config,
                        vllm_config: VllmConfig) -> mk.FusedMoEModularKernel:

    def next_power_of_2(x):
        import math
        if x == 0:
            return 1
        return 2**math.ceil(math.log2(x))

    # make moe config
    moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        dp_size_=get_dp_group().world_size,
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
        max_num_tokens=next_power_of_2(config.M),
    )

    print(config.prepare_finalize_type)

    # make modular kernel
    prepare_finalize = None
    if config.needs_all2all():
        prepare_finalize = FusedMoEMethodBase._maybe_make_prepare_finalize(moe)
        assert prepare_finalize is not None
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoEP()

    fused_experts = make_fused_experts(config.fused_experts_type, moe,
                                       prepare_finalize.num_dispatchers())

    modular_kernel = mk.FusedMoEModularKernel(
        prepare_finalize=prepare_finalize, fused_experts=fused_experts)

    return modular_kernel


def run_modular_kernel(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    config: Config,
    weights: WeightTensors,
    rank_tensors: RankTensors,
) -> torch.Tensor:
    assert isinstance(config.Ms, int)
    assert isinstance(config.topks, int)

    # weights for rank
    rank_weights = weights.slice_weights(pgi.rank, config.num_local_experts)

    mk = make_modular_kernel(config, vllm_config)

    mk_kwargs = {
        "hidden_states":
        rank_tensors.hidden_states.clone(
        ),  # impls might update the tensor in place
        "w1":
        rank_weights.w1,
        "w2":
        rank_weights.w2,
        "topk_weights":
        rank_tensors.topk_weights,
        "topk_ids":
        rank_tensors.topk_ids.to(mk.prepare_finalize.topk_indices_dtype()),
        "expert_map":
        rank_tensors.expert_map,
        "w1_scale":
        rank_weights.w1_scale,
        "w2_scale":
        rank_weights.w2_scale,
        "a1_scale":
        rank_tensors.hidden_states_scale,
        "global_num_experts":
        config.E,
        "apply_router_weight_on_input":
        config.topk == 1,
    }
    out = mk.forward(**mk_kwargs)

    return out
