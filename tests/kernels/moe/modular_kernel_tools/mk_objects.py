# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Union

import torch

# Fused experts and PrepareFinalize imports
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (  # noqa: E501
    BatchedTritonOrDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                         FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts, NaiveBatchedExperts)
from vllm.model_executor.layers.fused_moe.layer import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported)
from vllm.utils import has_deep_ep, has_deep_gemm, has_pplx
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe


@dataclass
class PrepareFinalizeInfo:
    activation_format: mk.FusedMoEActivationFormat
    supported_dtypes: list[Union[torch.dtype, str]]
    blocked_quantization_support: bool
    backend: Optional[str]


@dataclass
class ExpertInfo:
    activation_format: mk.FusedMoEActivationFormat
    supported_dtypes: list[Union[torch.dtype, str]]
    blocked_quantization_support: bool
    supports_chunking: bool
    supports_expert_map: bool
    needs_matching_quant: bool = False
    needs_deep_gemm: bool = False


PREPARE_FINALIZE_INFO: dict[mk.FusedMoEPrepareAndFinalize,
                            PrepareFinalizeInfo] = {}
EXPERT_INFO: dict[mk.FusedMoEPermuteExpertsUnpermute, ExpertInfo] = {}
MK_ALL_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_MULTI_GPU_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_FUSED_EXPERT_TYPES: list[mk.FusedMoEPermuteExpertsUnpermute] = []

standard_format = mk.FusedMoEActivationFormat.Standard
batched_format = mk.FusedMoEActivationFormat.BatchedExperts
common_float_types: list[Union[torch.dtype, str]] = [
    torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32
]
common_float_and_int_types = common_float_types + [torch.int8]
nv_fp4_types = ["nvfp4"]
fp8_types = [torch.float8_e4m3fn]


def register_prepare_and_finalize(
    kind,
    activation_format: mk.FusedMoEActivationFormat,
    supported_dtypes: list[Union[torch.dtype, str]],
    blocked_quantization_support: bool,
    backend: Optional[str],
):
    global PREPARE_FINALIZE_INFO
    global MK_ALL_PREPARE_FINALIZE_TYPES
    global MK_MULTI_GPU_PREPARE_FINALIZE_TYPES
    global MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES
    assert kind not in PREPARE_FINALIZE_INFO

    PREPARE_FINALIZE_INFO[kind] = PrepareFinalizeInfo(
        activation_format,
        supported_dtypes,
        blocked_quantization_support,
        backend,
    )
    MK_ALL_PREPARE_FINALIZE_TYPES.append(kind)
    if backend is not None:
        MK_MULTI_GPU_PREPARE_FINALIZE_TYPES.append(kind)
    else:
        MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES.append(kind)


def register_experts(
    kind,
    activation_format: mk.FusedMoEActivationFormat,
    supported_dtypes: list[Union[torch.dtype, str]],
    blocked_quantization_support: bool,
    supports_chunking: bool,
    supports_expert_map: bool,
    needs_matching_quant: bool = False,
    needs_deep_gemm: bool = False,
):
    global EXPERT_INFO
    global MK_FUSED_EXPERT_TYPES
    assert kind not in EXPERT_INFO

    EXPERT_INFO[kind] = ExpertInfo(
        activation_format,
        supported_dtypes,
        blocked_quantization_support,
        supports_chunking,
        supports_expert_map,
        needs_matching_quant,
        needs_deep_gemm,
    )

    MK_FUSED_EXPERT_TYPES.append(kind)


def prepare_finalize_info(kind) -> PrepareFinalizeInfo:
    info = PREPARE_FINALIZE_INFO.get(kind)
    assert info is not None
    return info


def expert_info(kind) -> ExpertInfo:
    info = EXPERT_INFO.get(kind)
    assert info is not None
    return info


register_prepare_and_finalize(
    MoEPrepareAndFinalizeNoEP,
    standard_format,
    common_float_types,
    blocked_quantization_support=True,
    backend=None,  # naive?
)

register_experts(
    BatchedTritonExperts,
    batched_format,
    common_float_types,
    blocked_quantization_support=True,
    supports_chunking=False,
    supports_expert_map=False,
    needs_matching_quant=True,
)

register_experts(
    TritonExperts,
    standard_format,
    common_float_and_int_types,
    blocked_quantization_support=True,
    supports_chunking=True,
    supports_expert_map=True,
    needs_matching_quant=True,
)

register_experts(
    NaiveBatchedExperts,
    batched_format,
    common_float_and_int_types,
    blocked_quantization_support=True,
    supports_chunking=False,
    supports_expert_map=True,
)

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (  # noqa: E501
        DeepEPHTPrepareAndFinalize)
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
        DeepEPLLPrepareAndFinalize)

    register_prepare_and_finalize(
        DeepEPHTPrepareAndFinalize,
        standard_format,
        common_float_types,
        blocked_quantization_support=True,
        backend="deepep_high_throughput",
    )

    register_prepare_and_finalize(
        DeepEPLLPrepareAndFinalize,
        batched_format,
        common_float_types,
        blocked_quantization_support=True,
        backend="deepep_low_latency",
    )

if has_pplx():
    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize)
    register_prepare_and_finalize(
        PplxPrepareAndFinalize,
        batched_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        backend="pplx",
    )

if False and has_flashinfer_cutlass_fused_moe():
    from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (  # noqa: E501
        FlashInferExperts)
    from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
        FlashInferCutlassMoEPrepareAndFinalize)

    register_prepare_and_finalize(
        FlashInferCutlassMoEPrepareAndFinalize,
        standard_format,
        nv_fp4_types,
        blocked_quantization_support=True,  # ?
        backend=None,
    )

    register_experts(
        FlashInferExperts,
        standard_format,
        nv_fp4_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=False,
    )

if has_deep_gemm():
    register_experts(
        BatchedDeepGemmExperts,
        batched_format,
        fp8_types,
        blocked_quantization_support=True,
        supports_chunking=False,
        supports_expert_map=False,
        needs_matching_quant=False,
        needs_deep_gemm=True,
    )
    register_experts(
        DeepGemmExperts,
        standard_format,
        fp8_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=True,
        needs_matching_quant=False,
        needs_deep_gemm=True,
    ),
    register_experts(
        BatchedTritonOrDeepGemmExperts,
        batched_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        supports_chunking=False,
        supports_expert_map=False,
        needs_matching_quant=True,
    )
    register_experts(
        TritonOrDeepGemmExperts,
        standard_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=True,
        needs_matching_quant=True,
    )

if cutlass_fp8_supported():
    from vllm.model_executor.layers.fused_moe import (CutlassBatchedExpertsFp8,
                                                      CutlassExpertsFp8)
    register_experts(
        CutlassExpertsFp8,
        standard_format,
        fp8_types,
        blocked_quantization_support=False,
        supports_chunking=True,
        supports_expert_map=False,
    )
    register_experts(
        CutlassBatchedExpertsFp8,
        batched_format,
        fp8_types,
        blocked_quantization_support=False,
        supports_chunking=False,
        supports_expert_map=False,
    )

if False and cutlass_fp4_supported():
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        CutlassExpertsFp4)
    register_experts(
        CutlassExpertsFp4,
        standard_format,
        nv_fp4_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=False,
    )

MK_QUANT_CONFIGS = [
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

if False and (cutlass_fp4_supported() or has_flashinfer_cutlass_fused_moe()):
    MK_QUANT_CONFIGS += [
        FusedMoEQuantConfig(quant_dtype="nvfp4",
                            per_out_ch_quant=False,
                            per_act_token_quant=False,
                            block_shape=None),
    ]


def make_fused_experts(
    fused_experts_type: mk.FusedMoEPermuteExpertsUnpermute,
    moe: FusedMoEConfig,
    num_dispatchers: int,
) -> mk.FusedMoEPermuteExpertsUnpermute:

    use_fp8 = moe.quant_dtype == torch.float8_e4m3fn
    batch_kwargs = {
        "max_num_tokens": moe.max_num_tokens,
        "num_dispatchers": num_dispatchers,
    }
    quant_kwargs = {
        "use_fp8_w8a8": use_fp8,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "block_shape": moe.block_shape,
        "per_act_token_quant": moe.per_act_token_quant,
    }
    deepgemm_kwargs = {"allow_deep_gemm": has_deep_gemm()}

    if fused_experts_type == BatchedDeepGemmExperts:
        kwargs = batch_kwargs | {
            "block_shape": moe.block_shape,
            "per_act_token_quant": moe.per_act_token_quant,
        }
        print(f"Making BatchedDeepGemmExperts {kwargs} ...")
        experts = BatchedDeepGemmExperts(**kwargs)
    elif fused_experts_type == BatchedTritonExperts:
        kwargs = batch_kwargs | quant_kwargs
        print(f"Making BatchedTritonExperts {kwargs} ...")
        experts = BatchedTritonExperts(**kwargs)
    elif fused_experts_type == BatchedTritonOrDeepGemmExperts:
        kwargs = batch_kwargs | quant_kwargs | deepgemm_kwargs
        print(f"Making BatchedTritonOrDeepGemmExperts {kwargs} ...")
        experts = BatchedTritonOrDeepGemmExperts(**kwargs)
    elif fused_experts_type == DeepGemmExperts:
        print("Making DeepGemmExperts () ...")
        experts = DeepGemmExperts()
    elif fused_experts_type == TritonExperts:
        kwargs = quant_kwargs
        print(f"Making TritonExperts {kwargs} ...")
        experts = TritonExperts(**kwargs)
    elif fused_experts_type == TritonOrDeepGemmExperts:
        kwargs = quant_kwargs | deepgemm_kwargs
        print(f"Making TritonOrDeepGemmExperts {kwargs} ...")
        experts = TritonOrDeepGemmExperts(**kwargs)
    elif fused_experts_type == NaiveBatchedExperts:
        kwargs = batch_kwargs | quant_kwargs
        print(f"Making NaiveBatchedExperts {kwargs} ...")
        experts = NaiveBatchedExperts(**kwargs)
    elif fused_experts_type == CutlassExpertsFp8:
        kwargs = {
            "out_dtype": moe.in_dtype,
            "per_act_token_quant": moe.per_act_token_quant,
            "per_out_ch_quant": moe.per_out_ch_quant,
            "block_shape": moe.block_shape,
        }
        print(f"Making CutlassExpertsFp8 {kwargs} ...")
        experts = CutlassExpertsFp8(**kwargs)
    elif fused_experts_type == CutlassBatchedExpertsFp8:
        kwargs = {
            "max_experts_per_worker": moe.num_local_experts,
            "num_dispatchers": num_dispatchers,
            "out_dtype": moe.in_dtype,
            "per_act_token_quant": moe.per_act_token_quant,
            "per_out_ch_quant": moe.per_out_ch_quant,
            "block_shape": moe.block_shape,
        }
        print(f"Making CutlassBatchedExpertsFp8 {kwargs} ...")
        experts = CutlassBatchedExpertsFp8(**kwargs)
    elif fused_experts_type == CutlassExpertsFp4:
        num_experts = moe.num_local_experts
        kwargs = {
            "g1_alphas": 0,  #TBD
            "g2_alphas": 0,  #TBD
            "a1_gscale": 0,  #TBD
            "a2_gscale": 0,  #TBD
            "device": 0,  # TBD
            "max_experts_per_worker": num_experts,
            "out_dtype": moe.in_dtype,
            "per_act_token_quant": moe.per_act_token_quant,
            "per_out_ch_quant": moe.per_out_ch_quant,
            "block_shape": moe.block_shape,
            "num_dispatchers": num_dispatchers,
        }
        print(f"Making CutlassExpertsFp4 {kwargs} ...")
        experts = CutlassExpertsFp4(**kwargs)
    elif fused_experts_type == FlashInferExperts:
        print(f"Making FlashInferExperts {kwargs} ...")
        kwargs = {
            "g1_alphas": 0,  #TBD
            "g2_alphas": 0,  #TBD
            "a1_gscale": 0,  #TBD
            "a2_gscale": 0,  #TBD
            "out_dtype": moe.in_dtype,
            "use_nvfp4_w4a4": True,
            "ep_rank": moe.ep_rank,
            "ep_size": moe.ep_size,
            "tp_rank": moe.tp_rank,
            "tp_size": moe.tp_size,
        }
        experts = FlashInferExperts(**kwargs)
    else:
        raise RuntimeError(f"Unknown fused experts type: {fused_experts_type}")

    return experts
