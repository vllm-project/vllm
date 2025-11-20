# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

# Fused experts and PrepareFinalize imports
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (
    BatchedTritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts,
    NaiveBatchedExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.import_utils import has_deep_ep, has_deep_gemm, has_pplx


@dataclass
class TestMoEQuantConfig:
    quant_dtype: torch.dtype | str | None
    per_out_ch_quant: bool
    per_act_token_quant: bool
    block_shape: list[int] | None


@dataclass
class PrepareFinalizeInfo:
    activation_format: mk.FusedMoEActivationFormat
    supported_dtypes: list[torch.dtype | str]
    blocked_quantization_support: bool
    backend: str | None
    supports_apply_weight_on_input: bool = True


@dataclass
class ExpertInfo:
    activation_format: mk.FusedMoEActivationFormat
    supported_dtypes: list[torch.dtype | str]
    blocked_quantization_support: bool
    supports_chunking: bool
    supports_expert_map: bool
    needs_matching_quant: bool = False
    needs_deep_gemm: bool = False


PREPARE_FINALIZE_INFO: dict[mk.FusedMoEPrepareAndFinalize, PrepareFinalizeInfo] = {}
EXPERT_INFO: dict[mk.FusedMoEPermuteExpertsUnpermute, ExpertInfo] = {}
MK_ALL_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_MULTI_GPU_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES: list[mk.FusedMoEPrepareAndFinalize] = []
MK_FUSED_EXPERT_TYPES: list[mk.FusedMoEPermuteExpertsUnpermute] = []

standard_format = mk.FusedMoEActivationFormat.Standard
batched_format = mk.FusedMoEActivationFormat.BatchedExperts
common_float_types: list[torch.dtype | str] = [
    torch.float8_e4m3fn,
    torch.bfloat16,
    torch.float16,
    torch.float32,
]
common_float_and_int_types = common_float_types + [torch.int8]
nvfp4_types = ["nvfp4"]
fp8_types = [torch.float8_e4m3fn]


def register_prepare_and_finalize(
    kind,
    activation_format: mk.FusedMoEActivationFormat,
    supported_dtypes: list[torch.dtype | str],
    blocked_quantization_support: bool,
    backend: str | None,
    force_multigpu: bool = False,
    supports_apply_weight_on_input: bool = True,
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
        supports_apply_weight_on_input,
    )
    MK_ALL_PREPARE_FINALIZE_TYPES.append(kind)
    if backend is not None or force_multigpu:
        MK_MULTI_GPU_PREPARE_FINALIZE_TYPES.append(kind)
    else:
        MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES.append(kind)


def register_experts(
    kind,
    activation_format: mk.FusedMoEActivationFormat,
    supported_dtypes: list[torch.dtype | str],
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
    backend=None,
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

# Disable on blackwell for now
if has_deep_ep() and not current_platform.has_device_capability(100):
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
        DeepEPLLPrepareAndFinalize,
    )

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
        PplxPrepareAndFinalize,
    )

    register_prepare_and_finalize(
        PplxPrepareAndFinalize,
        batched_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        backend="pplx",
    )

if has_flashinfer_cutlass_fused_moe() and current_platform.has_device_capability(100):
    from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )
    from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
        FlashInferCutlassMoEPrepareAndFinalize,
        create_flashinfer_prepare_finalize,
    )

    register_prepare_and_finalize(
        FlashInferCutlassMoEPrepareAndFinalize,
        standard_format,
        nvfp4_types + fp8_types,
        blocked_quantization_support=True,
        backend=None,
        force_multigpu=True,
        supports_apply_weight_on_input=False,
    )

    register_experts(
        FlashInferExperts,
        standard_format,
        nvfp4_types + fp8_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        # Note: this is a hack to get it to run for now
        supports_expert_map=True,
    )
else:
    FlashInferCutlassMoEPrepareAndFinalize = None

if has_deep_gemm() and is_deep_gemm_supported():
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
    )
    register_experts(
        BatchedTritonOrDeepGemmExperts,
        batched_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        supports_chunking=False,
        supports_expert_map=False,
        needs_matching_quant=True,
        needs_deep_gemm=True,
    )
    register_experts(
        TritonOrDeepGemmExperts,
        standard_format,
        common_float_and_int_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=True,
        needs_matching_quant=True,
        needs_deep_gemm=True,
    )

if cutlass_fp8_supported():
    from vllm.model_executor.layers.fused_moe import (
        CutlassBatchedExpertsFp8,
        CutlassExpertsFp8,
    )

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

if cutlass_fp4_supported():
    from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4

    register_experts(
        CutlassExpertsFp4,
        standard_format,
        nvfp4_types,
        blocked_quantization_support=True,
        supports_chunking=True,
        supports_expert_map=False,
    )

MK_QUANT_CONFIGS: list[TestMoEQuantConfig | None] = [
    None,
    # per-channel / per-column weights and per-tensor activations
    TestMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn,
        per_out_ch_quant=True,
        per_act_token_quant=False,
        block_shape=None,
    ),
    # per-channel / per-column weights and per-token activations
    TestMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn,
        per_out_ch_quant=True,
        per_act_token_quant=True,
        block_shape=None,
    ),
    # per-tensor weights and per-tensor activations
    TestMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn,
        per_out_ch_quant=False,
        per_act_token_quant=False,
        block_shape=None,
    ),
    # per-tensor weights and per-token activations
    TestMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn,
        per_out_ch_quant=False,
        per_act_token_quant=True,
        block_shape=None,
    ),
    # block-quantized weights and 128 block per-token activations
    TestMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn,
        per_out_ch_quant=False,
        per_act_token_quant=False,
        block_shape=[128, 128],
    ),
    # TODO (varun) : Should we test the following combinations ?
    # block-quantized weights and per-token activations
    # block-quantized weights and per-tensor activations
]

if cutlass_fp4_supported() or has_flashinfer_cutlass_fused_moe():
    MK_QUANT_CONFIGS += [
        TestMoEQuantConfig(
            quant_dtype="nvfp4",
            per_out_ch_quant=False,
            per_act_token_quant=False,
            block_shape=None,
        ),
    ]


def make_prepare_finalize(
    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize,
    backend: str | None,
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEPrepareAndFinalize:
    if backend != "naive" and backend is not None:
        prepare_finalize = maybe_make_prepare_finalize(moe, quant_config)
        assert prepare_finalize is not None
        return prepare_finalize
    elif prepare_finalize_type == FlashInferCutlassMoEPrepareAndFinalize:
        return create_flashinfer_prepare_finalize(
            use_dp=moe.moe_parallel_config.dp_size > 1
        )
    else:
        return MoEPrepareAndFinalizeNoEP()


def _slice(rank: int, num_local_experts: int, t: torch.Tensor) -> torch.Tensor:
    s = rank * num_local_experts
    e = s + num_local_experts
    return t[s:e]


def make_cutlass_strides(
    e: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ab_strides1 = torch.full((e,), k, device="cuda", dtype=torch.int64)
    ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
    c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
    c_strides2 = torch.full((e,), k, device="cuda", dtype=torch.int64)
    return ab_strides1, ab_strides2, c_strides1, c_strides2


def make_fused_experts(
    fused_experts_type: mk.FusedMoEPermuteExpertsUnpermute,
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    num_dispatchers: int,
    N: int,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    batch_kwargs = {
        "max_num_tokens": moe.max_num_tokens,
        "num_dispatchers": num_dispatchers,
    }
    quant_kwargs = {
        "quant_config": quant_config,
    }
    deepgemm_kwargs = {"allow_deep_gemm": has_deep_gemm()}

    torch.set_printoptions(threshold=0, edgeitems=0, linewidth=10000)

    if fused_experts_type == BatchedDeepGemmExperts:
        kwargs = batch_kwargs | quant_kwargs
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
        print(f"Making DeepGemmExperts {quant_config} ...")
        experts = DeepGemmExperts(quant_config)
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
        strides = make_cutlass_strides(moe.num_experts, N, moe.hidden_dim)
        kwargs = {
            "out_dtype": moe.in_dtype,
            "ab_strides1": strides[0],
            "ab_strides2": strides[1],
            "c_strides1": strides[2],
            "c_strides2": strides[3],
        } | quant_kwargs
        print(f"Making CutlassExpertsFp8 {kwargs} ...")
        experts = CutlassExpertsFp8(**kwargs)
    elif fused_experts_type == CutlassBatchedExpertsFp8:
        strides = make_cutlass_strides(moe.num_experts, N, moe.hidden_dim)
        kwargs = {
            "max_experts_per_worker": moe.num_local_experts,
            "num_dispatchers": num_dispatchers,
            "out_dtype": moe.in_dtype,
            "ab_strides1": strides[0],
            "ab_strides2": strides[1],
            "c_strides1": strides[2],
            "c_strides2": strides[3],
        } | quant_kwargs
        print(f"Making CutlassBatchedExpertsFp8 {kwargs} ...")
        experts = CutlassBatchedExpertsFp8(**kwargs)
    elif fused_experts_type == CutlassExpertsFp4:
        kwargs = {
            "max_experts_per_worker": moe.num_local_experts,
            "num_dispatchers": num_dispatchers,
            "out_dtype": moe.in_dtype,
        } | quant_kwargs
        print(f"Making CutlassExpertsFp4 {kwargs} ...")
        experts = CutlassExpertsFp4(**kwargs)
    elif fused_experts_type == FlashInferExperts:
        kwargs = {
            "out_dtype": moe.in_dtype,
            "ep_rank": moe.ep_rank,
            "ep_size": moe.ep_size,
            "tp_rank": moe.tp_rank,
            "tp_size": moe.tp_size,
        } | quant_kwargs
        print(f"Making FlashInferExperts {kwargs} ...")
        experts = FlashInferExperts(**kwargs)
    else:
        raise RuntimeError(f"Unknown fused experts type: {fused_experts_type}")

    torch.set_printoptions(threshold=1000, edgeitems=5, linewidth=80)

    return experts
