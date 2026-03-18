# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .common import AttentionBackendCase, Matches, ModelFusionInfo, is_blackwell

# Attn backends
FLASHINFER_ATTN = pytest.param(
    AttentionBackendCase(
        backend=AttentionBackendEnum.FLASHINFER,
        model_kwargs=dict(kv_cache_dtype="fp8"),
    ),
    id="FLASHINFER",
    marks=pytest.mark.skipif(
        not is_blackwell() or not has_flashinfer(),
        reason="FI backend requires Blackwell and FlashInfer",
    ),
)

TRITON_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.TRITON_ATTN), id="TRITON_ATTN"
)

ROCM_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.ROCM_ATTN),
    id="ROCM_ATTN",
    marks=pytest.mark.skipif(
        not current_platform.is_rocm(),
        reason="ROCm attention only for AMD",
    ),
)

ROCM_AITER_UNIFIED_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN),
    id="ROCM_AITER_UNIFIED_ATTN",
    marks=pytest.mark.skipif(
        not is_aiter_found_and_supported(),
        reason="ROCM_AITER_UNIFIED_ATTN only for AMD when AITER is installed",
    ),
)

FLASHINFER_MLA_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.FLASHINFER_MLA),
    id="FLASHINFER_MLA",
    marks=pytest.mark.skipif(
        not is_blackwell() or not has_flashinfer(),
        reason="FI backend requires Blackwell and FlashInfer",
    ),
)

TRITON_MLA_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.TRITON_MLA),
    id="TRITON_MLA",
)

FLASHMLA_SPARSE_ATTN = pytest.param(
    AttentionBackendCase(backend=AttentionBackendEnum.FLASHMLA_SPARSE),
    id="FLASHMLA_SPARSE",
    marks=pytest.mark.skipif(
        not is_blackwell(),
        reason="FlashMLA Sparse requires Blackwell",
    ),
)

# Models
llama3_8b = ModelFusionInfo(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    matches=lambda n_layers: Matches(
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 4,
    ),
)

llama3_8b_fp8 = ModelFusionInfo(
    model_name="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
    matches=lambda n_layers: Matches(
        rms_quant_fusion=n_layers * 2,
        act_quant_fusion=n_layers,
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 4,
    ),
)

llama3_8b_fp4 = ModelFusionInfo(
    model_name="nvidia/Llama-3.1-8B-Instruct-FP4",
    matches=lambda n_layers: Matches(
        act_quant_fusion=n_layers,
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 4,
    ),
)

# MoEs cannot do act+quant fusion because those ops are hidden from torch.compile.
# MoEs also only expose 1 rms+quant fusion because the quant for up_proj is hidden.
# TODO(luka): https://github.com/vllm-project/vllm/issues/31985
# Also, for MoEs, gemm+collective fusion only happens for dense GEMMs (o_proj/qkv proj)

llama4_scout_fp8 = ModelFusionInfo(
    model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    hf_overrides=lambda n_layers: {"text_config": {"num_hidden_layers": n_layers}},
    matches=lambda n_layers: Matches(
        rms_quant_fusion=n_layers,
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2,
        sequence_parallel=n_layers * 2,
        async_tp=n_layers * 2 - 1,
    ),
)

llama4_scout_fp4 = ModelFusionInfo(
    model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4",
    hf_overrides=lambda n_layers: {"text_config": {"num_hidden_layers": n_layers}},
    matches=lambda n_layers: Matches(
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2,
        sequence_parallel=n_layers * 2,
        async_tp=n_layers * 2 - 1,
    ),
)

qwen3_a3b = ModelFusionInfo(
    model_name="Qwen/Qwen3-30B-A3B",
    matches=lambda n_layers: Matches(
        norm_rope_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 2,
    ),
)

qwen3_a3b_fp8 = ModelFusionInfo(
    model_name="Qwen/Qwen3-30B-A3B-FP8",
    matches=lambda n_layers: Matches(
        rms_quant_fusion=n_layers,
        norm_rope_fusion=n_layers,
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 2,
    ),
)

deepseek_coder_v2_lite_fp8 = ModelFusionInfo(
    model_name="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
    matches=lambda n_layers: Matches(
        # first_k_dense_replace=1; MoE hides most rms+quant sites
        rms_quant_fusion=1,
        act_quant_fusion=min(1, n_layers),  # dense layers only
        # MLA attn + static FP8 quant
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
    ),
)

deepseek_v3_fp8 = ModelFusionInfo(
    model_name="deepseek-ai/DeepSeek-V3",
    matches=lambda n_layers: Matches(
        # 3 per dense layer (first 3):
        # - input_rms + qkv_proj
        # - q_a_layernorm + q_b_proj (inside MLA wrapper)
        # - post_attn_layernorm + MLP
        # 2 per MoE layer (remaining) due to MoE wrapping
        rms_quant_fusion=n_layers * 2 + min(3, n_layers),  # add for 3 dense layers
        # silu+block quant
        act_quant_fusion=min(3, n_layers),  # dense layers only
        # MLA attn + per-group FP8 quant not supported yet:
        # https://github.com/vllm-project/vllm/issues/35792
        attn_quant_fusion=0,
        ar_rms_fusion=n_layers * 2 + 1,
        # TODO
        # sequence_parallel= n_layers * 2 + 1,
        # async_tp=n_layers * 2,
    ),
)

deepseek_v32_fp4 = ModelFusionInfo(
    model_name="nvidia/DeepSeek-V3.2-NVFP4",
    matches=lambda n_layers: Matches(
        rms_quant_fusion=0,
        act_quant_fusion=0,
        attn_quant_fusion=n_layers,
        ar_rms_fusion=n_layers * 2 + 1,
    ),
)

gpt_oss_20b = ModelFusionInfo(
    model_name="openai/gpt-oss-20b",
    matches=lambda n_layers: Matches(
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 2,
    ),
)
