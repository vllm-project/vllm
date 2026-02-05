# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

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
        rms_quant_fusion=0,
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
        rms_quant_fusion=0,
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
        # TODO broken on Blackwell:
        # https://github.com/vllm-project/vllm/issues/33295
        norm_rope_fusion=0 if is_blackwell() else n_layers,
        attn_quant_fusion=0,  # attn + group quant not supported
        ar_rms_fusion=n_layers * 2 + 1,
        sequence_parallel=n_layers * 2 + 1,
        async_tp=n_layers * 2,
    ),
)
