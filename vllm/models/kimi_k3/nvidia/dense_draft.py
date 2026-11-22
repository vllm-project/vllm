# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense MLA draft support for Kimi K3.1 (DSA target).

Dense EAGLE3, DSpark and MLA DFlash drafts keep their own fp8 MLA cache and
decode with FLASHINFER_MLA, or ROCM_AITER_MLA on ROCm (`configure_dense_draft`).
"""

from typing import Any

from vllm import envs
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def configure_dense_draft(spec_config: SpeculativeConfig, text_config: Any) -> None:
    """Default an EAGLE3/DSpark/MLA DFlash draft's attention backend and KV
    dtype on a KDA-hybrid DSA target: dense MLA + fp8 for MLA drafts, FLASH_ATTN
    + model dtype for GQA drafts. GQA DFlash drafts and MTP are unchanged."""
    if (
        spec_config.method not in ("eagle3", "dspark", "dflash")
        or getattr(text_config, "linear_attn_config", None) is None
    ):
        return
    draft_text_config = spec_config.draft_model_config.hf_text_config
    is_mla_draft = getattr(draft_text_config, "kv_lora_rank", None) is not None
    if spec_config.method == "dflash":
        if not is_mla_draft:
            return
        # A DFlash2 draft with MLA layers runs on K3's MLA draft model.
        draft_hf_config = spec_config.draft_model_config.hf_config
        if (
            draft_hf_config.architectures == ["DFlash2DraftModel"]
            and current_platform.is_cuda()
        ):
            draft_hf_config.architectures = ["DFlash2KimiK3DraftModel"]
            spec_config.update_arch_()
    if is_mla_draft:
        # MLA draft: no dense MLA kernel reads the target's fp8_ds_mla, so the
        # draft gets its own plain fp8 cache on the platform's dense MLA backend.
        backend = (
            AttentionBackendEnum.ROCM_AITER_MLA
            if current_platform.is_rocm()
            else AttentionBackendEnum.FLASHINFER_MLA
        )
        kv_cache_dtype = "fp8"
    else:
        # GQA draft: its own cache, not the target's sparse backend or fp8_ds_mla.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASH_ATTN, "auto"
    spec_config.attention_backend = spec_config.attention_backend or backend
    spec_config.kv_cache_dtype = spec_config.kv_cache_dtype or kv_cache_dtype


def aux_reads_attn_res(vllm_config: VllmConfig) -> bool:
    """Whether the draft reads the AttnRes stream of its target layers rather
    than the prefix sum: `dflash_config.aux_hidden_stream="attn_res"` in the
    draft config, or VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1."""
    if envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM:
        return True
    spec_config = vllm_config.speculative_config
    draft_model_config = getattr(spec_config, "draft_model_config", None)
    if draft_model_config is None:
        return False
    dflash_config = getattr(draft_model_config.hf_config, "dflash_config", None) or {}
    return dflash_config.get("aux_hidden_stream") == "attn_res"
