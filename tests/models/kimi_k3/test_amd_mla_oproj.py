# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.platforms import current_platform


def test_gated_o_proj_hook_exists():
    assert hasattr(MultiHeadLatentAttentionWrapper, "_gated_o_proj")


def test_rocm_wrapper_registers_on_rocm_only():
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.models.kimi_k3.amd.mla import ROCmMultiHeadLatentAttentionWrapper

    registered = op_registry_oot.get("MultiHeadLatentAttentionWrapper")
    if current_platform.is_rocm():
        assert registered is ROCmMultiHeadLatentAttentionWrapper
    else:
        assert registered is not ROCmMultiHeadLatentAttentionWrapper
