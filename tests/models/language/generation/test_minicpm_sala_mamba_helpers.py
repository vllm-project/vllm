# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hybrid Mamba state helpers on MiniCPMSALAForCausalLM (H4)."""

import torch

from vllm.model_executor.models.minicpm_sala import MiniCPMSALAForCausalLM


class TestMambaStateHelpers:
    def test_get_mamba_state_copy_func_returns_tuple(self) -> None:
        funcs = MiniCPMSALAForCausalLM.get_mamba_state_copy_func()
        assert isinstance(funcs, tuple)
        assert len(funcs) >= 1
        assert callable(funcs[0])

    def test_get_mamba_state_dtype_from_config(self) -> None:
        """The recurrent lightning state is fp32 regardless of model dtype:
        the HF reference computes the GLA recurrence in fp32, and this
        classmethod (used by the cache allocator) must agree with
        MiniCPMSALALightningAttention.get_state_dtype (used by the layer);
        a model-dtype (bf16) state would be silently downcast on every
        decode step."""
        from unittest.mock import MagicMock

        vllm_config = MagicMock()
        vllm_config.model_config.dtype = torch.bfloat16
        vllm_config.cache_config.mamba_cache_dtype = "auto"
        dtypes = MiniCPMSALAForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
        assert isinstance(dtypes, tuple)
        assert dtypes[0] == torch.float32
