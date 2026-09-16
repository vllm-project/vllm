# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf

pytestmark = pytest.mark.cpu_test


def test_align_hybrid_block_size_is_stable_across_tp(monkeypatch):
    class _FakeBackend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(32)]

    class _FakeModelCls:
        global_mamba_page_size = 32776

        @staticmethod
        def get_mamba_state_shape_from_config(vllm_config):
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            return ((
                _FakeModelCls.global_mamba_page_size // 2 // tp_size,
            ),)

        @staticmethod
        def get_mamba_state_dtype_from_config(vllm_config):
            return (torch.float16,)

    class _FakeModelConfig:
        architecture = "FakeHybridForCausalLM"
        dtype = torch.float16
        is_hybrid = True
        use_mla = False

        def get_head_size(self):
            return 64

        def get_total_num_kv_heads(self):
            return 2

        def get_num_kv_heads(self, parallel_config):
            return max(
                1,
                self.get_total_num_kv_heads()
                // parallel_config.tensor_parallel_size,
            )

        def get_mamba_chunk_size(self):
            return 16

    def _aligned_block_size(tp_size: int) -> int:
        cache_config = CacheConfig(
            block_size=32,
            cache_dtype="float16",
            mamba_cache_mode="none",
        )
        vllm_config = SimpleNamespace(
            cache_config=cache_config,
            model_config=_FakeModelConfig(),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        )
        Platform._align_hybrid_block_size(vllm_config, _FakeBackend)
        return cache_config.block_size

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (_FakeModelCls, None),
    )

    aligned_by_tp = {tp_size: _aligned_block_size(tp_size) for tp_size in (1, 2, 4)}

    assert aligned_by_tp == {
        1: 96,
        2: 96,
        4: 96,
    }
