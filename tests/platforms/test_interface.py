# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import Platform

pytestmark = pytest.mark.cpu_test


def test_dspark_uses_target_mamba_page_for_block_size(monkeypatch):
    class FakeHybridModel:
        @staticmethod
        def get_mamba_state_shape_from_config(vllm_config):
            # DSpark adds verification state that is twice as large as the
            # target model's recurrent state.
            state_size = 256 if vllm_config.speculative_config else 128
            return ((state_size,),)

        @staticmethod
        def get_mamba_state_dtype_from_config(vllm_config):
            return (torch.float32,)

    class FakeAttentionBackend:
        @classmethod
        def customize_spec(cls, spec):
            return spec

        @classmethod
        def get_supported_kernel_block_sizes(cls):
            return [16]

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        staticmethod(lambda *args, **kwargs: (FakeHybridModel, None)),
    )

    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=16,
        mamba_cache_mode="align",
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
    )
    model_config = SimpleNamespace(
        architecture="FakeHybridModel",
        dtype=torch.float32,
        use_mla=False,
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 4,
        get_mamba_chunk_size=lambda: 16,
    )
    config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(use_dspark=lambda: True),
    )

    Platform._align_hybrid_block_size(config, FakeAttentionBackend)

    # One target state is 512 bytes, exactly one 16-token attention page.
    # The extra DSpark verification state doubles the physical Mamba page to
    # 1024 bytes, but must not double the target's logical block granularity.
    assert cache_config.block_size == 16
    assert cache_config.mamba_block_size == 16
    assert cache_config.mamba_page_size_padded is None
