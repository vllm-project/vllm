# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.engine.core import EngineCore


def test_non_causal_attention_revalidates_scheduler_token_budget():
    max_model_len = 4096
    vllm_config = MagicMock()
    vllm_config.model_config.max_model_len = max_model_len
    vllm_config.scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    vllm_config.cache_config.enable_prefix_caching = True

    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = MagicMock()
    engine_core.model_executor.get_kv_cache_specs.return_value = [
        {"non_causal_layer": SimpleNamespace(non_causal=True)}
    ]

    with (
        patch("vllm.v1.engine.core.register_all_kvcache_specs"),
        pytest.raises(
            ValueError,
            match=r"max_num_batched_tokens \(2048\).*max_model_len \(4096\)",
        ),
    ):
        engine_core._initialize_kv_caches(vllm_config)

    assert not vllm_config.scheduler_config.enable_chunked_prefill
    assert not vllm_config.cache_config.enable_prefix_caching
