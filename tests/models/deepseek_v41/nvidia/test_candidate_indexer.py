# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate indexer configuration without loading model weights."""

import os

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Requires an SM90 GPU",
)
@pytest.mark.parametrize(
    "options,expected",
    [
        pytest.param({}, True, id="default"),
        pytest.param(
            {"attention_config": {"indexer_sparse_logits": False}},
            False,
            id="disabled",
        ),
        pytest.param(
            {"attention_config": {"indexer_sparse_logits": True}},
            True,
            id="enabled",
        ),
        pytest.param(
            {"hf_overrides": {"candidate_topk_blocks": 3}},
            True,
            id="candidate-geometry",
        ),
        pytest.param(
            {"hf_overrides": {"index_n_heads": 16}},
            True,
            id="head-geometry",
        ),
        pytest.param(
            {"attention_config": {"indexer_kv_dtype": "mxfp4"}},
            True,
            id="cache-validation",
        ),
        pytest.param(
            {
                "ubatch_size": 2,
                "hf_overrides": {"engram_layer_ids": []},
                "all2all_backend": "deepep_high_throughput",
            },
            True,
            id="microbatching-validation",
        ),
        pytest.param(
            {
                "tensor_parallel_size": 2,
                "decode_context_parallel_size": 2,
                "nnodes": 2,
            },
            False,
            id="decode-cp",
        ),
        pytest.param(
            {"prefill_context_parallel_size": 2, "nnodes": 2},
            False,
            id="prefill-cp",
        ),
    ],
)
def test_candidate_indexer_default_respects_support_and_override(options, expected):
    """Enable by platform and overrides; leave kernel constraints to the backend."""
    config = EngineArgs(
        model=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash"),
        tokenizer_mode="deepseek_v41",
        language_model_only=True,
        max_model_len=4096,
        **options,
    ).create_engine_config()
    assert config.attention_config.indexer_sparse_logits is expected
