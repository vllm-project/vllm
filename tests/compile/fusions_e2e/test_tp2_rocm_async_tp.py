# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded ROCm model coverage for actual sequence-parallel AsyncTP rewrites."""

import pytest

from tests.utils import multi_gpu_test
from vllm.config import PassConfig
from vllm.platforms import current_platform

from .common import INDUCTOR_GRAPH_PARTITION, custom_ops_combos
from .models import TRITON_ATTN, llama3_8b


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm AsyncTP contract")
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.parametrize("custom_ops", tuple(custom_ops_combos("rms_norm")))
@pytest.mark.parametrize("attn_backend", [TRITON_ATTN])
def test_tp2_rocm_async_tp_bf16_fusions(
    inductor_graph_partition, custom_ops, attn_backend, run_e2e_fusion_test
):
    n_layers = 4
    model_kwargs = dict(
        hf_overrides=dict(
            num_hidden_layers=n_layers,
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
        ),
        load_format="dummy",
        max_model_len=1024,
        kv_cache_memory_bytes=64 * 1024**2,
        gpu_memory_utilization=0.01,
        max_num_batched_tokens=1024,
        max_num_seqs=32,
    )
    run_e2e_fusion_test(
        llama3_8b.model_name,
        llama3_8b.matches(n_layers),
        model_kwargs,
        attn_backend,
        dict(
            use_inductor_graph_partition=inductor_graph_partition,
            custom_ops=custom_ops.split(","),
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                enable_sp=True,
                fuse_gemm_comms=True,
                fuse_allreduce_rms=False,
                sp_min_token_num=512,
            ),
        ),
        ["norm_rope_fusion", "sequence_parallel", "async_tp"],
        tp_size=2,
    )
