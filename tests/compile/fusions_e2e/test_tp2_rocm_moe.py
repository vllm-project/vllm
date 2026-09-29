# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE coverage for the TP2 GEMM/collective fusion contract."""

import pytest

from tests.utils import multi_gpu_test
from vllm.config import PassConfig
from vllm.platforms import current_platform

from .common import INDUCTOR_GRAPH_PARTITION
from .models import TRITON_ATTN, qwen3_a3b


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm AsyncTP contract")
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION[:1])
@pytest.mark.parametrize("attn_backend", [TRITON_ATTN])
def test_tp2_rocm_moe_async_tp_bf16_fusions(
    inductor_graph_partition, attn_backend, run_e2e_fusion_test
):
    """MoE dense projections retain the Qwen SP/AsyncTP rewrite counts."""
    n_layers = 4
    run_e2e_fusion_test(
        qwen3_a3b.model_name,
        qwen3_a3b.matches(n_layers),
        dict(
            hf_overrides=dict(
                num_hidden_layers=n_layers,
                hidden_size=512,
                intermediate_size=1024,
                moe_intermediate_size=128,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                num_experts=8,
                num_experts_per_tok=2,
            ),
            load_format="dummy",
            max_model_len=1024,
            kv_cache_memory_bytes=64 * 1024**2,
            gpu_memory_utilization=0.01,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
        ),
        attn_backend,
        dict(
            use_inductor_graph_partition=inductor_graph_partition,
            custom_ops=["-rms_norm"],
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
