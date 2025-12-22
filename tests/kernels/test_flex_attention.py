# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for FlexAttention backend vs default backend"""

import random

import numpy as np
import pytest
import torch
from packaging import version

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionMetadataBuilder,
    physical_to_logical_mapping,
)

from ..models.utils import check_embeddings_close, check_logprobs_close

TORCH_VERSION = version.parse(torch.__version__)
MINIMUM_TORCH_VERSION = version.parse("2.7.0")
DIRECT_BUILD_VERSION = version.parse("2.9.dev0")


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend, ensuring they are similar when using the same seed.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    num_logprobs = 5
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    set_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        output_flex = llm_flex.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    # Run with default backend
    set_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    ) as llm_default:
        output_default = llm_default.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=output_flex,
        outputs_1_lst=output_default,
        name_0="flex",
        name_1="default",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_encoder_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend for encoder models.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=100,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        flex_outputs = llm_flex.embed(prompts)

    # Run with default backend
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=100,
        enforce_eager=True,
    ) as llm_default:
        default_outputs = llm_default.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=flex_outputs,
        embeddings_1_lst=default_outputs,
        name_0="flex",
        name_1="default",
        tol=1e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < DIRECT_BUILD_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_block_mask_direct_vs_slow_path():
    """Test that direct path block mask is a superset of slow path.

    The direct path may include extra blocks for performance (over-estimation),
    but must include all blocks that the slow path determines are necessary.
    """
    device = torch.device("cuda")

    vllm_config = create_vllm_config(
        model_name="meta-llama/Meta-Llama-3-8B", block_size=16, max_model_len=1024
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # Use a mixed batch that will create groups spanning multiple sequences
    batch_spec = BatchSpec(
        seq_lens=[35, 64, 128, 256], query_lens=[33, 5, 32, 64], name="test_mixed_batch"
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    builder = FlexAttentionMetadataBuilder(kv_cache_spec, [], vllm_config, device)

    metadata_direct = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    builder.direct_build = False
    metadata_slow = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    assert metadata_direct.block_mask is not None
    assert metadata_slow.block_mask is not None

    # Extract block indices for comparison, B, H are the same
    direct_indices = metadata_direct.block_mask.kv_indices[0, 0]
    slow_indices = metadata_slow.block_mask.kv_indices[0, 0]
    direct_num = metadata_direct.block_mask.kv_num_blocks[0, 0]
    slow_num = metadata_slow.block_mask.kv_num_blocks[0, 0]

    # main test: every block needed by slow path must be in direct path
    num_groups = direct_num.shape[0]
    all_contained = True
    missing_details = []

    for group_idx in range(num_groups):
        direct_blocks = set(direct_indices[group_idx, : direct_num[group_idx]].tolist())
        slow_blocks = set(slow_indices[group_idx, : slow_num[group_idx]].tolist())

        missing_blocks = slow_blocks - direct_blocks
        if missing_blocks:
            all_contained = False
            missing_details.append(
                f"Group {group_idx}: missing {sorted(missing_blocks)}"
            )

    assert all_contained, (
        "Direct path is missing blocks required by slow path:\n"
        + "\n".join(missing_details)
    )


def test_physical_to_logical_mapping_handles_reused_blocks():
    """Regression test: reused physical blocks map to the latest logical block.

    For sliding-window / hybrid attention layers, physical KV-cache blocks can be
    reused over time. The inverse mapping must therefore select the latest
    logical block index for a physical block id.
    """
    # Padding should not make physical block 0 look live.
    block_table = torch.tensor([[6, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([1 * 16], dtype=torch.int32)  # only 1 block valid
    out = physical_to_logical_mapping(
        block_table=block_table, seq_lens=seq_lens, block_size=16, total_blocks=10
    )
    assert out[0, 0].item() == -1
    assert out[0, 6].item() == 0

    # If a physical block id appears multiple times (block reuse), mapping should
    # point to the latest logical block index.
    block_table2 = torch.tensor([[2, 2, 5]], dtype=torch.int32)
    seq_lens2 = torch.tensor([3 * 16], dtype=torch.int32)
    out2 = physical_to_logical_mapping(
        block_table=block_table2, seq_lens=seq_lens2, block_size=16, total_blocks=8
    )
    assert out2[0, 2].item() == 1


if __name__ == "__main__":
    pytest.main([__file__])
