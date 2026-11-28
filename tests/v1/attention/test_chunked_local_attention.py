# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import SchedulerConfig
from vllm.model_executor.layers.attention.chunked_local_attention import (
    create_chunked_local_attention_backend,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.utils import make_local_attention_virtual_batches


@dataclass(slots=True)
class LocalAttentionTestData:
    # Input parameters
    batch_spec: BatchSpec
    attn_chunk_size: int
    block_size: int
    # Expected return values
    expected_q_seqlens: list[int]
    expected_k_seqlens: list[int]
    expected_local_block_table: list[list[int]]


DEVICE_TYPE = current_platform.device_type

test_data_list = [
    # Same as example in docstring of make_local_attention_virtual_batches
    # except block table has 9 columns instead of 10
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[4, 10, 5],
            seq_lens=[6, 17, 9],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[2, 2, 1, 4, 4, 1, 4, 1],
        expected_k_seqlens=[4, 2, 4, 4, 4, 1, 4, 1],
        # 2 pages per local branch
        # (chunk size 4 // block size 2)
        expected_local_block_table=[
            [0, 1],  # local-batch 0, (batch 0, starting from k[0])
            [2, 3],  # local-batch 1, (batch 0, starting from k[4])
            [11, 12],  # local-batch 2, (batch 1, starting from k[4])
            [13, 14],  # local-batch 3, (batch 1, starting from k[8])
            [15, 16],  # local-batch 4, (batch 1, starting from k[12])
            [17, 17],  # local-batch 5, (batch 1, starting from k[16])
            [20, 21],  # local-batch 6, (batch 2, starting from k[4])
            [22, 23],  # local-batch 7, (batch 2, starting from k[8])
        ],
    ),
    # Case where block indices are not clipped to block table ncols-1
    # because tokens_in_last_block == attn_chunk_size
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[8],
            seq_lens=[12],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[4, 4],
        expected_k_seqlens=[4, 4],
        expected_local_block_table=[
            [2, 3],
            [4, 5],
        ],
    ),
    # Case where all kv_seq positions are involved in attn
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[7],
            # 10 - 7 = 3 previously computed tokens
            seq_lens=[10],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[1, 4, 2],
        expected_k_seqlens=[4, 4, 2],
        expected_local_block_table=[
            [0, 1],
            [2, 3],
            [4, 4],
        ],
    ),
    # Case where attn_chunk_size > kv_seq_len
    # so no extra mini virtual batches are created
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[4],
            seq_lens=[6],
        ),
        # Larger than kv_seq_len
        attn_chunk_size=10,
        block_size=2,
        # No change to q_seqlens and k_seqlens
        expected_q_seqlens=[4],
        expected_k_seqlens=[6],
        # In this case, we only need a block-table like:
        #  block_table = [ [0, 1, 2] ] # 1 batch, 3 pages
        # But we need to pad it to 5 pages per local batch
        # because currently the pages_per_local_batch
        # is calculated as (attn_chunk_size // block_size)
        expected_local_block_table=[
            [0, 1, 2, 2, 2],
        ],
    ),
    # Block size equal to chunk size
    # Expect single page per batch in local batch table
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[6, 6],
            seq_lens=[8, 8],
        ),
        attn_chunk_size=4,
        block_size=4,
        expected_q_seqlens=[2, 4, 2, 4],
        expected_k_seqlens=[4, 4, 4, 4],
        # Initial block table = [
        #    [0, 1], < batch 0
        #    [2, 3], < batch 1
        # ]
        expected_local_block_table=[
            [0],  # local-batch 0, (batch 0, starting from k[0])
            [1],  # local-batch 1, (batch 0, starting from k[4])
            [2],  # local-batch 1, (batch 0, starting from k[0])
            [3],  # local-batch 1, (batch 0, starting from k[4])
        ],
    ),
    # Case where query falls in the second attention chunk
    #  k_toks >   0 1 2 3 4
    #  q_toks v  _____________
    #         0 | 1
    #         1 | 1 1
    #         2 | 1 1 1
    #         3 | 1 1 1 1
    #         4 |         1
    #  where tokens 0,1,2,3 have been pre-computed
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[1],
            seq_lens=[5],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[1],
        expected_k_seqlens=[1],
        expected_local_block_table=[
            [2, 2],
        ],
    ),
]


@pytest.mark.parametrize("test_data", test_data_list)
def test_local_attention_virtual_batches(test_data: LocalAttentionTestData):
    device = torch.device(f"{DEVICE_TYPE}:0")
    batch_spec = test_data.batch_spec
    attn_chunk_size = test_data.attn_chunk_size
    block_size = test_data.block_size
    expected_q_seqlens = test_data.expected_q_seqlens
    expected_k_seqlens = test_data.expected_k_seqlens
    expected_local_block_table = test_data.expected_local_block_table

    # Create common attention metadata
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size,
        device,
        # Use torch.arange instead of torch.randint so we can assert on
        # block table tensor values. The block table will have shape
        # (num_batches, cdiv(max_seq_len, block_size)) and the values will be
        # arranged from 0 to cdiv(max_seq_len, block_size)-1
        arange_block_indices=True,
    )

    # Call the function
    result, _ = make_local_attention_virtual_batches(
        attn_chunk_size, common_attn_metadata, block_size
    )

    # Convert to numpy for easier comparison
    actual_q_seqlens = np.diff(result.query_start_loc_cpu.numpy())
    assert result.seq_lens_cpu_upper_bound is not None
    actual_k_seqlens = result.seq_lens_cpu_upper_bound.numpy()

    # Check that all query lengths are less than or equal to attn_chunk_size
    assert all(q_len <= attn_chunk_size for q_len in actual_q_seqlens)
    # Check that all key lengths are less than or equal to attn_chunk_size
    assert all(k_len <= attn_chunk_size for k_len in actual_k_seqlens)
    # Check that the total number of query tokens is preserved
    assert sum(actual_q_seqlens) == sum(batch_spec.query_lens)

    # Verify results
    np.testing.assert_array_equal(actual_q_seqlens, expected_q_seqlens)
    np.testing.assert_array_equal(actual_k_seqlens, expected_k_seqlens)

    expected_block_table_tensor = torch.tensor(
        expected_local_block_table, dtype=torch.int32, device=device
    )

    print(f"Expected block table:\n{expected_block_table_tensor}")
    print(f"Actual block table:\n{result.block_table_tensor}")

    torch.testing.assert_close(result.block_table_tensor, expected_block_table_tensor)


@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        # 255 decodes and a 16,129-token prefill chunk that starts at 4,096.
        ([1] * 255 + [16129], [3000] * 255 + [4096 + 16129]),
        # Every request starts one token before a chunk boundary.
        ([2] * 255 + [15874], [8193] * 255 + [8191 + 15874]),
    ],
    ids=["decodes-and-prefill", "all-straddle"],
)
def test_chunked_local_builder_holds_virtual_batches(query_lens, seq_lens):
    """The underlying builder gets one request per virtual batch, so its
    request capacity must cover the split of any batch the scheduler forms."""
    attn_chunk_size, block_size = 8192, 64
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        max_model_len=131072,
        is_encoder_decoder=False,
    )
    assert len(query_lens) <= scheduler_config.max_num_seqs
    assert sum(query_lens) <= scheduler_config.max_num_batched_tokens
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size,
        torch.device("cpu"),
    )
    virtual, _ = make_local_attention_virtual_batches(
        attn_chunk_size, common_attn_metadata, block_size
    )
    assert virtual.num_reqs > scheduler_config.max_num_seqs

    builder_cls = create_chunked_local_attention_backend(
        TritonAttentionBackend, attn_chunk_size
    ).get_builder_cls()
    vllm_config = SimpleNamespace(scheduler_config=scheduler_config)
    assert virtual.num_reqs <= builder_cls.get_max_num_reqs(vllm_config)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="Requires FlashInfer on CUDA",
)
def test_flashinfer_chunked_local_prefill_beyond_max_num_seqs(vllm_runner, tmp_path):
    """A prompt spanning several chunks splits into more virtual batches than
    max_num_seqs, and FlashInfer's native prefill must still serve it."""
    from transformers import Llama4TextConfig

    from vllm import SamplingParams, TokensPrompt

    Llama4TextConfig(
        architectures=["Llama4ForCausalLM"],
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        intermediate_size_mlp=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
        attention_chunk_size=64,
        no_rope_layers=[1, 0],
        interleave_moe_layer_step=100,
        num_local_experts=2,
        use_qk_norm=False,
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        attention_config={"backend": "FLASHINFER", "use_trtllm_attention": False},
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        block_size=16,
        kv_cache_memory_bytes=16 << 20,
    ) as runner:
        prompt = TokensPrompt(prompt_token_ids=[3 + i % 500 for i in range(200)])
        params = SamplingParams(
            temperature=0, max_tokens=4, ignore_eos=True, detokenize=False
        )
        output = runner.get_llm().generate(prompt, params, use_tqdm=False)[0]
    assert len(output.outputs[0].token_ids) == 4
