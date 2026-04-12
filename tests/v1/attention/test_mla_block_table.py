# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for MLA expanded block table padding (DeepSeek V3.2).

In vllm/v1/attention/backends/mla/indexer.py, when the expanded block
table has padding rows (actual_expanded < num_decode_tokens), the
original code only zeroed column 0 of those padding rows, leaving
columns 1+ with stale block IDs from previous iterations. If FlashMLA
reads those columns for padding entries, it accesses random KV cache
blocks — causing silent corruption.
"""

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config.vllm import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

DEVICE = torch.device(current_platform.device_type)

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
BLOCK_SIZE = 64
# DeepSeek V3.2 MLA params
NUM_KV_HEADS = 1
HEAD_SIZE = 576


def make_builder(
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 512,
    max_model_len: int = 4096,
) -> DeepseekV32IndexerMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name=MODEL_NAME,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=BLOCK_SIZE,
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    with set_current_vllm_config(vllm_config):
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["layer0"],
            vllm_config=vllm_config,
            device=DEVICE,
        )
    return builder


class TestMLABlockTablePadding:
    """Regression test for MLA expanded block table partial padding bug.

    Calls _prepare_decode_tensors on a real DeepseekV32IndexerMetadataBuilder
    to verify that padding rows in the expanded block table are fully zeroed.
    """

    def test_padding_rows_fully_zeroed(self):
        """Simulate two decode iterations where the batch shrinks.

        Iteration 1 fills 12 rows via repeat_interleave.
        Iteration 2 only needs 3 rows but num_decode_tokens is still 12.
        Padding rows (3..11) must be fully zeroed — not just column 0.

        Fails without fix: columns 1+ retain stale block IDs from iter 1.
        """
        builder = make_builder()
        max_blocks_per_req = builder.expanded_block_table_buffer.shape[1]

        # --- Iteration 1: 4 requests, decode_lens [3, 2, 4, 3] = 12 tokens
        num_reqs_1 = 4
        decode_lens_1 = torch.tensor([3, 2, 4, 3], dtype=torch.int32,
                                     device=DEVICE)
        seq_lens_1 = torch.tensor([30, 20, 40, 25], dtype=torch.int32,
                                  device=DEVICE)
        query_start_loc_1 = torch.tensor([0, 3, 5, 9], dtype=torch.int32,
                                         device=DEVICE)
        block_table_1 = torch.arange(
            1, max_blocks_per_req + 1, dtype=torch.int32, device=DEVICE,
        ).unsqueeze(0).expand(num_reqs_1, -1).contiguous()
        num_decode_tokens_1 = int(decode_lens_1.sum().item())  # 12

        builder._prepare_decode_tensors(
            seq_lens=seq_lens_1,
            block_table=block_table_1,
            decode_lens=decode_lens_1,
            decode_lens_cpu=decode_lens_1.cpu(),
            query_start_loc=query_start_loc_1,
            num_decodes=num_reqs_1,
            num_decode_tokens=num_decode_tokens_1,
            use_native=False,
            next_n=3,
            max_decode_len=4,
        )

        # --- Iteration 2: 2 requests, decode_lens [2, 1] = 3 actual tokens
        # but num_decode_tokens padded to 12 (e.g. CUDA graph batch size)
        num_reqs_2 = 2
        decode_lens_2 = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
        seq_lens_2 = torch.tensor([50, 35], dtype=torch.int32, device=DEVICE)
        query_start_loc_2 = torch.tensor([0, 2], dtype=torch.int32,
                                         device=DEVICE)
        block_table_2 = torch.full(
            (num_reqs_2, max_blocks_per_req), 99,
            dtype=torch.int32, device=DEVICE,
        )
        num_decode_tokens_2 = 12  # padded

        _, out_block_table, _, _, _ = builder._prepare_decode_tensors(
            seq_lens=seq_lens_2,
            block_table=block_table_2,
            decode_lens=decode_lens_2,
            decode_lens_cpu=decode_lens_2.cpu(),
            query_start_loc=query_start_loc_2,
            num_decodes=num_reqs_2,
            num_decode_tokens=num_decode_tokens_2,
            use_native=False,
            next_n=2,
            max_decode_len=2,
        )

        torch.cuda.synchronize()

        actual_expanded = int(decode_lens_2.sum().item())  # 3
        padding = out_block_table[actual_expanded:].cpu()
        assert (padding == 0).all(), (
            f"Padding rows have non-zero values (stale block IDs from "
            f"previous iteration):\n{padding}\n"
            f"This means FlashMLA could read wrong KV cache blocks for "
            f"padding entries."
        )
