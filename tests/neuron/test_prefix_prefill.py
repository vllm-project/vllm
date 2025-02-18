# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch_xla.core.xla_model as xm
from utils import (convert_to_kernel_input_format, pad_to_multiple,
                   pad_to_next_power_of_2, ref_context_attention,
                   sample_input_sizes, sample_paged_attention_inputs)

from vllm.attention.ops.nki_flash_attn import (context_mask_reorder_helper,
                                               flash_attn_varlen_nkifunc)


class FlashPagedAttentionTest:

    def __init__(
        self,
        query_lens,
        ctx_lens,
        max_model_len,
        num_heads,
        num_queries_per_kv,
        head_size,
        block_size,
        large_tile_size,
        mixed_precision,
        return_debug_tensors=False,
    ):
        self.query_lens = query_lens
        self.ctx_lens = ctx_lens
        self.seq_lens = self.query_lens + self.ctx_lens
        self.max_model_len = max_model_len
        self.num_heads = num_heads
        self.num_queries_per_kv = num_queries_per_kv
        self.head_size = head_size
        self.block_size = block_size
        self.large_tile_size = large_tile_size
        self.mixed_precision = mixed_precision
        self.return_debug_tensors = return_debug_tensors
        self.dtype = torch.float32

    def run(self, **kwargs):
        query, k_active, v_active, k_cache, v_cache, block_table, key, value = (
            self.prepare_test_inputs())

        output_ref, debug_tensors_ref = self.run_reference_version(
            query, key, value, **kwargs)

        compiler_flags = [
            "--model-type=transformer -O1",
            "--internal-hlo2tensorizer-options='--verify-hlo'",
            "--retry_failed_compilation",
        ]
        compiler_flags_str = " ".join(compiler_flags)
        os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
        output_nki, debug_tensors_nki = self.run_neuron_version(
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            block_table,
            **kwargs,
        )
        self.compare_outputs(output_nki, output_ref, debug_tensors_nki,
                             debug_tensors_ref)

    def prepare_test_inputs(self):
        max_block_per_request = (self.max_model_len + self.block_size -
                                 1) // self.block_size
        batch_size = len(self.query_lens)
        num_blocks_in_cache = (batch_size * max_block_per_request) * 2
        num_kv_heads = self.num_heads // self.num_queries_per_kv

        return sample_paged_attention_inputs(
            query_lens=self.query_lens,
            ctx_lens=self.ctx_lens,
            max_block_per_request=max_block_per_request,
            num_blocks_in_cache=num_blocks_in_cache,
            block_size=self.block_size,
            num_heads=self.num_heads,
            num_kv_heads=num_kv_heads,
            head_size=self.head_size,
            dtype=self.dtype,
        )

    def run_reference_version(self, query, key, value, **kwargs):
        output_ref, *debug_tensors = ref_context_attention(
            query,
            key,
            value,
            self.query_lens,
            self.seq_lens,
            self.head_size,
            self.num_queries_per_kv,
            return_max_reduce=self.return_debug_tensors,
        )
        return output_ref, debug_tensors

    def convert_to_neuron_inputs(self, query, k_active, v_active, k_cache,
                                 v_cache, block_table):
        # build neuron program
        B_P_SIZE = 128
        B_F_SIZE = 512
        LARGE_TILE_SZ = self.large_tile_size
        assert LARGE_TILE_SZ >= B_P_SIZE

        max_num_queries = sum(self.query_lens)
        if max_num_queries > B_F_SIZE:
            # Due to underlying PE tiling
            max_num_queries = pad_to_multiple(max_num_queries, B_F_SIZE)
        else:
            max_num_queries = pad_to_next_power_of_2(max_num_queries)
        # kernel defines REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
        MAX_REDUCTION_TILE = 2048
        if max_num_queries // 2 > MAX_REDUCTION_TILE:
            max_num_queries = pad_to_multiple(max_num_queries,
                                              MAX_REDUCTION_TILE)

        return convert_to_kernel_input_format(
            query_lens=self.query_lens,
            context_lens=self.ctx_lens,
            block_table=block_table,
            k_cache=k_cache,
            v_cache=v_cache,
            query=query,
            k_active=k_active,
            v_active=v_active,
            block_size=self.block_size,
            LARGE_TILE_SZ=LARGE_TILE_SZ,
            max_num_queries=max_num_queries,
        )

    def run_neuron_version(
        self,
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        block_table,
        **kwargs,
    ):
        (
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            active_block_table,
            attn_mask,
        ) = self.convert_to_neuron_inputs(
            query,
            k_active,
            v_active,
            k_cache,
            v_cache,
            block_table,
        )
        reorder_mask_outside_kernel = kwargs.get("reorder_mask_outside_kernel",
                                                 False)
        attn_mask = context_mask_reorder_helper(
            attn_mask,
            self.large_tile_size,
            self.block_size,
        )

        device = xm.xla_device()
        input_args = (
            query.to(device=device),
            k_active.to(device=device),
            v_active.to(device=device),
            k_cache.to(device=device),
            v_cache.to(device=device),
            active_block_table.to(device=device),
            attn_mask.to(device=device),
        )
        LARGE_TILE_SZ = self.large_tile_size
        num_kv_heads = self.num_heads // self.num_queries_per_kv
        input_kwargs = dict(
            n_kv_head=num_kv_heads,
            head_size=self.head_size,
            mixed_precision=self.mixed_precision,
            LARGE_TILE_SZ=LARGE_TILE_SZ,
            mask_reordered=reorder_mask_outside_kernel,
            return_debug_tensors=self.return_debug_tensors,
        )

        if self.return_debug_tensors:
            output_nki, *debug_tensors = flash_attn_varlen_nkifunc(
                *input_args, **input_kwargs)
            debug_tensors = [torch.tensor(dt).cpu() for dt in debug_tensors]
        else:
            output_nki = flash_attn_varlen_nkifunc(*input_args, **input_kwargs)
            debug_tensors = []

        num_actual_tokens = sum(self.query_lens)
        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        output_nki = output_nki.cpu().permute(0, 2, 1, 3)
        output_nki = output_nki[0, :num_actual_tokens, :, :]
        return output_nki, debug_tensors

    def compare_outputs(self, out_nki, out_ref, debug_nki, debug_ref):
        torch.testing.assert_close(out_nki, out_ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "block_size,large_tile_size",
    [
        (256, 2048),  # 8 blocks
        (32, 2048),  # 64 blocks
        (16, 2048),  # 128 blocks
        (32, 4096),  # 128 blocks
        (64, 8192),  # 128 blocks
        (32, 8192),  # 256 blocks
        (4, 1024),  # 256 blocks
        (1, 512),  # 512 blocks
    ],
)
@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size",
    [
        (4, 2, 8),
        (32, 8, 64),
        (4, 4, 128),
        (8, 1, 32),
    ],
)
@pytest.mark.parametrize(
    "prefill_batch_size,decode_batch_size",
    [
        (4, 12),
        (1, 199),
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@torch.inference_mode()
def test_flash_paged_attention_numerical(
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    large_tile_size,
    mixed_precision: bool,
) -> None:

    torch.manual_seed(1000)
    torch.set_printoptions(sci_mode=False)

    assert large_tile_size % block_size == 0

    reorder_mask_outside_kernel = True

    min_ctx_len = 32
    max_ctx_len = 1024
    min_query_len = 16
    max_query_len = 512
    query_lens, ctx_lens = sample_input_sizes(
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        min_query_len=min_query_len,
        max_query_len=max_query_len,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
    )

    max_model_len = max(max_query_len, max_ctx_len) * 4
    test = FlashPagedAttentionTest(
        query_lens,
        ctx_lens,
        max_model_len,
        num_heads,
        num_queries_per_kv,
        head_size,
        block_size,
        large_tile_size,
        mixed_precision,
    )
    test.run(reorder_mask_outside_kernel=reorder_mask_outside_kernel)
