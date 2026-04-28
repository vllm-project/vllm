# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import torch
from torch import Tensor

from vllm import _custom_ops as ops
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.third_party.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import (
    build_prefill_chunk_metadata,
    split_indexer_prefill_chunks,
)

NUM_HEADS = 64
HEAD_DIM = 128
MXFP4_BLOCK_SIZE = 32
DEEPGEMM_WORKSPACE_SIZE = 4096
DEEPGEMM_MAX_LOGITS_MB = 16


def make_random_fp4(*prefix_shape: int) -> tuple[Tensor, Tensor]:
    fp4_shape = (*prefix_shape, HEAD_DIM // 2)
    scales_shape = (*prefix_shape, HEAD_DIM // MXFP4_BLOCK_SIZE)
    fp4 = torch.randint(0, 256, fp4_shape, dtype=torch.uint8)
    scales = torch.randint(124, 131, scales_shape, dtype=torch.uint8)
    return fp4, scales


class DeepGemmFlat:
    def __init__(
        self,
        block_table_cpu: Tensor,
        query_lens_cpu: Tensor,
        seq_lens_cpu: Tensor,
        compressed_seq_lens_cpu: Tensor,
        query_start_loc_cpu: Tensor,
        compress_ratio: int,
        block_size: int,
        use_fp4: bool,
    ) -> None:
        block_table = block_table_cpu.cuda()
        query_start_loc = query_start_loc_cpu.cuda()
        seq_lens = seq_lens_cpu.cuda()
        compressed_seq_lens = compressed_seq_lens_cpu.cuda()

        chunk_specs = split_indexer_prefill_chunks(
            compressed_seq_lens_cpu,
            query_lens_cpu,
            DEEPGEMM_WORKSPACE_SIZE,
            DEEPGEMM_MAX_LOGITS_MB * 1024 * 1024,
        )
        self.chunks = []
        for req_slice, query_slice in chunk_specs:
            metadata = build_prefill_chunk_metadata(
                req_slice.start,
                req_slice.stop,
                query_start_loc,
                query_start_loc_cpu,
                seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                compress_ratio,
                query_slice=query_slice,
                skip_kv_gather=query_slice.start > 0,
            )
            if metadata is not None:
                self.chunks.append(metadata)
        if not self.chunks:
            raise RuntimeError("generated requests produced no prefill chunks")

        workspace_tokens = max(chunk.total_seq_lens for chunk in self.chunks)
        if use_fp4:
            k_quant_shape = (workspace_tokens, HEAD_DIM // 2)
            k_scale_shape = (workspace_tokens, HEAD_DIM // MXFP4_BLOCK_SIZE)
            k_quant_dtype = k_scale_dtype = torch.uint8
        else:
            k_quant_shape = (workspace_tokens, HEAD_DIM)
            k_scale_shape = (workspace_tokens, 4)
            k_quant_dtype = torch.float8_e4m3fn
            k_scale_dtype = torch.uint8

        self.k_quant_full = torch.empty(
            k_quant_shape, dtype=k_quant_dtype, device=block_table.device
        )
        self.k_scale_full = torch.empty(
            k_scale_shape, dtype=k_scale_dtype, device=block_table.device
        )

    def run(
        self,
        q_quant: Tensor,
        q_scale: Tensor | None,
        kv_cache: Tensor,
        weights: Tensor,
        topk_indices_buffer: Tensor,
    ) -> Tensor:
        use_fp4 = q_scale is not None
        _, topk = topk_indices_buffer.shape

        for chunk in self.chunks:
            k_quant = self.k_quant_full[: chunk.total_seq_lens]
            k_scale = self.k_scale_full[: chunk.total_seq_lens]

            # This mirrors SparseAttentionIndexer: query sub-chunks from the same
            # request can reuse the previously gathered K workspace.
            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            token_slice = slice(chunk.token_start, chunk.token_end)
            q_slice = q_quant[token_slice]

            if use_fp4:
                q_slice = q_slice.view(torch.int8)
                q_scale_slice = q_scale[token_slice]
                k_quant = k_quant.view(torch.int8)
                k_scale = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_scale_slice = None
                k_scale = k_scale.view(torch.float32).squeeze(-1)

            logits = fp8_fp4_mqa_logits(
                (q_slice, q_scale_slice),
                (k_quant, k_scale),
                weights[token_slice],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_buffer[token_slice],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk,
            )
        return topk_indices_buffer


class DeepGemmPaged:
    def __init__(
        self,
        block_table_cpu: Tensor,
        query_lens_cpu: Tensor,
        seq_lens_cpu: Tensor,
        compressed_seq_lens_cpu: Tensor,
        query_start_loc_cpu: Tensor,
        compress_ratio: int,
        block_size: int,
        use_fp4: bool,
    ) -> None:
        total_query_len = int(query_start_loc_cpu[-1].item())
        if total_query_len == 0 or int(compressed_seq_lens_cpu.max().item()) == 0:
            raise RuntimeError("generated requests produced no prefill KV tokens")

        req_ids = torch.arange(query_lens_cpu.numel(), device="cpu")
        req_idx = torch.repeat_interleave(
            req_ids, query_lens_cpu, output_size=total_query_len
        )
        token_idx = torch.arange(total_query_len, dtype=torch.int32, device="cpu")
        query_offsets = token_idx - query_start_loc_cpu[req_idx]
        context_lens = (
            seq_lens_cpu[req_idx] - query_lens_cpu[req_idx] + query_offsets + 1
        ) // compress_ratio

        req_idx = req_idx.cuda()
        self.indices = req_idx.to(torch.int32)
        self.context_lens = context_lens.cuda().view(-1, 1)
        self.block_table = block_table_cpu.cuda().index_select(0, req_idx)
        self.max_context_len = int(compressed_seq_lens_cpu.max().item())

        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.schedule_metadata = get_paged_mqa_logits_metadata(
            self.context_lens, block_size, num_sms, indices=self.indices
        )

    def run(
        self,
        q_quant: Tensor,
        q_scale: Tensor | None,
        kv_cache: Tensor,
        weights: Tensor,
        topk_indices_buffer: Tensor,
    ) -> Tensor:
        num_tokens = self.context_lens.shape[0]
        q_values = q_quant.view(num_tokens, 1, *q_quant.shape[1:])
        if q_scale is not None:
            q_values = q_values.view(torch.int8)
            q_scale = q_scale.view(num_tokens, 1, *q_scale.shape[1:])

        logits = fp8_fp4_paged_mqa_logits(
            (q_values, q_scale),
            kv_cache_as_quant_view(kv_cache, HEAD_DIM, q_scale is not None),
            weights,
            self.context_lens,
            self.block_table,
            self.schedule_metadata,
            self.max_context_len,
            False,
            indices=self.indices,
        )
        torch.ops._C.top_k_per_row_decode(
            logits,
            1,
            self.context_lens,
            topk_indices_buffer,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_indices_buffer.shape[1],
        )
        return topk_indices_buffer


def parse_args():
    parser = FlexibleArgumentParser(
        description="Run the DeepGEMM prefill sparse-indexer path."
    )
    parser.add_argument(
        "--impl",
        choices=["deepgemm_flat", "deepgemm_paged"],
        default="deepgemm_flat",
    )
    parser.add_argument("--dtype", choices=["fp4", "fp8"], required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--min-query-len", type=int, default=16)
    parser.add_argument("--max-query-len", type=int, default=128)
    parser.add_argument("--min-context-len", type=int, default=0)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> Tensor:
    args = parse_args()
    print(args)

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Prepare request lengths.
    use_fp4 = args.dtype == "fp4"
    if args.impl == "deepgemm_paged" and args.block_size not in (32, 64):
        raise ValueError("deepgemm_paged requires --block-size 32 or 64")

    query_lens: list[int] = []
    seq_lens: list[int] = []
    compressed_seq_lens: list[int] = []
    query_start_locs = [0]

    for _ in range(args.num_requests):
        query_len = random.randint(args.min_query_len, args.max_query_len)
        context_len = random.randint(args.min_context_len, args.max_context_len)
        seq_len = query_len + context_len

        query_lens.append(query_len)
        seq_lens.append(seq_len)
        # Production uses floor division: the compressed cache only has entries
        # for positions where (pos + 1) % compress_ratio == 0.
        compressed_seq_lens.append(seq_len // args.compress_ratio)
        query_start_locs.append(query_start_locs[-1] + query_len)

    total_query_len = query_start_locs[-1]
    cpu_i32 = dict(dtype=torch.int32, device="cpu")
    query_lens_cpu = torch.tensor(query_lens, **cpu_i32)
    seq_lens_cpu = torch.tensor(seq_lens, **cpu_i32)
    compressed_seq_lens_cpu = torch.tensor(compressed_seq_lens, **cpu_i32)
    query_start_loc_cpu = torch.tensor(query_start_locs, **cpu_i32)

    # Prepare Q and per-token weights.
    weights = torch.randn(total_query_len, NUM_HEADS)
    if use_fp4:
        q_quant, q_scale = make_random_fp4(total_query_len, NUM_HEADS)
        q_scale = q_scale.view(torch.int32).squeeze(-1)
    else:
        q_bf16 = torch.randn(
            total_query_len * NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16
        )
        q_quant, q_fp8_scale = ops.scaled_fp8_quant(
            q_bf16, use_per_token_if_dynamic=True
        )
        weights *= q_fp8_scale.view(total_query_len, NUM_HEADS)
        q_quant = q_quant.view(total_query_len, NUM_HEADS, HEAD_DIM)
        q_scale = None

    # Prepare paged KV cache.
    block_size = args.block_size
    block_counts = [cdiv(seq_len, block_size) for seq_len in compressed_seq_lens]
    max_blocks = max(1, max(block_counts))
    valid_blocks = sum(block_counts)
    poison_block = valid_blocks
    kq_dim = HEAD_DIM // 2 if use_fp4 else HEAD_DIM
    ks_dim = 4
    cache_dim = kq_dim + ks_dim

    # FP8: 0xff is an FP8 NaN payload, and 0xffffffff is a float32 NaN
    # scale. FP4 has no true NaN representation, but 0xff is still a useful
    # poison value for spotting unintended reads.
    kv_cache = torch.full(
        (valid_blocks + 1, block_size * cache_dim), 0xFF, dtype=torch.uint8
    )
    block_table_cpu = torch.full(
        (len(block_counts), max_blocks), poison_block, **cpu_i32
    )
    physical_blocks = iter(torch.randperm(valid_blocks, device="cpu").tolist())

    for req_idx, seq_len in enumerate(compressed_seq_lens):
        if use_fp4:
            kq, ks = make_random_fp4(seq_len)
        else:
            k_bf16 = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16)
            kq, ks = ops.scaled_fp8_quant(k_bf16, use_per_token_if_dynamic=True)
            ks = ks.view(torch.uint8).reshape(seq_len, 4)

        for logical_block in range(block_counts[req_idx]):
            block_id = next(physical_blocks)
            block_table_cpu[req_idx, logical_block] = block_id

            start = logical_block * block_size
            end = min(start + block_size, seq_len)
            block_kq = kq[start:end].view(torch.uint8).view(-1)
            block_ks = ks[start:end].view(-1)

            scale_off = block_size * kq_dim
            kv_cache[block_id, : block_kq.numel()] = block_kq
            kv_cache[block_id, scale_off : scale_off + block_ks.numel()] = block_ks

    kv_cache = kv_cache.view(-1, block_size, cache_dim)
    topk_indices_buffer = torch.empty((total_query_len, args.topk), dtype=torch.int32)

    impl_cls = {
        "deepgemm_flat": DeepGemmFlat,
        "deepgemm_paged": DeepGemmPaged,
    }[args.impl]
    impl = impl_cls(
        block_table_cpu,
        query_lens_cpu,
        seq_lens_cpu,
        compressed_seq_lens_cpu,
        query_start_loc_cpu,
        args.compress_ratio,
        block_size,
        use_fp4,
    )
    return impl.run(q_quant, q_scale, kv_cache, weights, topk_indices_buffer)


if __name__ == "__main__":
    main()
