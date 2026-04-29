# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import gc
import random
from dataclasses import dataclass
from functools import partial

import pandas as pd
import torch
from torch import Tensor

from vllm import _custom_ops as ops
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.third_party.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.triton_utils import triton
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
DEEPGEMM_MAX_LOGITS_BYTES = DEEPGEMM_MAX_LOGITS_MB * 1024 * 1024
NUM_SMS = torch.cuda.get_device_properties(
    torch.cuda.current_device()
).multi_processor_count


def make_random_fp4(*prefix_shape: int) -> tuple[Tensor, Tensor]:
    fp4_shape = (*prefix_shape, HEAD_DIM // 2)
    scales_shape = (*prefix_shape, HEAD_DIM // MXFP4_BLOCK_SIZE)
    fp4 = torch.randint(0, 256, fp4_shape, dtype=torch.uint8)
    scales = torch.randint(124, 131, scales_shape, dtype=torch.uint8)
    return fp4, scales


def dequant(values: Tensor, scales: Tensor | None, use_fp4: bool) -> Tensor:
    # FP8: values are e4m3, and K has one float32 scale per token.
    if not use_fp4:
        values = values.view(torch.float8_e4m3fn).float()
        if scales is not None:
            values *= scales.view(torch.float32)
        return values

    # FP4: each byte stores two e2m1 values, scaled per 32-value group.
    assert scales is not None
    values = values.to(torch.int64)
    nibbles = torch.stack((values & 0xF, (values >> 4) & 0xF), dim=-1).flatten(-2)
    # fmt: off
    FP4_VALUES = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    # fmt: on
    LUT = torch.tensor(FP4_VALUES, device=values.device)
    scale = scales.view(torch.float8_e8m0fnu).float()
    dq = LUT[nibbles].unflatten(-1, (-1, MXFP4_BLOCK_SIZE)) * scale[..., None]
    return dq.flatten(-2)


class Reference:
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
        topk: int,
    ) -> None:
        self.block_size = block_size
        self.block_table = block_table_cpu.tolist()
        self.query_lens = query_lens_cpu.tolist()
        self.seq_lens = seq_lens_cpu.tolist()
        self.compressed_seq_lens = compressed_seq_lens_cpu.tolist()
        self.query_start_locs = query_start_loc_cpu.tolist()
        self.compress_ratio = compress_ratio
        self.use_fp4 = use_fp4
        self.topk = topk

    def run(
        self, q_quant: Tensor, q_scale: Tensor | None, kv_cache: Tensor, weights: Tensor
    ) -> Tensor:
        total_query_len = self.query_start_locs[-1]
        topk_indices_buffer = torch.empty(
            (total_query_len, self.topk), dtype=torch.int32, device="cuda"
        )
        kq_dim = HEAD_DIM // 2 if self.use_fp4 else HEAD_DIM

        # Walk requests exactly as the logical prefill batch is defined.
        for req_idx, query_len in enumerate(self.query_lens):
            query_start = self.query_start_locs[req_idx]
            context_len = self.seq_lens[req_idx] - query_len
            compressed_seq_len = self.compressed_seq_lens[req_idx]
            token_slice = slice(query_start, query_start + query_len)

            if compressed_seq_len == 0:
                continue

            k_values = torch.empty(
                compressed_seq_len, kq_dim, dtype=torch.uint8, device="cuda"
            )
            k_scales = torch.empty(
                compressed_seq_len,
                MXFP4_BLOCK_SIZE // 8,
                dtype=torch.uint8,
                device="cuda",
            )

            # Gather this request's compressed K from paged cache blocks.
            for block_idx in range(cdiv(compressed_seq_len, self.block_size)):
                block_id = self.block_table[req_idx][block_idx]
                token_start = block_idx * self.block_size
                token_end = min(token_start + self.block_size, compressed_seq_len)
                block_tokens = token_end - token_start

                cache_page = kv_cache[block_id].view(-1)
                value_count = block_tokens * kq_dim
                scale_offset = self.block_size * kq_dim
                scale_count = block_tokens * (MXFP4_BLOCK_SIZE // 8)
                k_values[token_start:token_end] = cache_page[:value_count].view(
                    block_tokens, kq_dim
                )
                k_scales[token_start:token_end] = cache_page[
                    scale_offset : scale_offset + scale_count
                ].view(block_tokens, MXFP4_BLOCK_SIZE // 8)

            # Dequantize K and Q into plain float tensors.
            k = dequant(k_values, k_scales, self.use_fp4)

            if self.use_fp4:
                q_scales = q_scale[token_slice].view(torch.uint8)
                q_scales = q_scales.reshape(query_len, NUM_HEADS, -1)
            else:
                q_scales = None
            q = dequant(q_quant[token_slice], q_scales, self.use_fp4)

            # Reference logits and top-k for each causal row.
            scores = torch.einsum("mhd,nd->hmn", q, k)
            logits = (scores.relu() * weights[token_slice].T.unsqueeze(-1)).sum(dim=0)
            for offset in range(query_len):
                row_len = (context_len + offset + 1) // self.compress_ratio
                k_top = min(self.topk, row_len)
                if k_top > 0:
                    topk_indices_buffer[query_start + offset, :k_top] = (
                        logits[offset, :row_len].topk(k_top).indices.to(torch.int32)
                    )
        return topk_indices_buffer


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
        topk: int,
        do_chunk: bool = False,
    ) -> None:
        total_query_len = int(query_start_loc_cpu[-1].item())
        block_table = block_table_cpu.cuda()
        query_start_loc = query_start_loc_cpu.cuda()
        seq_lens = seq_lens_cpu.cuda()
        compressed_seq_lens = compressed_seq_lens_cpu.cuda()
        self.topk_indices_buffer = torch.empty(
            (total_query_len, topk), dtype=torch.int32, device="cuda"
        )

        if do_chunk:
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu,
                query_lens_cpu,
                DEEPGEMM_WORKSPACE_SIZE,
                DEEPGEMM_MAX_LOGITS_BYTES,
            )
        else:
            chunk_specs = [
                (slice(0, query_lens_cpu.numel()), slice(0, total_query_len))
            ]

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
        self.num_chunks = len(self.chunks)
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
        self, q_quant: Tensor, q_scale: Tensor | None, kv_cache: Tensor, weights: Tensor
    ) -> Tensor:
        use_fp4 = q_scale is not None
        _, topk = self.topk_indices_buffer.shape

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
                self.topk_indices_buffer[token_slice],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk,
            )
        return self.topk_indices_buffer


class DeepGemmPaged:
    @dataclass
    class Metadata:
        token_start: int
        token_end: int
        indices: Tensor
        context_lens: Tensor
        block_table: Tensor
        schedule_metadata: Tensor
        max_context_len: int

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
        topk: int,
        do_chunk: bool = False,
    ) -> None:
        total_query_len = int(query_start_loc_cpu[-1].item())
        self.topk_indices_buffer = torch.empty(
            (total_query_len, topk), dtype=torch.int32, device="cuda"
        )

        req_ids = torch.arange(query_lens_cpu.numel(), device="cpu")
        all_req_idx = torch.repeat_interleave(
            req_ids, query_lens_cpu, output_size=total_query_len
        )
        token_idx = torch.arange(total_query_len, dtype=torch.int32, device="cpu")
        query_offsets = token_idx - query_start_loc_cpu[all_req_idx]
        all_context_lens = (
            seq_lens_cpu[all_req_idx] - query_lens_cpu[all_req_idx] + query_offsets + 1
        ) // compress_ratio

        block_table = block_table_cpu.cuda()
        if do_chunk:
            # The paged path reads K directly from the KV cache, so it does not
            # need the flat path's gathered-K workspace bound.
            paged_chunk_workspace_size = int(compressed_seq_lens_cpu.sum().item())
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu,
                query_lens_cpu,
                paged_chunk_workspace_size,
                DEEPGEMM_MAX_LOGITS_BYTES,
            )
        else:
            chunk_specs = [
                (slice(0, query_lens_cpu.numel()), slice(0, total_query_len))
            ]

        self.chunks: list[DeepGemmPaged.Metadata] = []
        for req_slice, query_slice in chunk_specs:
            if int(compressed_seq_lens_cpu[req_slice].sum().item()) == 0:
                continue

            token_start = int(query_start_loc_cpu[req_slice.start].item())
            token_end = token_start + query_slice.stop
            token_start += query_slice.start
            token_slice = slice(token_start, token_end)

            req_idx = all_req_idx[token_slice].cuda()
            indices = req_idx.to(torch.int32)
            context_lens = all_context_lens[token_slice].cuda().view(-1, 1)
            schedule_metadata = get_paged_mqa_logits_metadata(
                context_lens, block_size, NUM_SMS, indices=indices
            )
            chunk = DeepGemmPaged.Metadata(
                token_start,
                token_end,
                indices,
                context_lens,
                block_table[req_idx],
                schedule_metadata,
                int(compressed_seq_lens_cpu[req_slice].max().item()),
            )
            self.chunks.append(chunk)

        self.num_chunks = len(self.chunks)
        if not self.chunks:
            raise RuntimeError("generated requests produced no prefill chunks")

    def run(
        self, q_quant: Tensor, q_scale: Tensor | None, kv_cache: Tensor, weights: Tensor
    ) -> Tensor:
        use_fp4 = q_scale is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, HEAD_DIM, use_fp4)

        for chunk in self.chunks:
            token_slice = slice(chunk.token_start, chunk.token_end)
            num_tokens = chunk.context_lens.shape[0]

            q_values = q_quant[token_slice].view(num_tokens, 1, *q_quant.shape[1:])
            if q_scale is not None:
                q_values = q_values.view(torch.int8)
                q_scale_slice = q_scale[token_slice].view(
                    num_tokens, 1, *q_scale.shape[1:]
                )
            else:
                q_scale_slice = None

            logits = fp8_fp4_paged_mqa_logits(
                (q_values, q_scale_slice),
                kv_cache,
                weights[token_slice],
                chunk.context_lens,
                chunk.block_table,
                chunk.schedule_metadata,
                chunk.max_context_len,
                False,
                indices=chunk.indices,
            )
            torch.ops._C.top_k_per_row_decode(
                logits,
                1,
                chunk.context_lens,
                self.topk_indices_buffer[token_slice],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                self.topk_indices_buffer.shape[1],
            )
        return self.topk_indices_buffer


IMPLEMENTATIONS = (
    ("deepgemm_flat", DeepGemmFlat),
    ("deepgemm_flat_chunk", partial(DeepGemmFlat, do_chunk=True)),
    ("deepgemm_paged", DeepGemmPaged),
    ("deepgemm_paged_chunk", partial(DeepGemmPaged, do_chunk=True)),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all DeepGEMM prefill sparse-indexer paths."
    )
    parser.add_argument("--dtype", choices=["fp4", "fp8"], required=True)
    parser.add_argument("--num_requests", type=int, default=8)
    parser.add_argument("--min_query_len", type=int, default=16)
    parser.add_argument("--max_query_len", type=int, default=128)
    parser.add_argument("--min_context_len", type=int, default=0)
    parser.add_argument("--max_context_len", type=int, nargs="+", default=[1024])
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--compress_ratio", type=int, default=4)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_axis", action="store_true")
    parser.add_argument("--graph_path")
    return parser.parse_args()


def check_topk_result(
    actual: Tensor,
    expected: Tensor,
    query_lens: list[int],
    seq_lens: list[int],
    compress_ratio: int,
    topk: int,
) -> tuple[int, int]:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    row_idx = 0
    compared_entries = 0
    wrong_entries = 0

    for query_len, seq_len in zip(query_lens, seq_lens):
        context_len = seq_len - query_len
        for offset in range(query_len):
            row_len = (context_len + offset + 1) // compress_ratio
            k = min(topk, row_len)
            if k > 0:
                actual_row = actual_cpu[row_idx, :k].sort().values
                expected_row = expected_cpu[row_idx, :k].sort().values
                row_mismatches = int(
                    (~torch.isin(actual_row, expected_row)).sum().item()
                )
                compared_entries += k
                wrong_entries += row_mismatches
            row_idx += 1

    return wrong_entries, compared_entries


def make_context_lens(
    min_context_len: int, max_context_len: int, num_requests: int
) -> list[int]:
    if num_requests == 1:
        return [max_context_len]

    span = max_context_len - min_context_len
    return [
        min_context_len + span * req_idx // (num_requests - 1)
        for req_idx in range(num_requests)
    ]


def export_sweep_graph(
    results_df: pd.DataFrame, output_path: str, log_axis: bool
) -> None:
    import matplotlib.pyplot as plt

    fig, (latency_ax, memory_ax, chunks_ax) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True
    )

    for impl_name, impl_results in results_df.groupby("implementation", sort=False):
        impl_results = impl_results.sort_values("max_context_len")
        x_values = impl_results["max_context_len"].to_numpy()
        latency_us = impl_results["latency_us"].to_numpy()
        p20_us = impl_results["p20_us"].to_numpy()
        p80_us = impl_results["p80_us"].to_numpy()
        peak_memory_mib = impl_results["peak_memory_mib"].to_numpy()
        num_chunks = impl_results["num_chunks"].to_numpy()

        latency_ax.plot(x_values, latency_us, marker="o", label=impl_name)
        latency_ax.fill_between(x_values, p20_us, p80_us, alpha=0.15)
        memory_ax.plot(x_values, peak_memory_mib, marker="o", label=impl_name)
        chunks_ax.plot(x_values, num_chunks, marker="o", label=impl_name)

    latency_ax.set_title("Latency")
    latency_ax.set_ylabel("us")
    latency_ax.tick_params(axis="x", labelbottom=True)
    if log_axis:
        latency_ax.set_xscale("log", base=10)
    latency_ax.set_ylim(bottom=0)
    latency_ax.grid(True, axis="y", alpha=0.25)
    latency_ax.legend()

    memory_ax.set_title("Peak memory")
    memory_ax.set_ylabel("MiB")
    if log_axis:
        memory_ax.set_xscale("log", base=10)
    memory_ax.set_ylim(bottom=0)
    memory_ax.grid(True, axis="y", alpha=0.25)
    memory_ax.legend()

    chunks_ax.set_xlabel("max_context_len")
    chunks_ax.set_title("Chunks")
    chunks_ax.set_ylabel("count")
    if log_axis:
        chunks_ax.set_xscale("log", base=10)
    chunks_ax.set_ylim(bottom=0)
    chunks_ax.grid(True, axis="y", alpha=0.25)
    chunks_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_benchmark_batch(
    args: argparse.Namespace,
    max_context_len: int,
    query_lens: list[int],
    q_quant: Tensor,
    q_scale: Tensor | None,
    weights: Tensor,
    use_fp4: bool,
) -> list[dict[str, object]]:
    context_lens = make_context_lens(
        args.min_context_len, max_context_len, args.num_requests
    )
    seq_lens: list[int] = []
    compressed_seq_lens: list[int] = []
    query_start_locs = [0]

    for query_len, context_len in zip(query_lens, context_lens):
        seq_len = query_len + context_len

        seq_lens.append(seq_len)
        # Production uses floor division: the compressed cache only has entries
        # for positions where (pos + 1) % compress_ratio == 0.
        compressed_seq_lens.append(seq_len // args.compress_ratio)
        query_start_locs.append(query_start_locs[-1] + query_len)

    cpu_i32 = dict(dtype=torch.int32, device="cpu")
    query_lens_cpu = torch.tensor(query_lens, **cpu_i32)
    seq_lens_cpu = torch.tensor(seq_lens, **cpu_i32)
    compressed_seq_lens_cpu = torch.tensor(compressed_seq_lens, **cpu_i32)
    query_start_loc_cpu = torch.tensor(query_start_locs, **cpu_i32)

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
    reference = Reference(
        block_table_cpu,
        query_lens_cpu,
        seq_lens_cpu,
        compressed_seq_lens_cpu,
        query_start_loc_cpu,
        args.compress_ratio,
        block_size,
        use_fp4,
        args.topk,
    )
    expected_topk = reference.run(q_quant, q_scale, kv_cache, weights)

    results: list[dict[str, object]] = []
    for impl_name, impl_cls in IMPLEMENTATIONS:
        impl = None
        topk_indices = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        impl = impl_cls(
            block_table_cpu,
            query_lens_cpu,
            seq_lens_cpu,
            compressed_seq_lens_cpu,
            query_start_loc_cpu,
            args.compress_ratio,
            block_size,
            use_fp4,
            args.topk,
        )
        topk_indices = impl.run(q_quant, q_scale, kv_cache, weights)
        torch.cuda.synchronize()
        peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
        wrong_entries, compared_entries = check_topk_result(
            topk_indices,
            expected_topk,
            query_lens,
            seq_lens,
            args.compress_ratio,
            args.topk,
        )
        latency_ms, p20_ms, p80_ms = triton.testing.do_bench(
            lambda impl=impl: impl.run(q_quant, q_scale, kv_cache, weights),
            warmup=25,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
        results.append(
            {
                "max_context_len": max_context_len,
                "implementation": impl_name,
                "num_chunks": impl.num_chunks,
                "wrong_entries": f"{wrong_entries}/{compared_entries}",
                "peak_memory_mib": peak_memory_mib,
                "latency_us": latency_ms * 1000,
                "p20_us": p20_ms * 1000,
                "p80_us": p80_ms * 1000,
            }
        )

    return results


def main() -> None:
    args = parse_args()
    print(args)
    print("implementations:", ", ".join(name for name, _ in IMPLEMENTATIONS))

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    use_fp4 = args.dtype == "fp4"
    query_lens = [
        random.randint(args.min_query_len, args.max_query_len)
        for _ in range(args.num_requests)
    ]
    total_query_len = sum(query_lens)

    # Prepare Q and per-token weights once so each sweep point uses the same
    # query-side inputs.
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

    results: list[dict[str, object]] = []
    for max_context_len in args.max_context_len:
        batch_results = run_benchmark_batch(
            args,
            max_context_len,
            query_lens,
            q_quant,
            q_scale,
            weights,
            use_fp4,
        )
        results.extend(batch_results)

    results_df = pd.DataFrame(results)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "peak_memory_mib": "{:.2f}".format,
                "latency_us": "{:.2f}".format,
                "p20_us": "{:.2f}".format,
                "p80_us": "{:.2f}".format,
            },
        )
    )
    if args.graph_path:
        export_sweep_graph(results_df, args.graph_path, args.log_axis)
        print(f"wrote {args.graph_path}")


if __name__ == "__main__":
    main()
