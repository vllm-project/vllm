# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GDN paged state I/O with checkpoint-buffer gather/export/scatter.

Optionally load the original chunk.py and a buffer implementation's chunk.py
and chunk_delta_h.py. All arms include convolution checkpoint export. Timing
excludes metadata and input allocation, and uses CUPTI with cold L2 and graphs.
Requires FlashInfer and cupti-python >= 13 matching the installed CUDA runtime.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_paged,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)


def load_module(path: Path, name: str):
    name = f"vllm.third_party.flash_linear_attention.ops.{name}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_case(
    state_dtype,
    key_heads,
    value_heads,
    num_seqs,
    buffer_fn,
    original,
    passes,
    exporter,
    buffer_autotuner,
):
    seq_len = 8192 // num_seqs
    q, k = (
        F.normalize(torch.randn(1, 8192, key_heads, 128), dim=-1).bfloat16()
        for _ in range(2)
    )
    v = torch.randn(1, 8192, value_heads, 128).bfloat16()
    g = F.logsigmoid(torch.randn(1, 8192, value_heads))
    beta = torch.sigmoid(torch.randn(1, 8192, value_heads))
    inputs = (q, k, v, g, beta)
    cu = torch.arange(num_seqs + 1, dtype=torch.int32) * seq_len
    common = dict(
        cu_seqlens=cu,
        chunk_indices=prepare_chunk_indices(cu, 64),
        chunk_offsets=prepare_chunk_offsets(cu, 64),
    )
    rows = torch.arange(1, num_seqs + 1, dtype=torch.int32)
    has_initial = torch.ones(num_seqs, dtype=torch.bool)
    offsets = torch.full_like(rows, seq_len // 128 * 64)
    checkpoint_rows = rows + num_seqs
    checkpoint = MambaPrefillCheckpointMetadata(offsets, checkpoint_rows)
    row_size = value_heads * 128 * 128
    pages = torch.randn(2 * num_seqs + 1, row_size + 96).to(state_dtype)
    pages.mul_(0.1)
    cache = pages[:, :row_size].view(-1, value_heads, 128, 128)
    initial_pages = pages.clone()
    buffer = torch.empty(num_seqs, value_heads, 128, 128, dtype=state_dtype)
    width = (2 * key_heads + value_heads) * 128
    raw_qkv = torch.randn(8192, width, dtype=torch.bfloat16)
    conv = torch.zeros(2 * num_seqs + 1, width, 3, dtype=torch.bfloat16)

    def paged():
        out = chunk_gated_delta_rule_paged(
            *inputs,
            state_cache=cache,
            state_indices=rows,
            has_initial_state=has_initial,
            checkpoint_offsets=offsets,
            checkpoint_state_indices=checkpoint_rows,
            **common,
        )
        exporter.export(checkpoint, raw_qkv=raw_qkv, conv_state=conv, cu_seqlens=cu)
        return out

    def buffered():
        initial = gather_initial_states(cache, rows, has_initial)
        out, final = buffer_fn(
            *inputs,
            initial_state=initial,
            output_final_state=True,
            checkpoint_state=buffer,
            checkpoint_offsets=offsets,
            **common,
        )
        exporter.export(
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=conv,
            recurrent_checkpoint=buffer,
            recurrent_state=cache,
            cu_seqlens=cu,
        )
        cache[rows] = final.to(state_dtype)
        return out

    arms = {"paged": paged, "buffer": buffered}
    if original is not None:

        def old_paged():
            out, _ = original(
                *inputs,
                state_cache=cache,
                state_indices=rows,
                has_initial_state=has_initial,
                checkpoint_state=cache,
                checkpoint_offsets=offsets,
                checkpoint_state_indices=checkpoint_rows,
                **common,
            )
            exporter.export(
                checkpoint,
                raw_qkv=raw_qkv,
                conv_state=conv,
                cu_seqlens=cu,
            )
            return out

        arms["original"] = old_paged

    expected_out = expected_pages = expected_conv = None
    launch_configs = {}
    for name, fn in arms.items():
        pages.copy_(initial_pages)
        conv.zero_()
        out = fn()
        if expected_out is None:
            expected_out = out.clone()
            expected_pages = pages.clone()
            expected_conv = conv.clone()
        else:
            for actual, expected in (
                (out, expected_out),
                (pages, expected_pages),
                (conv, expected_conv),
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for _ in range(5):
            fn()
        tuner = (
            buffer_autotuner
            if name == "buffer" and buffer_autotuner is not None
            else chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
        )
        config = tuner.best_config
        launch_configs[name] = dict(
            kwargs=config.kwargs,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    samples = {name: [] for name in arms}
    for pass_index in range(passes):
        order = list(arms.items())
        if pass_index % 2:
            order.reverse()
        for name, fn in order:
            pages.copy_(initial_pages)
            times = bench_gpu_time_with_cupti(
                fn, use_cuda_graph=True, cold_l2_cache=True
            )
            samples[name].append(statistics.median(times) * 1000)
    result = dict(
        dtype=str(state_dtype),
        key_heads=key_heads,
        value_heads=value_heads,
        num_seqs=num_seqs,
        seq_len=seq_len,
        samples_us=samples,
        launch_configs=launch_configs,
        median_us={k: statistics.median(v) for k, v in samples.items()},
    )
    return result


@torch.inference_mode()
def main():
    warnings.filterwarnings("error", message=".*[Ff]alling back.*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-chunk", type=Path)
    parser.add_argument("--buffer-dir", type=Path)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = (
        load_module(args.original_chunk, "_original_chunk").chunk_gated_delta_rule
        if args.original_chunk
        else None
    )
    buffer_fn = chunk_gated_delta_rule
    buffer_autotuner = None
    if args.buffer_dir:
        recurrence = load_module(args.buffer_dir / "chunk_delta_h.py", "_buffer_h")
        buffer_module = load_module(args.buffer_dir / "chunk.py", "_buffer_chunk")
        buffer_module.chunk_gated_delta_rule_fwd_h = (
            recurrence.chunk_gated_delta_rule_fwd_h
        )
        buffer_fn = buffer_module.chunk_gated_delta_rule
        buffer_autotuner = recurrence.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
        buffer_autotuner.cache_results = False

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    results = []
    exporter = MambaPrefillCheckpointExporter(state_len=3)
    for state_dtype in (torch.bfloat16, torch.float32):
        for key_heads, value_heads in ((4, 8), (16, 32)):
            for num_seqs in (1, 4, 16, 32):
                if buffer_autotuner is not None:
                    buffer_autotuner.cache.clear()
                result = benchmark_case(
                    state_dtype,
                    key_heads,
                    value_heads,
                    num_seqs,
                    buffer_fn,
                    original,
                    args.passes,
                    exporter,
                    buffer_autotuner,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=current_platform.get_device_name(),
                torch=torch.__version__,
                timing="CUPTI, CUDA graph, cold L2",
                arguments={k: str(v) for k, v in vars(args).items()},
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
