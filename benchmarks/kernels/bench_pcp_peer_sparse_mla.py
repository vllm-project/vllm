# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare sparse MLA kernels reading local and CUDA VMM peer KV.

The benchmark gives every rank byte-identical KV contents, then times the same
query and selected token rows through:

* a rank-local VMM view;
* a rank-major view with pages striped across all ranks; and
* peer rows gathered once into local scratch before attention.

This isolates an attention kernel's peer-read sensitivity from model and
scheduler work. The staged path intentionally shares one selected set across
all query rows, representing the favorable high-reuse regime for staging.

Example::

    torchrun --standalone --nproc-per-node=4 \
      benchmarks/kernels/bench_pcp_peer_sparse_mla.py \
      --query-tokens 1,16,128,512 \
      --backends flashinfer,flashmla \
      --json-output /tmp/pcp-peer-sparse-mla.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)

PAGE_SIZE = 64
HEAD_DIM = 576
VALUE_DIM = 512
NUM_HEADS = 64
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
FLASHMLA_BYTES_PER_TOKEN = 656


def _parse_int_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def _parse_choices(value: str, supported: set[str], label: str) -> list[str]:
    result = value.split(",")
    if not result or any(item not in supported for item in result):
        raise argparse.ArgumentTypeError(
            f"{label} must be selected from {sorted(supported)}"
        )
    return result


def _parse_backends(value: str) -> list[str]:
    return _parse_choices(value, {"flashinfer", "flashmla"}, "backends")


def _parse_selection_patterns(value: str) -> list[str]:
    return _parse_choices(value, {"shared", "independent"}, "selection patterns")


def _parse_stage_modes(value: str) -> list[str]:
    return _parse_choices(value, {"selected", "history"}, "stage modes")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(samples_us: list[float]) -> dict[str, float]:
    return {
        "p50_us": _percentile(samples_us, 0.50),
        "p90_us": _percentile(samples_us, 0.90),
        "p99_us": _percentile(samples_us, 0.99),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _time_cuda(operation: Callable[[], Any], repetitions: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _slowest_rank_samples(
    local_samples: list[float], cpu_group: dist.ProcessGroup
) -> list[float] | None:
    per_rank: list[list[float] | None] = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(per_rank, local_samples, group=cpu_group)
    if dist.get_rank(cpu_group) != 0:
        return None
    samples = [item for item in per_rank if item is not None]
    return [max(values) for values in zip(*samples, strict=True)]


def _make_selected_slots(
    *,
    query_tokens: int,
    topk: int,
    local_tokens: int,
    rows_per_rank: int,
    world_size: int,
    selection_pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.arange(topk, device=device, dtype=torch.int64)
    if selection_pattern == "shared":
        local = (selected * 8191 + 1234) % local_tokens
        local = local.expand(query_tokens, -1).contiguous()
    else:
        query_offsets = torch.arange(
            query_tokens, device=device, dtype=torch.int64
        ).view(-1, 1)
        local = (selected * 8191 + query_offsets * 2053 + 1234) % local_tokens
    local = local.to(torch.int32)
    owner = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .expand(query_tokens, -1)
        .contiguous()
    )
    owner += torch.arange(query_tokens, device=device, dtype=torch.int32).view(-1, 1)
    owner %= world_size
    peer = owner * (rows_per_rank * PAGE_SIZE) + local
    return local, peer


def _make_owner_history_slots(
    *,
    query_tokens: int,
    topk: int,
    history_tokens: int,
    rows_per_rank: int,
    world_size: int,
    selection_pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = torch.arange(topk, device=device, dtype=torch.int64)
    if selection_pattern == "shared":
        logical = (selected * 8191 + 1234) % history_tokens
        logical = logical.expand(query_tokens, -1).contiguous()
    else:
        query_offsets = torch.arange(
            query_tokens, device=device, dtype=torch.int64
        ).view(-1, 1)
        logical = (selected * 8191 + query_offsets * 2053 + 1234) % history_tokens

    def logical_to_peer(slots: torch.Tensor) -> torch.Tensor:
        page = slots // PAGE_SIZE
        offset = slots % PAGE_SIZE
        owner = page % world_size
        local_page = page // world_size
        return ((owner * rows_per_rank + local_page) * PAGE_SIZE + offset).to(
            torch.int32
        )

    all_logical = torch.arange(history_tokens, device=device, dtype=torch.int64)
    return (
        logical.to(torch.int32),
        logical_to_peer(logical),
        logical_to_peer(all_logical),
    )


def _fill_flashinfer_cache(local_view: torch.Tensor) -> None:
    generator = torch.Generator(device=local_view.device)
    generator.manual_seed(5678)
    values = torch.randn(
        local_view.shape,
        generator=generator,
        device=local_view.device,
        dtype=torch.float32,
    )
    local_view.copy_((values * 0.1).to(torch.float8_e4m3fn))


def _fill_flashmla_cache(local_view: torch.Tensor) -> None:
    blocks, page, _ = local_view.shape
    generator = torch.Generator(device=local_view.device)
    generator.manual_seed(5678)
    nope = (
        torch.randn(
            (blocks, page, 512),
            generator=generator,
            device=local_view.device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.ones(
        (blocks, page, 4), device=local_view.device, dtype=torch.float32
    )
    rope = (
        torch.randn(
            (blocks, page, 64),
            generator=generator,
            device=local_view.device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    local_view[..., :512].copy_(nope.view(torch.uint8))
    local_view[..., 512:528].copy_(scales.view(torch.uint8))
    local_view[..., 528:].copy_(rope.view(torch.uint8))


def _assert_close(
    reference: torch.Tensor, candidate: torch.Tensor, label: str
) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    if not torch.isfinite(reference_float).all():
        raise AssertionError("local attention output is not finite")
    torch.testing.assert_close(
        candidate_float,
        reference_float,
        rtol=2e-2,
        atol=2e-2,
        msg=lambda message: f"{label}: {message}",
    )
    difference = (candidate_float - reference_float).abs()
    denominator = reference_float.abs().clamp_min(1e-5)
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_relative_error": (difference / denominator).max().item(),
    }


def _make_flashinfer_runner(
    *,
    query_tokens: int,
    topk: int,
    workspace_mib: int,
    device: torch.device,
) -> tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    generator = torch.Generator(device=device)
    generator.manual_seed(9012)
    query = (
        torch.randn(
            (query_tokens, 1, NUM_HEADS, HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    workspace = torch.empty(
        workspace_mib * 1024 * 1024, dtype=torch.int8, device=device
    )
    seq_lens = torch.full((query_tokens,), topk, dtype=torch.int32, device=device)

    def run(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        output = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=cache.unsqueeze(1),
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=VALUE_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=indices.view(query_tokens, 1, topk),
            seq_lens=seq_lens,
            max_seq_len=topk,
            bmm1_scale=(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
            bmm2_scale=1.0,
            sparse_mla_top_k=topk,
        )
        assert isinstance(output, torch.Tensor)
        return output

    return query, run


def _make_flashmla_runner(
    *,
    query_tokens: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        raise RuntimeError(f"FlashMLA sparse is unavailable: {reason}")
    generator = torch.Generator(device=device)
    generator.manual_seed(9012)
    query = (
        torch.randn(
            (query_tokens, 1, NUM_HEADS, HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    metadata, num_splits = fm.get_mla_metadata()

    def run(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        output, _ = fm.flash_mla_with_kvcache(
            q=query,
            k_cache=cache.unsqueeze(-2),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=VALUE_DIM,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=indices.view(query_tokens, 1, topk),
            softmax_scale=HEAD_DIM**-0.5,
        )
        return output

    return query, run


def _run_case(
    *,
    backend: str,
    query_tokens: int,
    selection_pattern: str,
    stage_mode: str,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group: dist.ProcessGroup,
) -> dict[str, Any] | None:
    local_blocks = args.local_tokens // PAGE_SIZE
    if backend == "flashinfer":
        dtype = torch.float8_e4m3fn
        width = HEAD_DIM
        fill_cache = _fill_flashinfer_cache
    else:
        dtype = torch.uint8
        width = FLASHMLA_BYTES_PER_TOKEN
        fill_cache = _fill_flashmla_cache

    allocation: RankMajorPeerView | None = None
    try:
        allocation = create_rank_major_peer_view(
            (local_blocks, PAGE_SIZE, width),
            dtype=dtype,
            group=cpu_group,
            first_dim_multiple=1,
            map_rank_local=False,
            require_native_atomics=False,
            device=device,
        )
        assert allocation.local_view is not None
        assert allocation.global_view is not None
        local_cache = allocation.local_view[:local_blocks]
        fill_cache(local_cache)
        dist.barrier(group=cpu_group)

        if stage_mode == "selected":
            local_slots, peer_slots = _make_selected_slots(
                query_tokens=query_tokens,
                topk=args.topk,
                local_tokens=args.local_tokens,
                rows_per_rank=allocation.rows_per_rank,
                world_size=world_size,
                selection_pattern=selection_pattern,
                device=device,
            )
            local_indices = local_slots
            peer_indices = peer_slots
            staged_rows_count = (
                args.topk if selection_pattern == "shared" else query_tokens * args.topk
            )
            if selection_pattern == "shared":
                staged_indices = (
                    torch.arange(args.topk, dtype=torch.int32, device=device)
                    .expand(query_tokens, -1)
                    .contiguous()
                )
                gather_slots = peer_slots[0]
            else:
                staged_indices = torch.arange(
                    staged_rows_count, dtype=torch.int32, device=device
                ).view(query_tokens, args.topk)
                gather_slots = peer_slots.view(-1)
        else:
            history_tokens = args.local_tokens * world_size
            local_indices, peer_indices, gather_slots = _make_owner_history_slots(
                query_tokens=query_tokens,
                topk=args.topk,
                history_tokens=history_tokens,
                rows_per_rank=allocation.rows_per_rank,
                world_size=world_size,
                selection_pattern=selection_pattern,
                device=device,
            )
            staged_rows_count = history_tokens
            staged_indices = local_indices
        staged_cache = torch.empty(
            (staged_rows_count // PAGE_SIZE, PAGE_SIZE, width),
            dtype=dtype,
            device=device,
        )
        staged_rows = staged_cache.view(staged_rows_count, width)
        peer_rows = allocation.global_view.view(-1, width)

        if backend == "flashinfer":
            _, run_attention = _make_flashinfer_runner(
                query_tokens=query_tokens,
                topk=args.topk,
                workspace_mib=args.workspace_mib,
                device=device,
            )
        else:
            _, run_attention = _make_flashmla_runner(
                query_tokens=query_tokens,
                topk=args.topk,
                device=device,
            )

        def gather_peer_rows() -> None:
            torch.index_select(peer_rows, 0, gather_slots.long(), out=staged_rows)

        def run_local() -> torch.Tensor:
            cache = staged_cache if stage_mode == "history" else local_cache
            return run_attention(cache, local_indices)

        def run_peer() -> torch.Tensor:
            return run_attention(allocation.global_view, peer_indices)

        def run_staged() -> torch.Tensor:
            gather_peer_rows()
            return run_attention(staged_cache, staged_indices)

        if stage_mode == "history":
            gather_peer_rows()
            torch.cuda.synchronize(device)
        local_output = run_local().clone()
        peer_output = run_peer().clone()
        staged_output = run_staged().clone()
        torch.cuda.synchronize(device)
        peer_error = _assert_close(local_output, peer_output, "peer")
        staged_error = _assert_close(local_output, staged_output, "staged")

        operations = {
            "local": run_local,
            "peer": run_peer,
            "gather_only": gather_peer_rows,
            "gather_then_local": run_staged,
        }
        timings: dict[str, Any] = {}
        for name, operation in operations.items():
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize(device)
            dist.barrier(group=cpu_group)
            local_samples = _time_cuda(operation, args.repetitions)
            slowest = _slowest_rank_samples(local_samples, cpu_group)
            if rank == 0:
                assert slowest is not None
                timings[name] = {
                    "latency": _summary(slowest),
                    "slowest_rank_samples_us": slowest,
                }
        dist.barrier(group=cpu_group)
        if rank != 0:
            return None

        local_p50 = timings["local"]["latency"]["p50_us"]
        peer_p50 = timings["peer"]["latency"]["p50_us"]
        staged_p50 = timings["gather_then_local"]["latency"]["p50_us"]
        return {
            "backend": backend,
            "query_tokens": query_tokens,
            "selection_pattern": selection_pattern,
            "stage_mode": stage_mode,
            "topk": args.topk,
            "local_history_tokens_per_rank": args.local_tokens,
            "remote_selected_fraction": (world_size - 1) / world_size,
            "selection_reuse_across_query_rows": (
                query_tokens if selection_pattern == "shared" else 1
            ),
            "staged_rows": staged_rows_count,
            "global_history_tokens": args.local_tokens * world_size,
            "correctness": {
                "peer_vs_local": peer_error,
                "staged_vs_local": staged_error,
            },
            "timings": timings,
            "peer_over_local_slowdown_x": peer_p50 / local_p50,
            "peer_over_local_regression_percent": (peer_p50 / local_p50 - 1.0) * 100.0,
            "staged_over_local_slowdown_x": staged_p50 / local_p50,
            "staged_over_local_regression_percent": (staged_p50 / local_p50 - 1.0)
            * 100.0,
            "staged_over_peer_speedup_x": peer_p50 / staged_p50,
        }
    finally:
        if allocation is not None:
            torch.cuda.synchronize(device)
            dist.barrier(group=cpu_group)
            allocation.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-tokens",
        type=_parse_int_list,
        default=_parse_int_list("1,16,128,512"),
    )
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=_parse_backends("flashinfer,flashmla"),
    )
    parser.add_argument(
        "--selection-patterns",
        type=_parse_selection_patterns,
        default=_parse_selection_patterns("shared"),
    )
    parser.add_argument(
        "--stage-modes",
        type=_parse_stage_modes,
        default=_parse_stage_modes("selected"),
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--local-tokens", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.topk <= 0 or args.topk % PAGE_SIZE != 0:
        parser.error(f"--topk must be a positive multiple of {PAGE_SIZE}")
    if args.local_tokens % PAGE_SIZE != 0:
        parser.error(f"--local-tokens must be a multiple of {PAGE_SIZE}")
    if "selected" in args.stage_modes and args.local_tokens < args.topk:
        parser.error("--local-tokens must be >= topk for selected-row staging")
    if (
        "history" in args.stage_modes
        and args.local_tokens * args.expected_world_size < args.topk
    ):
        parser.error("global history must be >= topk for full-history staging")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be nonnegative and repetitions must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this benchmark with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected {args.expected_world_size} ranks, got {world_size}"
        )

    results: list[dict[str, Any]] = []
    try:
        for backend in args.backends:
            for stage_mode in args.stage_modes:
                for selection_pattern in args.selection_patterns:
                    for query_tokens in args.query_tokens:
                        if rank == 0:
                            print(
                                f"benchmarking backend={backend} "
                                f"stage={stage_mode} "
                                f"selection={selection_pattern} "
                                f"query_tokens={query_tokens}",
                                file=sys.stderr,
                                flush=True,
                            )
                        result = _run_case(
                            backend=backend,
                            query_tokens=query_tokens,
                            selection_pattern=selection_pattern,
                            stage_mode=stage_mode,
                            args=args,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            cpu_group=cpu_group,
                        )
                        if result is not None:
                            results.append(result)
    finally:
        dist.barrier(group=cpu_group)
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()

    if rank == 0:
        report = {
            "benchmark": "pcp_peer_sparse_mla",
            "device": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "scope": (
                "same-query sparse MLA local versus rank-major CUDA VMM peer "
                "reads; staged path reuses one selected set across query rows"
            ),
            "results": results,
        }
        encoded = json.dumps(report, indent=2, sort_keys=True)
        print(encoded)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
