# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark SchedulerOutput execute_model RPC pack/unpack vs pickle.

Usage (from vLLM repo root, with vLLM deps installed):

    python benchmarks/overheads/benchmark_scheduler_output_rpc_pack.py
    python benchmarks/overheads/benchmark_scheduler_output_rpc_pack.py \\
        --profile hy3-decode
    python benchmarks/overheads/benchmark_scheduler_output_rpc_pack.py \\
        --iterations 2000

Profiles:
- ``small``: smoke test (~5KB baseline pickle)
- ``hy3-decode``: steady-state decode with 200 requests (~130KB baseline pickle,
  aligned with HY3 Stage4 ``message_bytes`` p50)
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import time

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import (
    CachedRequestData,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
    pack_scheduler_output_for_execute_model_fast_path,
    unpack_scheduler_output_from_execute_model_fast_path,
)


def _make_scheduler_output_small() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="req-0",
                prompt_token_ids=list(range(512)),
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=16),
                pooling_params=None,
                block_ids=([10, 11, 12], [20, 21]),
                num_computed_tokens=0,
                lora_request=None,
                prompt_is_token_ids=[True] * 512,
                prefill_token_ids=list(range(512)),
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[f"req-{i}" for i in range(1, 17)],
            resumed_req_ids={f"req-{i}" for i in range(1, 9)},
            new_token_ids=[[42]] * 16,
            all_token_ids={f"req-{i}": [7, 42] for i in range(1, 17)},
            new_block_ids=[([30, 31],)] * 16,
            num_computed_tokens=[128] * 16,
            num_output_tokens=[1] * 16,
        ),
        num_scheduled_tokens={f"req-{i}": 1 for i in range(17)},
        total_num_scheduled_tokens=17,
        scheduled_spec_decode_tokens={f"req-{i}": [99] for i in range(1, 5)},
        scheduled_encoder_inputs={"req-0": [0, 1]},
        num_common_prefix_blocks=[2, 0],
        finished_req_ids={"req-done"},
        free_encoder_mm_hashes=["mm-hash"],
        scheduled_encoder_input_stats=ScheduledEncoderInputStats(
            num_inputs=1,
            output_tokens=512,
        ),
        preempted_req_ids={"req-preempt"},
        has_structured_output_requests=True,
        pending_structured_output_tokens=False,
        num_invalid_spec_tokens={"req-1": 0},
        new_block_ids_to_zero=[100, 101],
        kv_cache_block_copies=[KVCacheBlockCopy(src_block_id=1, dst_block_id=2)],
        kv_connector_block_state=KVConnectorBlockState(
            block_ids={"req-1": ([10, 11], [20])},
            boundary_state_offloads={"req-1": [(0, 5, 16)]},
        ),
        has_sync_kv_loads=False,
        num_spec_tokens_to_schedule=2,
    )


def _make_scheduler_output_hy3_decode() -> SchedulerOutput:
    """Synthetic 2K/200 steady-state decode step (~130KB baseline pickle).

    Tuned to match HY3 Stage4 ``message_bytes`` p50 (~137KB) for
    ``input=2048, output=1, num_prompts=200``.
    """
    num_cached = 180
    blocks_per_req = 96
    num_new = 5
    prompt_len = 2048

    req_ids = [f"cmpl-{i:04d}-uuid-{i * 7919 % 100000:05d}" for i in range(num_cached)]
    cached = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(req_ids[:8]),
        new_token_ids=[[(i + 100) % 32000] for i in range(num_cached)],
        all_token_ids={
            rid: [(i + j) % 32000 for j in range(16)]
            for i, rid in enumerate(req_ids[:32])
        },
        new_block_ids=[
            tuple(
                [
                    list(
                        range(
                            10000 + i * blocks_per_req,
                            10000 + i * blocks_per_req + blocks_per_req,
                        )
                    )
                ]
            )
            for i in range(num_cached)
        ],
        num_computed_tokens=[2047] * num_cached,
        num_output_tokens=[1] * num_cached,
    )
    new_reqs = []
    for i in range(num_new):
        new_reqs.append(
            NewRequestData(
                req_id=f"new-{i:04d}",
                prompt_token_ids=list(range(prompt_len)),
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=1, temperature=0.7),
                pooling_params=None,
                block_ids=tuple(
                    list(
                        range(
                            5000 + i * 64 + group * 64,
                            5000 + i * 64 + group * 64 + 64,
                        )
                    )
                    for group in range(2)
                ),
                num_computed_tokens=0,
                lora_request=None,
                prompt_is_token_ids=[True] * prompt_len,
                prefill_token_ids=list(range(prompt_len)),
            )
        )
    all_req_ids = [r.req_id for r in new_reqs] + req_ids
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={req_id: 1 + (hash(req_id) % 2) for req_id in all_req_ids},
        total_num_scheduled_tokens=sum(
            1 + (hash(req_id) % 2) for req_id in all_req_ids
        ),
        scheduled_spec_decode_tokens={rid: [1, 2, 3] for rid in req_ids[:40]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[4, 2, 0],
        finished_req_ids={f"done-{i}" for i in range(12)},
        free_encoder_mm_hashes=[f"hash-{i}" for i in range(6)],
        scheduled_encoder_input_stats=ScheduledEncoderInputStats(
            num_inputs=0,
            output_tokens=0,
        ),
        preempted_req_ids=set(),
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        num_invalid_spec_tokens={req_ids[0]: 0},
        new_block_ids_to_zero=list(range(9000, 9020)),
        kv_cache_block_copies=[
            KVCacheBlockCopy(src_block_id=i, dst_block_id=i + 1) for i in range(24)
        ],
        kv_connector_block_state=KVConnectorBlockState(
            block_ids={req_ids[0]: ([9000, 9001], [9100])},
            boundary_state_offloads={req_ids[0]: [(0, 5, 16)]},
        ),
        has_sync_kv_loads=False,
        num_spec_tokens_to_schedule=2,
    )


PROFILE_BUILDERS = {
    "small": _make_scheduler_output_small,
    "hy3-decode": _make_scheduler_output_hy3_decode,
}


def _bench(fn, iterations: int) -> float:
    warmup = min(50, max(1, iterations // 10))
    for _ in range(warmup):
        fn()
    samples_us: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        samples_us.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples_us)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_BUILDERS),
        default="hy3-decode",
        help="small (~5KB) or hy3-decode (~130KB baseline pickle)",
    )
    args = parser.parse_args()

    scheduler_output = PROFILE_BUILDERS[args.profile]()
    baseline_blob = pickle.dumps(scheduler_output, protocol=pickle.HIGHEST_PROTOCOL)
    packed_fast_path = pack_scheduler_output_for_execute_model_fast_path(
        scheduler_output
    )
    fast_path_blob = pickle.dumps(packed_fast_path, protocol=pickle.HIGHEST_PROTOCOL)

    cases = [
        (
            "mode 0 baseline",
            "writer pickle_us (MQ)",
            lambda: len(
                pickle.dumps(scheduler_output, protocol=pickle.HIGHEST_PROTOCOL)
            ),
        ),
        (
            "mode 0 baseline",
            "worker load_us (MQ)",
            lambda: (pickle.loads(baseline_blob), len(baseline_blob))[1],
        ),
        (
            "mode 1 fast path",
            "writer pickle_us (MQ)",
            lambda: len(
                pickle.dumps(packed_fast_path, protocol=pickle.HIGHEST_PROTOCOL)
            ),
        ),
        (
            "mode 1 fast path",
            "worker load_us (MQ)",
            lambda: (pickle.loads(fast_path_blob), len(fast_path_blob))[1],
        ),
        (
            "mode 0 baseline",
            "writer pack+pickle (e2e)",
            lambda: len(
                pickle.dumps(scheduler_output, protocol=pickle.HIGHEST_PROTOCOL)
            ),
        ),
        (
            "mode 1 fast path",
            "writer pack+pickle (e2e)",
            lambda: len(
                pickle.dumps(
                    pack_scheduler_output_for_execute_model_fast_path(scheduler_output),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            ),
        ),
        (
            "mode 1 fast path",
            "worker load+unpack (e2e)",
            lambda: (
                unpack_scheduler_output_from_execute_model_fast_path(
                    pickle.loads(fast_path_blob)
                ),
                len(fast_path_blob),
            )[1],
        ),
    ]

    print(f"profile={args.profile} iterations={args.iterations}")
    print(f"baseline message_bytes={len(baseline_blob)}")
    print(f"mode 1 message_bytes={len(fast_path_blob)}")
    print()
    print("| mode | metric | p50_us_per_iter | message_bytes | speedup vs mode 0 |")
    print("|---|---|---:|---:|---:|")

    baseline_writer_mq: float | None = None
    baseline_worker_mq: float | None = None
    mode1_writer_mq: float | None = None
    mode1_worker_mq: float | None = None

    for mode, metric, fn in cases:
        p50 = _bench(fn, args.iterations)
        msg_bytes = len(baseline_blob) if "mode 0" in mode else len(fast_path_blob)
        speedup = ""
        if metric == "writer pickle_us (MQ)":
            if mode == "mode 0 baseline":
                baseline_writer_mq = p50
            elif mode == "mode 1 fast path" and baseline_writer_mq:
                mode1_writer_mq = p50
                speedup = f"{baseline_writer_mq / p50:.2f}x"
        elif metric == "worker load_us (MQ)":
            if mode == "mode 0 baseline":
                baseline_worker_mq = p50
            elif mode == "mode 1 fast path" and baseline_worker_mq:
                mode1_worker_mq = p50
                speedup = f"{baseline_worker_mq / p50:.2f}x"
        print(f"| {mode} | {metric} | {p50:.1f} | {msg_bytes} | {speedup} |")

    if (
        baseline_writer_mq is not None
        and baseline_worker_mq is not None
        and mode1_writer_mq is not None
        and mode1_worker_mq is not None
    ):
        writer_reduction = (1 - mode1_writer_mq / baseline_writer_mq) * 100
        worker_reduction = (1 - mode1_worker_mq / baseline_worker_mq) * 100
        combined_baseline = baseline_writer_mq + baseline_worker_mq
        combined_mode1 = mode1_writer_mq + mode1_worker_mq
        combined_reduction = (1 - combined_mode1 / combined_baseline) * 100
        print()
        print("PR Test Result (MQ-equivalent: pickle.dumps / pickle.loads only):")
        print("| Mode | Writer pickle (μs) | Worker load (μs) | Combined (μs) |")
        print("|---|---:|---:|---:|")
        print(
            f"| 0 baseline | {baseline_writer_mq:.1f} | "
            f"{baseline_worker_mq:.1f} | {combined_baseline:.1f} |"
        )
        print(
            f"| 1 fast path | {mode1_writer_mq:.1f} | "
            f"{mode1_worker_mq:.1f} | {combined_mode1:.1f} |"
        )
        print()
        direction = "faster" if writer_reduction >= 0 else "slower"
        print(
            f"mode 1 vs mode 0: writer {abs(writer_reduction):.1f}% {direction}, "
            f"worker {worker_reduction:.1f}% faster, "
            f"combined serialize path {combined_reduction:.1f}% faster "
            f"({combined_baseline:.0f} -> {combined_mode1:.0f} μs)"
        )
        print()
        print("HY3 Stage4 reference (2K/200 prompts, production capture):")
        print("| Mode | Writer pickle (μs) | Worker load (μs) | message_bytes |")
        print("|---|---:|---:|---:|")
        print("| 0 baseline | 845 | 1054 | 136904 |")
        print("| 1 fast path | 154 | 169 | 144366 |")


if __name__ == "__main__":
    main()
