# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.benchmarks.forward.batch import build_batch
from vllm.benchmarks.forward.main import run_case
from vllm.benchmarks.forward.profile import summarize
from vllm.benchmarks.forward.shapes import jitter_kv_lens, parse_case
from vllm.platforms import current_platform
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def _group(block_size: int, replay: int = 0, dcp_sharded: bool = False):
    return SimpleNamespace(
        block_size=block_size,
        prefix_replay_tokens=replay,
        dcp_sharded=dcp_sharded,
        manager_config=FullAttentionManager.Config(),
    )


def test_build_batch_layout():
    """The runner sees each request kind as the scheduler would schedule it."""
    case = parse_case("c=2*1@101,4@64:3,32,8@48c,16@316")
    groups = [_group(16), _group(32, replay=128)]
    batch = build_batch(case, groups, 1000, 1000, np.random.default_rng(0))

    # (scheduled, computed, is_decode) per request: decode, verify, prefill,
    # chunked prefill, then a prefix hit replaying the last 128 hit tokens.
    assert list(batch.expected.values()) == [
        (1, 100, True),
        (1, 100, True),
        (4, 60, True),
        (32, 0, False),
        (8, 40, False),
        (16 + 128, 300 - 128, False),
    ]
    output = batch.scheduler_output
    assert output.total_num_scheduled_tokens == 1 + 1 + 4 + 32 + 8 + 144
    assert len(output.scheduled_spec_decode_tokens["bench-2"]) == 3
    assert output.scheduled_new_reqs[-1].replay_start == 300 - 128
    for req_id, (scheduled, _, _) in batch.expected.items():
        assert len(batch.query_token_ids[req_id]) == scheduled
    # Groups share one block pool: block IDs must not overlap.
    ranges = sorted(r for group in batch.group_block_ranges for r in group)
    assert ranges[0][0] == 1
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    with pytest.raises(ValueError, match="KV blocks"):
        build_batch(case, groups, 10, 1000, np.random.default_rng(0))

    # DCP spreads each block's tokens over the DCP ranks, like the scheduler.
    groups = [_group(16, dcp_sharded=True), _group(16)]
    batch = build_batch(
        parse_case("c=1@640"), groups, 1000, 1000, np.random.default_rng(0), 4
    )
    assert batch.group_block_ranges == [[(1, 11)], [(11, 51)]]


def test_jitter_kv_lens():
    """Prefixes vary, capped at the max length; queries and fresh prefills don't.
    A case's draw depends only on the seed and its name, so adding or reordering
    other cases doesn't change it."""
    case = parse_case("c=1000*1@1001,4@64:3,32")
    jittered = jitter_kv_lens(case, 0.5, 1400, 0)
    assert jitter_kv_lens(case, 0.5, 1400, 0) == jittered
    other = jitter_kv_lens(replace(case, name="d"), 0.5, 1400, 0)
    assert other.requests != jittered.requests
    prefixes = np.array([r.prefix_len for r in jittered.requests[:1000]])
    assert prefixes.min() >= 500 and max(r.kv_len for r in jittered.requests) == 1400
    assert prefixes.std() > 200
    assert [r.query_len for r in jittered.requests] == [
        r.query_len for r in case.requests
    ]
    assert jittered.requests[-1] == case.requests[-1]
    assert jitter_kv_lens(case, 0, 2048, 0) == case
    with pytest.raises(ValueError, match="KV length"):
        jitter_kv_lens(case, 0, 1000, 0)


@pytest.mark.parametrize("capacity", [0, 2])
def test_bench_forward_rejects_oversized_verify(capacity):
    """Reject drafts beyond the runner's buffer before launching CUDA kernels."""
    runner = SimpleNamespace(num_speculative_steps=capacity)
    case = parse_case("verify=2*4@64:3")
    with pytest.raises(
        ValueError,
        match=rf"Case 'verify' needs 3 draft tokens.*supports {capacity}",
    ):
        run_case(runner, case, SimpleNamespace())


def test_profile_summary():
    """Each replay's leading restore/rendezvous activities are left out, and
    kernels are grouped by name without template or function arguments."""

    def act(name, ts, dur, cat="kernel"):
        return {"cat": cat, "name": name, "ts": ts, "dur": dur}

    replay = [
        ("Memcpy DtoD (Device -> Device)", "gpu_memcpy", 5),
        ("void ns::gemm<128>(Params)", "kernel", 10),
        ("void ns::gemm<64>(Params)", "kernel", 20),
        ("std::enable_if<(1>0), void>::type ns::norm<float>(float*)", "kernel", 4),
    ]
    events = [{"cat": "cpu_op", "name": "x", "ts": 0, "dur": 1}]
    for r in range(2):
        for i, (name, cat, dur) in enumerate(replay):
            events.append(act(name, 1000 * r + 100 * i, dur, cat))
    summary = summarize(events, replays=2, skip=1)
    assert summary["pre_forward_us"] == 5
    assert summary["busy_us"] == 34
    assert summary["span_us"] == 204
    assert [(k["name"], k["us"], k["calls"]) for k in summary["kernels"]] == [
        ("ns::gemm", 30, 2),
        ("ns::norm", 4, 1),
    ]
    with pytest.raises(ValueError, match="replays"):
        summarize(events[:-1], replays=2, skip=1)


@pytest.mark.benchmark
# TODO(woosuk): Support ROCm and re-enable this test there.
@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Hangs on ROCm; the bench is CUDA-only."
)
def test_bench_forward(tmp_path):
    output = tmp_path / "forward.json"
    command = [
        "vllm",
        "bench",
        "forward",
        "--model",
        MODEL_NAME,
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.5",
        "--case",
        "decode=8*1@512",
        "--case",
        "prefill=256",
        "--case",
        "mix=128@256,4*1@512",
        "--iters",
        "3",
        "--samples",
        "2",
        "--output-json",
        str(output),
        "--profile-dir",
        str(tmp_path / "profile"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    results = {row["case"]: row for row in json.loads(output.read_text())}
    assert list(results) == ["decode", "prefill", "mix"]
    for row in results.values():
        rank0 = row["ranks"][0]
        assert len(rank0["batch_order"]) == len(row["requests"])
        assert rank0["finite"] and rank0["graph_deterministic"]
        assert row["median_us"] > 0
        profile = rank0["profile"]
        assert profile["top_kernels"] and profile["span_us"] > 0
        summary = json.loads(Path(profile["summary"]).read_text())
        assert Path(summary["trace"]).exists()
