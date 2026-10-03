# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU end-to-end coverage of graph-discard sleep and idle recapture."""

import json
import os
import time

from vllm import LLM, SamplingParams


def _graph_snapshot(worker):
    import torch

    manager = worker.model_runner.cudagraph_manager
    if not hasattr(worker, "_test_graph_replays"):
        worker._test_graph_replays = 0
        original = manager.run_fullgraph

        def replay(desc):
            worker._test_graph_replays += 1
            return original(desc)

        manager.run_fullgraph = replay
        # Observe graph release before the backend changes weight/KV mappings
        # or creates CPU backups. Those operations can change driver residency.
        backend = worker.sleep_mode_backend
        suspend = backend.suspend
        worker._test_before_suspend_free = None

        def observe_suspend(level=1):
            worker._test_before_suspend_free = torch.accelerator.get_memory_info()[0]
            return suspend(level)

        backend.suspend = observe_suspend
    return {
        "pid": os.getpid(),
        "graphs": len(manager.graphs),
        "replays": worker._test_graph_replays,
        "free_bytes": torch.accelerator.get_memory_info()[0],
        "before_suspend_free_bytes": worker._test_before_suspend_free,
    }


def test_graph_discard_sleep_recaptures_after_first_request(monkeypatch):
    """Exercise real graphs, partial wake and outputs through the public API."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # The local test probe is a callable RPC, never exposed by the server API.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_SLEEP_DISCARD_GRAPHS", "1")
    model = os.environ.get("SLEEP_GRAPH_TEST_MODEL", "Qwen/Qwen3-0.6B")
    llm = LLM(
        model=model,
        dtype=os.environ.get("SLEEP_GRAPH_TEST_DTYPE", "bfloat16"),
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.8,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The opposite of hot is",
        "The first three positive integers are",
    ]
    params = SamplingParams(temperature=0, max_tokens=16, seed=42)

    def generate():
        return [o.outputs[0].token_ids for o in llm.generate(prompts, params)]

    def snapshot():
        return llm.collective_rpc(_graph_snapshot)[0]

    try:
        reference = generate()
        initial = snapshot()
        assert initial["graphs"] > 0 and all(reference)
        rounds = int(os.environ.get("SLEEP_GRAPH_TEST_ROUNDS", "2"))
        for iteration in range(rounds):
            before = snapshot()
            started = time.perf_counter()
            llm.sleep(level=1)
            sleep_s = time.perf_counter() - started
            asleep = snapshot()
            assert asleep["graphs"] == 0
            assert asleep["free_bytes"] > before["free_bytes"]
            llm.sleep(level=1)
            tags = ["weights", "kv_cache"]
            if iteration % 2:
                tags.reverse()
            llm.wake_up(tags=tags[:1])
            assert snapshot()["graphs"] == 0
            started = time.perf_counter()
            llm.wake_up(tags=tags[1:])
            final_tag_wake_s = time.perf_counter() - started
            awake = snapshot()
            assert awake["graphs"] == 0
            started = time.perf_counter()
            assert generate() == reference
            eager_s = time.perf_counter() - started
            first = snapshot()
            assert first["replays"] == awake["replays"]
            deadline = time.monotonic() + 120
            while first["graphs"] < initial["graphs"]:
                assert time.monotonic() < deadline, "Idle recapture did not finish"
                time.sleep(0.05)
                first = snapshot()
            started = time.perf_counter()
            assert generate() == reference
            graph_s = time.perf_counter() - started
            final = snapshot()
            assert final["replays"] > first["replays"]
            assert final["pid"] == initial["pid"]
            print(
                "SLEEP_GRAPH_RESULT "
                + json.dumps(
                    {
                        "model": model,
                        "iteration": iteration,
                        "timestamp": time.time(),
                        "sleep_s": sleep_s,
                        "final_tag_wake_s": final_tag_wake_s,
                        "eager_s": eager_s,
                        "graph_s": graph_s,
                        "before": before,
                        "asleep": asleep,
                        "awake": awake,
                        "recaptured": final,
                    }
                ),
                flush=True,
            )
    finally:
        llm.llm_engine.engine_core.shutdown()
