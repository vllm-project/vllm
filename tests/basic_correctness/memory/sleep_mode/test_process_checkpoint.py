# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Linux/NVIDIA driver integration test, with exclusive GPU access.

Run with VLLM_TEST_CUDA_PROCESS_CHECKPOINT=1. The driver must expose CUDA
process checkpoint APIs. Override VLLM_TEST_MODEL and VLLM_TEST_SLEEP_CYCLES
for larger-model validation; this test requires enough RAM for weight backups.
The selected model must support batch invariance. Hybrid GDN models require
separate paired accuracy evaluation because that mode is unsupported for GDN.
"""

import os
import pathlib
import time

import pytest

import vllm
import vllm.platforms
import vllm.v1.engine
from vllm import LLM, SamplingParams


def _graph_identity(worker):
    import torch

    if not hasattr(worker, "_checkpoint_test_replays"):
        worker._checkpoint_test_replays = 0
        worker._checkpoint_test_captures = 0
        replay = torch.cuda.CUDAGraph.replay
        capture = torch.cuda.CUDAGraph.capture_begin

        def counted_replay(graph):
            worker._checkpoint_test_replays += 1
            return replay(graph)

        def counted_capture(graph, *args, **kwargs):
            worker._checkpoint_test_captures += 1
            return capture(graph, *args, **kwargs)

        torch.cuda.CUDAGraph.replay = counted_replay
        torch.cuda.CUDAGraph.capture_begin = counted_capture
    torch.accelerator.synchronize()
    graphs = worker.model_runner.cudagraph_manager.graphs
    return {
        "pid": os.getpid(),
        "graphs": sorted(g.raw_cuda_graph_exec() for g in graphs.values()),
        "replays": worker._checkpoint_test_replays,
        "captures": worker._checkpoint_test_captures,
    }


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_CUDA_PROCESS_CHECKPOINT") != "1",
    reason="requires exclusive GPUs and a checkpoint-capable NVIDIA driver",
)
@pytest.mark.parametrize(
    "tp,transport", [(1, "p2p"), (2, "p2p"), (4, "p2p"), (2, "shm"), (4, "shm")]
)
def test_process_checkpoint_preserves_outputs_and_graphs(tp, transport, monkeypatch):
    import torch

    if torch.accelerator.device_count() < tp:
        pytest.skip(f"requires {tp} GPUs")
    config = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",  # Test observation RPC only.
        "VLLM_BATCH_INVARIANT": "1",
        "NCCL_P2P_DISABLE": "1" if transport == "shm" else "0",
        "NCCL_SHM_DISABLE": "0" if transport == "shm" else "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_CUMEM_HOST_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
        "VLLM_USE_NCCL_SYMM_MEM": "0",
    }
    for key, value in config.items():
        monkeypatch.setenv(key, value)
    llm = LLM(
        os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        dtype="bfloat16",
        tensor_parallel_size=tp,
        distributed_executor_backend=(
            "vllm.v1.executor.cuda_checkpoint_executor.CUDACheckpointExecutor"
        ),
        disable_custom_all_reduce=tp > 1,
        enable_nccl_comm_suspend=tp > 1,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=1024**3,
        max_model_len=256,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "pass_config": {"fuse_allreduce_rms": False},
        },
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)

        def generate():
            outputs = llm.generate(
                ["The capital of France is", "Compute 7 times 8:"],
                params,
                use_tqdm=False,
            )
            return [output.outputs[0].token_ids for output in outputs]

        reference = generate()
        graphs = llm.collective_rpc(_graph_identity)
        assert all(rank["graphs"] for rank in graphs)
        for cycle in range(int(os.environ.get("VLLM_TEST_SLEEP_CYCLES", "3"))):
            assert generate() == reference
            llm.sleep(level=1)
            llm.sleep(level=1)
            order = (("weights", "kv_cache"), ("kv_cache", "weights"), (None,))[
                cycle % 3
            ]
            for tag in order:
                llm.wake_up(tags=[tag] if tag else None)
            awake = llm.collective_rpc(_graph_identity)
            assert generate() == reference
            after = llm.collective_rpc(_graph_identity)
            for original, before_generate, current in zip(
                graphs, awake, after, strict=True
            ):
                assert current["pid"] == original["pid"]
                assert current["graphs"] == original["graphs"]
                assert current["replays"] > before_generate["replays"]
                assert current["captures"] == 0
            graphs = after
        llm.sleep(level=1)
    finally:
        llm.llm_engine.engine_core.shutdown()


def _get_cpu_offload_mappings(worker: object) -> list[str]:
    """Read the worker's CPU offload mappings.

    Args:
        worker: The worker supplied by collective_rpc.

    Returns:
        CPU offload mapping entries from the process's memory map.

    """
    return [
        line
        for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
        if "/dev/shm/vllm_offload_" in line
    ]


@pytest.mark.skipif(
    not vllm.platforms.current_platform.is_cuda(),
    reason="CPU-tier process checkpoints require CUDA",
)
@pytest.mark.parametrize("async_init", [False, True])
@pytest.mark.parametrize("tp", [1, 2])
def test_checkpoint_releases_native_cpu_tier(
    async_init: bool, tp: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discard host mappings during checkpoint and restore identical output.

    Args:
        async_init: Whether restored host memory is pinned in the background.
        tp: Number of tensor-parallel workers sharing the host tier.
        monkeypatch: Fixture for process-wide vLLM test settings.

    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    llm = vllm.LLM(
        os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        enable_sleep_mode=True,
        tensor_parallel_size=tp,
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=True,
        disable_log_stats=False,
        kv_cache_memory_bytes=128 * 1024**2,
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": 128 * 1024**2,
                "async_init": async_init,
            },
        },
    )
    params = vllm.SamplingParams(temperature=0, max_tokens=16)
    prompt = "Remember this sentence about Paris. " * 40 + "The capital of France is"
    reference = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
    assert all(llm.collective_rpc(_get_cpu_offload_mappings))
    core = llm.llm_engine.engine_core
    try:
        for _ in range(3):
            assert core.call_utility("checkpoint_prepare", "abort") is None
            deadline = time.monotonic() + 60
            while True:
                state, error = core.call_utility("checkpoint_state")
                assert error is None
                if state == vllm.v1.engine.CheckpointState.PREPARED:
                    break
                assert time.monotonic() < deadline, state
                time.sleep(0.01)
            assert not any(llm.collective_rpc(_get_cpu_offload_mappings))
            core.call_utility("checkpoint_restore")
            assert all(llm.collective_rpc(_get_cpu_offload_mappings))
            actual = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
            assert actual.token_ids == reference.token_ids
            assert llm.reset_prefix_cache(reset_connector=False)
            cached = llm.generate(prompt, params, use_tqdm=False)[0]
            assert cached.outputs[0].token_ids == reference.token_ids
            assert cached.num_cached_tokens > 0
    finally:
        core.shutdown()
