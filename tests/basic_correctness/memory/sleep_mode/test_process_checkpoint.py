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

import pytest

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
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_process_checkpoint_preserves_outputs_and_graphs(tp, monkeypatch):
    import torch

    if torch.accelerator.device_count() < tp:
        pytest.skip(f"requires {tp} GPUs")
    config = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",  # Test observation RPC only.
        "VLLM_BATCH_INVARIANT": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "1",
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
            "vllm.v1.executor.process_checkpoint_executor.ProcessCheckpointExecutor"
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
