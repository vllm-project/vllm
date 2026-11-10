# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_managed_cache_survives_distributed_restart(
    vllm_runner, monkeypatch, tmp_path, tp, pp
):
    """Each rank persists managed winners and reuses them after worker restart."""
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompts = ["The capital of France is", "In Python, an empty list is"]
    results = []
    cache_entries = []
    for _ in range(2):
        with vllm_runner(
            "nvidia/Qwen3-8B-FP8",
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            distributed_executor_backend="mp",
            max_model_len=1024,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            kernel_config={
                "enable_flashinfer_autotune": True,
                "linear_backend": "flashinfer_cutlass",
            },
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        ) as model:
            results.append(model.generate_greedy_logprobs(prompts, 16, 5))
        entries = {
            p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in tmp_path.rglob("*.json")
            if "entries" in p.parts
        }
        assert entries, "No managed winners were published"
        for rank in range(2):
            assert any(f"managed_dp0_rank{rank}" in p.parts for p in entries)
        cache_entries.append(entries)

    assert cache_entries[0] == cache_entries[1], "Warm restart rewrote tuned entries"
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="cold managed cache",
        name_1="warm managed cache",
    )
