# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform


def _workspace_state(worker):
    from vllm.v1.worker.workspace import current_workspace_manager

    manager = current_workspace_manager()
    return (
        manager.is_locked(),
        [
            (0, 0) if buffer is None else (buffer.numel(), buffer.data_ptr())
            for buffer in manager._current_workspaces
        ],
        {
            type(layer.impl).__name__
            for layer in worker.model_runner.model.modules()
            if hasattr(layer, "impl")
        },
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA graph regression")
@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 32 * 1024**2])
def test_mla_reserves_inactive_microbatch_before_serving(
    vllm_runner, monkeypatch, kv_cache_memory_bytes
):
    """An inactive DBO slot must have scratch before graph capture locks it."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    prompts = ["Hello", "The capital of France is", "Write a short story about cats."]
    results = []
    for enable_dbo in (False, True):
        # With DP=1 the runner leaves the second slot inactive. This checks its
        # reservation without claiming to exercise multi-rank DBO overlap.
        with vllm_runner(
            "vllm-project/tiny-random-DeepseekV2ForCausalLM",
            dtype="half",
            max_model_len=256,
            max_num_seqs=8,
            max_num_batched_tokens=64,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.5,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            enable_prefix_caching=False,
            attention_config={"backend": "TRITON_MLA"},
            enable_dbo=enable_dbo,
            all2all_backend="deepep_low_latency",
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=2,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [2, 4, 8],
            },
        ) as runner:
            llm = runner.get_llm()
            locked, buffers, implementations = llm.collective_rpc(_workspace_state)[0]
            assert locked
            assert "TritonMLAImpl" in implementations
            assert len(buffers) == (2 if enable_dbo else 1)
            assert all(size > 0 for size, _ in buffers)
            assert len({pointer for _, pointer in buffers}) == len(buffers)
            results.append(runner.generate_greedy_logprobs(prompts, 8, 5))
            assert llm.collective_rpc(_workspace_state)[0][:2] == (locked, buffers)
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="single-slot",
        name_1="inactive DBO slot reserved",
        always_check_logprobs=True,
    )
