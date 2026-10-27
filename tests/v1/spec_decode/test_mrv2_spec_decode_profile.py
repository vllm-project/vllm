# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The profiled activation peak must cover spec-decode verification.

The KV cache is sized from the profiled peak, so a verification step that
needs more memory than profiling measured can run out of memory at high
gpu_memory_utilization.
"""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

NUM_SEQS = 256


def _profiled_peak(worker) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    worker.model_runner.profile_run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def _reset_peak(worker) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def _read_peak(worker) -> int:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _counter(metrics, name: str) -> int:
    return sum(m.value for m in metrics if m.name == name)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("num_speculative_tokens", [4, 8])
def test_profiled_peak_covers_spec_decode(monkeypatch, num_speculative_tokens):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Run the worker in-process so its CUDA memory counters can be read.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": num_speculative_tokens,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        },
        max_num_seqs=NUM_SEQS,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_log_stats=False,
    )
    try:
        (profiled,) = llm.collective_rpc(_profiled_peak)

        # Repetitive prompts make the n-gram drafter propose a full draft for
        # every request, so verification computes the largest logits of the run.
        prompts = [("alpha beta gamma delta " * 40) + f"#{i}" for i in range(NUM_SEQS)]
        # The heaviest sampling a request can combine with spec decode.
        params = SamplingParams(
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            logprobs=5,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repetition_penalty=1.2,
            seed=0,
            max_tokens=64,
            ignore_eos=True,
        )
        (base,) = llm.collective_rpc(_reset_peak)
        llm.generate(prompts, params, use_tqdm=False)
        (peak,) = llm.collective_rpc(_read_peak)
        metrics = llm.get_metrics()
    finally:
        del llm
        cleanup_dist_env_and_memory()

    assert _counter(metrics, "vllm:spec_decode_num_accepted_tokens") > 0
    assert peak - base <= profiled, (
        f"spec-decode activation peak {(peak - base) / 2**20:.1f} MiB exceeds "
        f"the profiled peak {profiled / 2**20:.1f} MiB"
    )
