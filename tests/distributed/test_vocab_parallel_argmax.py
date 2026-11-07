# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``LogitsProcessor.get_top_tokens`` (``use_local_argmax_reduction``) must
select the same token as the full-vocab greedy path,
``LogitsProcessor(lm_head, h).argmax(-1)``, with a real vocab-parallel
lm_head and vLLM's TP communicators, eagerly and from a captured CUDA graph.
Tie-heavy integer weights stress the lowest-id tie-break across ranks."""

import pytest
import ray
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

HIDDEN_SIZE = 4096


def _hidden(num_rows: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    # Identical on all ranks, as for the drafter.
    return torch.randint(-1, 2, (num_rows, HIDDEN_SIZE), generator=g, device=device).to(
        torch.bfloat16
    )


def _make_head(vocab_size: int, device: torch.device):
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        lm_head = ParallelLMHead(vocab_size, HIDDEN_SIZE, params_dtype=torch.bfloat16)
        logits_processor = LogitsProcessor(vocab_size)
    # Same full weight on every rank; the loader keeps this rank's shard.
    gen = torch.Generator(device=device).manual_seed(1234)
    weight = torch.randint(
        -2, 3, (vocab_size, HIDDEN_SIZE), generator=gen, device=device
    ).to(torch.bfloat16)
    lm_head.weight_loader(lm_head.weight, weight)
    return lm_head, logits_processor


@ray.remote(num_gpus=1, max_calls=1)
def get_top_tokens_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        # 32001 leaves vocab padding on the last shard.
        for vocab_size in (131072, 32001):
            lm_head, lp = _make_head(vocab_size, device)
            assert lm_head.tp_size == tp_size
            for num_rows in (1, 2, 4, 8):
                for trial in range(10):
                    hs = _hidden(num_rows, trial + 7 * num_rows, device)
                    got = lp.get_top_tokens(lm_head, hs)
                    assert got.dtype == torch.int64
                    ref = lp(lm_head, hs).argmax(dim=-1)
                    assert torch.equal(got, ref), (vocab_size, num_rows, trial)

        num_rows = 4
        static = torch.zeros(num_rows, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        with graph_capture(device=device) as graph_capture_context:
            lp.get_top_tokens(lm_head, static)  # warm-up on the capture stream
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                out = lp.get_top_tokens(lm_head, static)
        for trial in range(50):
            static.copy_(_hidden(num_rows, 1000 + trial, device))
            graph.replay()
            assert torch.equal(out, lp(lm_head, static).argmax(dim=-1)), trial
        torch.accelerator.synchronize()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_get_top_tokens_matches_full_argmax(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, get_top_tokens_worker)
