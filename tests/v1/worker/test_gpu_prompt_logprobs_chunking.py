# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked prompt-logprobs must match the unchunked result exactly.

`_get_prompt_logprobs_dict` materializes full-vocabulary logits in row
chunks so the float32 score tensor cannot dominate the activation peak
(issue #5907). Chunking rewrites the destination row mapping, so the
regression risk is misaligned rows rather than wrong values -- these
tests pin the mapping by comparing against a single-chunk run.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.utils import torch_utils
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

VOCAB_SIZE = 64
HIDDEN_SIZE = 8


@pytest.fixture(autouse=True)
def _no_pin_memory(monkeypatch):
    """Run the host-to-device copy unpinned.

    `_get_prompt_logprobs_dict` calls `async_tensor_h2d`, which pins the
    source buffer when `PIN_MEMORY` is set. Pinning requires a CUDA
    allocator, so leaving it enabled makes this test unrunnable on
    CPU-only machines. The chunking logic under test is unaffected.
    """
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)


def _make_runner(logprobs_mode: str, prompt_len: int, num_scheduled: int):
    """Build the minimal runner state `_get_prompt_logprobs_dict` reads."""
    torch.manual_seed(0)
    # A deterministic projection so compute_logits is reproducible across
    # chunk sizes: identical rows in must give identical logits out.
    weight = torch.randn(HIDDEN_SIZE, VOCAB_SIZE)

    request = SimpleNamespace(
        prompt_token_ids=list(range(prompt_len)),
        num_computed_tokens=0,
        in_progress_prompt_logprobs_cpu=None,
    )

    from vllm.v1.sample.sampler import Sampler

    runner = object.__new__(GPUModelRunner)
    runner.num_prompt_logprobs = {"req0": 3}
    runner.requests = {"req0": request}
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(compute_logits=lambda h: h @ weight)
    runner.sampler = SimpleNamespace(
        compute_logprobs=Sampler.compute_logprobs,
        gather_logprobs=Sampler.gather_logprobs,
    )
    runner.model_config = SimpleNamespace(logprobs_mode=logprobs_mode)
    runner.input_batch = SimpleNamespace(req_id_to_index={"req0": 0})
    runner.query_start_loc = SimpleNamespace(np=np.array([0], dtype=np.int32))
    runner._sync_device = lambda: None

    hidden_states = torch.randn(num_scheduled, HIDDEN_SIZE)
    return runner, hidden_states, request


def _run(monkeypatch, logprobs_mode, prompt_len, num_scheduled, chunk_size):
    monkeypatch.setattr(envs, "VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", chunk_size)
    runner, hidden_states, _request = _make_runner(
        logprobs_mode, prompt_len, num_scheduled
    )
    out = runner._get_prompt_logprobs_dict(hidden_states, {"req0": num_scheduled})
    tensors = out["req0"]
    assert isinstance(tensors, LogprobsTensors)
    return tensors


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits"])
@pytest.mark.parametrize(
    "chunk_size",
    [
        1,  # degenerate: one row per chunk
        3,  # does not divide evenly -> short final chunk
        4,  # divides evenly
        1024,  # larger than the batch -> single chunk
    ],
)
def test_chunked_matches_unchunked(monkeypatch, logprobs_mode, chunk_size):
    # Schedule the whole prompt in one step so prefill completes and the
    # request's tensors are returned. num_logits is then prompt_len - 1.
    prompt_len, num_scheduled = 13, 13

    reference = _run(
        monkeypatch, logprobs_mode, prompt_len, num_scheduled, chunk_size=10**9
    )
    chunked = _run(
        monkeypatch, logprobs_mode, prompt_len, num_scheduled, chunk_size=chunk_size
    )

    torch.testing.assert_close(chunked.logprobs, reference.logprobs)
    assert torch.equal(chunked.logprob_token_ids, reference.logprob_token_ids)
    assert torch.equal(
        chunked.selected_token_ranks, reference.selected_token_ranks
    )


def test_chunking_bounds_the_materialized_logits(monkeypatch):
    """No single compute_logits call may exceed the configured chunk size."""
    chunk_size = 4
    prompt_len, num_scheduled = 13, 13
    num_logits = prompt_len - 1
    monkeypatch.setattr(envs, "VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", chunk_size)

    runner, hidden_states, _ = _make_runner(
        "raw_logprobs", prompt_len, num_scheduled
    )
    seen_rows = []
    inner = runner.model.compute_logits

    def recording_compute_logits(h):
        seen_rows.append(h.shape[0])
        return inner(h)

    runner.model = SimpleNamespace(compute_logits=recording_compute_logits)
    runner._get_prompt_logprobs_dict(hidden_states, {"req0": num_scheduled})

    assert seen_rows, "compute_logits was never called"
    assert max(seen_rows) <= chunk_size
    # Every row is still covered exactly once.
    assert sum(seen_rows) == num_logits
