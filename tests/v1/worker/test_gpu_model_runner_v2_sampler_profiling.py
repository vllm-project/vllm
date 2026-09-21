# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE, VOCAB, NUM_REQS = torch.device("cuda"), 4096, 8


class _Probe(LogitsProcessor):
    def __init__(self) -> None:
        self.dtypes: list[torch.dtype] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return False

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self.dtypes.append(logits.dtype)
        return logits


def _peak_bytes(run) -> int:
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    run()
    torch.accelerator.synchronize()
    return torch.accelerator.max_memory_allocated() - before


def test_dummy_sampler_run_profiles_the_warmup_sampler_path():
    probe = _Probe()
    reasoning = SimpleNamespace(
        reasoning_start_token_ids=[90],
        reasoning_end_token_ids=[91],
        natural_reasoning_end_token_ids=[91],
    )
    req_states = RequestState(
        max_num_reqs=NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=NUM_REQS,
        num_speculative_steps=1,
        vocab_size=VOCAB,
        device=DEVICE,
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=reasoning),
        max_num_reqs=NUM_REQS,
        vocab_size=VOCAB,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=[probe],
    )
    runner.input_buffers = InputBuffers(
        max_num_reqs=NUM_REQS, max_num_tokens=NUM_REQS, device=DEVICE
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda h: torch.zeros(
            h.shape[0], VOCAB, dtype=torch.float16, device=DEVICE
        )
    )
    hidden = torch.zeros(NUM_REQS, 8, dtype=torch.float16, device=DEVICE)

    def before_this_change() -> None:
        # make_dummy registers no sampling params, so the pipeline early-exits.
        with torch.inference_mode():
            batch = InputBatch.make_dummy(NUM_REQS, NUM_REQS, runner.input_buffers)
            runner.sampler(runner.model.compute_logits(hidden), batch)

    baseline = _peak_bytes(before_this_change)
    assert probe.dtypes == []

    profiled = _peak_bytes(lambda: GPUModelRunner._dummy_sampler_run(runner, hidden))
    assert probe.dtypes == [torch.float32]
    # The fp32 logits copy made by apply_sampling_params is now inside the peak.
    assert profiled - baseline >= NUM_REQS * VOCAB * 4
