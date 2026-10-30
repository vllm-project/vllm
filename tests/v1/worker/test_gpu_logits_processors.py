# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 logits processor tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
TARGET_TOKEN = 7
BIAS = 5.0
BOOST = 100.0
TEMPERATURE = 2.0


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


class RecordingProcessor(LogitsProcessor):
    """Adds BOOST to TARGET_TOKEN and records every lifecycle callback."""

    def __init__(self, vllm_config: Any, req_state: Any):
        self.events: list[str] = []
        self.seen_logits: torch.Tensor | None = None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.events.append(f"add_request:{req_idx}")
        return True

    def apply_staged_writes(self) -> None:
        self.events.append("apply_staged_writes")

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self.events.append("apply")
        self.seen_logits = logits.clone()
        logits[:, TARGET_TOKEN] += BOOST
        return logits


def _make_sampler(processor: RecordingProcessor) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=[processor],
    )


def test_custom_processor_lifecycle_and_pipeline_position():
    """A custom processor's hooks fire in order, it runs after the built-in
    bias stage, and before temperature scaling."""
    proc = RecordingProcessor(None, None)
    sampler = _make_sampler(proc)
    sampler.add_request(
        0,
        SamplingParams(logit_bias={TARGET_TOKEN: BIAS}, temperature=TEMPERATURE),
    )
    sampler.apply_staged_writes()

    num_reqs = 1
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    raw = torch.zeros(num_reqs, VOCAB_SIZE, device=DEVICE)
    out = sampler.apply_sampling_params(
        raw,
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.ones(num_reqs, dtype=np.int64),
    )

    assert proc.events == [
        "add_request:0",
        "apply_staged_writes",
        "apply",
    ]
    # The processor sees the built-in bias applied but no temperature scaling.
    assert proc.seen_logits is not None
    assert proc.seen_logits[0, TARGET_TOKEN].item() == BIAS
    # Its own modification is then temperature-scaled by the sampler.
    assert out[0, TARGET_TOKEN].item() == (BIAS + BOOST) / TEMPERATURE
