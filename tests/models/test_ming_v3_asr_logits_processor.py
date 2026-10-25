# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming ASR's no-repeat constraint must preserve MRV2 request isolation."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    NoRepeatNGramLogitsProcessor as HFNoRepeat,
)

from vllm.config import VllmConfig
from vllm.model_executor.models.ming_v3_asr_logits_processor import (
    NoRepeatNGramLogitsProcessor,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState


@pytest.fixture
def processor_state():
    config = VllmConfig()
    device = torch.device(current_platform.device_type)
    requests = RequestState(4, 64, 64, 0, 128, device)
    processor = NoRepeatNGramLogitsProcessor(
        config, LogitsProcRequestState.from_request_state(requests)
    )
    return config, requests, processor


def _request_context(requests, req_ids):
    slots = [requests.req_id_to_index[req_id] for req_id in req_ids]
    lengths = requests.prefill_len.np[slots]
    indices = torch.tensor(slots, dtype=torch.int32, device=requests.device)
    pos = torch.tensor(lengths - 1, dtype=torch.int32, device=requests.device)
    return LogitsContext(
        expanded_idx_mapping=indices,
        idx_mapping=indices,
        idx_mapping_np=np.array(slots),
        expanded_local_pos=torch.zeros_like(indices),
        input_ids=requests.all_token_ids.gpu[indices, pos],
        pos=pos,
        seq_lens_upper_bound_np=lengths,
    )


@pytest.mark.parametrize("size", [1, 2, 10])
@pytest.mark.parametrize(
    "history",
    [
        pytest.param([1, 2, 3] * 10, id="in_vocab"),
        pytest.param([128, 1, 2, 200, 1, 2] * 5, id="outside_logits_vocab"),
    ],
)
def test_ngram_matches_hf_with_reordered_rows(processor_state, size, history):
    _, requests, processor = processor_state
    device = requests.device
    for req_id, n in (("blocked", size), ("ordinary", 0)):
        requests.add_request(req_id, 12, history, len(history), 16)
        processor.add_request(
            requests.req_id_to_index[req_id],
            SamplingParams(extra_args={"no_repeat_ngram_size": n}),
        )
    requests.apply_staged_writes()
    processor.apply_staged_writes()
    ctx = _request_context(requests, ["ordinary", "blocked"])
    expected = torch.zeros(2, 128, device=device)
    expected[1:] = HFNoRepeat(size)(
        torch.tensor([history], device=device), expected[1:]
    )
    actual = processor.apply(torch.zeros_like(expected), ctx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, 2])
def test_recycled_slot_uses_new_history_and_options(processor_state, size):
    _, requests, processor = processor_state
    history = [1, 2, 3] * 10
    requests.add_request("old", 12, history, len(history), 16)
    slot = requests.req_id_to_index["old"]
    processor.add_request(slot, SamplingParams(extra_args={"no_repeat_ngram_size": 2}))
    requests.apply_staged_writes()
    processor.apply_staged_writes()
    logits = torch.zeros(1, 128, device=requests.device)
    expected = torch.zeros_like(logits)
    expected[0, 1] = -float("inf")
    torch.testing.assert_close(
        processor.apply(logits.clone(), _request_context(requests, ["old"])),
        expected,
        rtol=0,
        atol=0,
    )

    assert requests.remove_request("old") == slot
    history = [4, 5, 4, 5]
    requests.add_request("new", len(history), history, len(history), 16)
    assert requests.req_id_to_index["new"] == slot
    processor.add_request(
        slot, SamplingParams(extra_args={"no_repeat_ngram_size": size})
    )
    requests.apply_staged_writes()
    processor.apply_staged_writes()
    expected = torch.zeros_like(logits)
    if size:
        expected[0, 4] = -float("inf")
    torch.testing.assert_close(
        processor.apply(logits, _request_context(requests, ["new"])),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("size", [0, 10])
def test_sampler_applies_blocking_with_logit_bias_and_temperature(
    processor_state, size
):
    config, requests, processor = processor_state
    history = [7] * 10
    requests.add_request("req", len(history), history, len(history), 16)
    sampler = Sampler(
        config,
        requests.max_num_reqs,
        requests.vocab_size,
        requests.device,
        requests,
        custom_logits_processors=[processor],
    )
    sampler.add_request(
        requests.req_id_to_index["req"],
        SamplingParams(
            temperature=2.0,
            logit_bias={7: 100, 8: 20},
            extra_args={"no_repeat_ngram_size": size},
        ),
    )
    requests.apply_staged_writes()
    sampler.apply_staged_writes()
    ctx = _request_context(requests, ["req"])
    logits = torch.zeros(1, requests.vocab_size, device=requests.device)
    actual = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping=ctx.expanded_idx_mapping,
        idx_mapping=ctx.idx_mapping,
        idx_mapping_np=ctx.idx_mapping_np,
        pos=ctx.pos,
        input_ids=ctx.input_ids,
        expanded_local_pos=ctx.expanded_local_pos,
        seq_lens_upper_bound_np=ctx.seq_lens_upper_bound_np,
    )
    expected = torch.zeros_like(logits)
    expected[0, 7] = -float("inf") if size else 50
    expected[0, 8] = 10
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_processor_import_stays_frontend_safe():
    """Loading Ming's sampler plugin must not initialize its model or GPU state."""
    code = (
        "import sys\n"
        "from vllm.v1.worker.gpu.sample.logits_processor.loader import "
        "build_custom_logits_processors_params_validator\n"
        "build_custom_logits_processors_params_validator(["
        "'vllm.model_executor.models.ming_v3_asr_logits_processor:"
        "NoRepeatNGramLogitsProcessor'])\n"
        "assert 'vllm.model_executor.models.ming_v3_asr' not in sys.modules\n"
        "assert 'vllm.model_executor.models.ming_audio' not in sys.modules\n"
        "assert 'vllm.v1.worker.gpu.states' not in sys.modules\n"
        "assert 'vllm.v1.worker.gpu.buffer_utils' not in sys.modules\n"
        "import torch\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
