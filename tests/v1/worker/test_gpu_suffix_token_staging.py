# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in suffix-only ``all_token_ids`` staging of MRV2."""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.suffix_staging import SuffixTokenStaging

MARGIN = 256
MIN_START = 4096


def _new_req(
    num_computed: int,
    prompt_len: int,
    sampling_params: SamplingParams | None = None,
    **kwargs,
):
    fields = dict(
        prompt_len=prompt_len,
        num_computed_tokens=num_computed,
        prefill_token_ids=list(range(max(prompt_len, num_computed) + 1)),
        sampling_params=sampling_params or SamplingParams(),
        pooling_params=None,
        mm_features=[],
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


@pytest.fixture
def staging():
    return SuffixTokenStaging(MARGIN, MIN_START)


_VLLM_CONFIG = SimpleNamespace(
    watermark_config=None, speculative_config=None, kv_transfer_config=None
)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_SUFFIX_TOKEN_STAGING", raising=False)
    assert SuffixTokenStaging.create(_VLLM_CONFIG, False, 0) is None


def test_margin_covers_model_lookback(monkeypatch):
    # Model states that read tokens below num_computed_tokens keep them staged.
    monkeypatch.setenv("VLLM_SUFFIX_TOKEN_STAGING", "0")
    staging = SuffixTokenStaging.create(_VLLM_CONFIG, False, 3)
    assert staging is not None
    assert staging.get_start(_new_req(10000, 12000)) == 10000 - 3


def test_suffix_start_for_eligible_requests(staging):
    assert staging.get_start(_new_req(10000, 12000)) == 10000 - MARGIN
    # Resumed request (num_computed > prompt_len): output tokens stay staged.
    assert staging.get_start(_new_req(13000, 12000)) == 12000 - MARGIN


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_computed=MIN_START, prompt_len=12000),  # short hit
        dict(num_computed=0, prompt_len=12000),
        dict(num_computed=10000, prompt_len=12000, mm_features=[object()]),
        dict(num_computed=10000, prompt_len=12000, pooling_params=object()),
        dict(
            num_computed=10000,
            prompt_len=12000,
            sampling_params=SamplingParams(prompt_logprobs=1),
        ),
        dict(
            num_computed=10000,
            prompt_len=12000,
            sampling_params=SamplingParams(repetition_penalty=1.1),
        ),
        dict(
            num_computed=10000,
            prompt_len=12000,
            sampling_params=SamplingParams(bad_words=["x"]),
        ),
        dict(
            num_computed=10000,
            prompt_len=12000,
            sampling_params=SamplingParams(thinking_token_budget=16),
        ),
    ],
)
def test_full_staging_fallbacks(staging, kwargs):
    assert staging.get_start(_new_req(**kwargs)) == 0


def test_no_sampling_params_stages_fully(staging):
    req = _new_req(10000, 12000)
    req.sampling_params = None
    assert staging.get_start(req) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_request_state_stages_only_the_suffix():
    from vllm.v1.worker.gpu.states import RequestState

    state = RequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=1000,
        device=torch.device("cuda"),
    )
    tokens = list(range(100, 150))
    state.add_request(
        req_id="r",
        prompt_len=len(tokens),
        all_token_ids=tokens,
        num_computed_tokens=45,
        max_tokens=8,
        staging_start=40,
    )
    state.apply_staged_writes()
    torch.accelerator.synchronize()
    row = state.all_token_ids.gpu[state.req_id_to_index["r"]].cpu()
    assert row[40:50].tolist() == tokens[40:]
    assert row[:40].eq(0).all()
