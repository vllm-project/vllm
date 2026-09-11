# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    _get_profile_num_reqs,
)
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


def test_default_dummy_run_request_count_is_unchanged():
    speculator = _DefaultSpeculator()
    assert speculator.get_num_reqs_for_dummy_run(128) == 128


def test_dflash_dummy_run_request_count_is_capped():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 8
    speculator.max_num_tokens = 512

    assert speculator.get_num_reqs_for_dummy_run(128) == 64


def test_profile_num_reqs_caps_to_full_query_blocks():
    assert _get_profile_num_reqs(
        num_reqs=512,
        max_num_tokens=2048,
        num_query_per_req=5,
    ) == 409


def test_profile_num_reqs_keeps_already_fitting_batch():
    assert _get_profile_num_reqs(
        num_reqs=128,
        max_num_tokens=2048,
        num_query_per_req=5,
    ) == 128


def test_profile_num_reqs_rejects_zero_query_capacity():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _get_profile_num_reqs(
            num_reqs=1,
            max_num_tokens=4,
            num_query_per_req=5,
        )


def test_profile_propose_caps_draft_query_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5
    speculator.max_num_tokens = 2048
    speculator.max_model_len = 4096
    speculator.hidden_states = torch.zeros(2048, 4)
    speculator.context_positions = torch.zeros(2048, dtype=torch.int64)
    speculator.draft_tokens = torch.zeros(512, 5, dtype=torch.int64)

    calls = []

    class ModelStub:
        def precompute_and_store_context_kv(self, hidden_states, context_positions):
            calls.append(
                ("precompute", hidden_states.shape[0], context_positions.shape[0])
            )

    def prepare_eplb(num_query_tokens):
        calls.append(("eplb", num_query_tokens))

    def generate_draft(
        num_reqs,
        num_query_tokens,
        *,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    ):
        calls.append(
            (
                "generate",
                num_reqs,
                num_query_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        )

    speculator.model = ModelStub()
    speculator._prepare_eplb_forward = prepare_eplb
    speculator._generate_draft = generate_draft
    input_batch = SimpleNamespace(
        num_reqs=512,
        num_tokens=2048,
        seq_lens_cpu_upper_bound=torch.full((512,), 4, dtype=torch.int32),
    )

    output = speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.ones(2048, 4),
        aux_hidden_states=None,
        num_sampled=torch.ones(512, dtype=torch.int32),
        num_rejected=torch.zeros(512, dtype=torch.int32),
        last_sampled=torch.zeros(512, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(512, dtype=torch.int64),
        temperature=torch.zeros(512),
        seeds=torch.zeros(512, dtype=torch.int64),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )

    assert output.shape == (512, 5)
    assert speculator.draft_max_seq_len == 9
    assert calls == [
        ("precompute", 2048, 2048),
        ("eplb", 2045),
        ("generate", 409, 2045, None, None, None, CUDAGraphMode.NONE),
    ]


def test_real_propose_rejects_oversized_query_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5
    speculator.max_num_tokens = 2048

    input_batch = SimpleNamespace(num_reqs=512, num_tokens=2048)

    with pytest.raises(ValueError, match="exceeds max_num_batched_tokens"):
        speculator.propose(
            input_batch=input_batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.ones(2048, 4),
            aux_hidden_states=None,
            num_sampled=torch.ones(512, dtype=torch.int32),
            num_rejected=torch.zeros(512, dtype=torch.int32),
            last_sampled=torch.zeros(512, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(512, dtype=torch.int64),
            temperature=torch.zeros(512),
            seeds=torch.zeros(512, dtype=torch.int64),
        )
