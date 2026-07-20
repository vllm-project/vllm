# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu import model_runner as mrv2


@pytest.mark.parametrize(
    ("scheduled_k", "supports_dynamic_draft_shapes", "proposal_width"),
    [(0, True, 0), (3, True, 3), (3, False, 7)],
)
def test_sample_tokens_honors_scheduled_speculative_k(
    monkeypatch, scheduled_k, supports_dynamic_draft_shapes, proposal_width
):
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    input_batch: Any = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-0", "req-1"],
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 1, 2]),
    )
    hidden_states = torch.zeros(2, 4)
    runner.execute_model_state = mrv2.ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=hidden_states,
        aux_hidden_states=None,
        finished_req_ids=set(),
        num_spec_tokens_to_schedule=scheduled_k,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.main_stream = None
    runner.output_copy_stream = None
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=Mock(return_value={})
    )
    runner.sample = Mock(
        return_value=(
            SimpleNamespace(sampled_token_ids=torch.zeros(2, 1, dtype=torch.long)),
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
    )
    runner.postprocess_sampled = Mock()
    runner.model_state = SimpleNamespace(gather_mm_embeddings=Mock())
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(2)),
            seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.long)),
        )
    )
    proposed_tokens = torch.arange(2 * proposal_width).reshape(2, proposal_width)
    runner.speculator = SimpleNamespace(
        supports_mm_inputs=False,
        supports_dynamic_draft_shapes=supports_dynamic_draft_shapes,
        propose=Mock(return_value=proposed_tokens),
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(2, 1, dtype=torch.long)),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        prompt_len=SimpleNamespace(np=torch.zeros(2, dtype=torch.int32).numpy()),
        last_sampled_tokens=torch.zeros(2, dtype=torch.long),
        next_prefill_tokens=torch.zeros(2, dtype=torch.long),
        draft_tokens=torch.full((2, 7), -1, dtype=torch.long),
    )
    runner.num_speculative_steps = 7
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())

    monkeypatch.setattr(
        mrv2.pcp,
        "maybe_restore_pcp_for_sampling",
        lambda _manager, states, batch: (states, batch),
    )
    monkeypatch.setattr(mrv2, "AsyncOutput", lambda **_: object())

    mrv2.GPUModelRunner.sample_tokens(runner, None)

    if scheduled_k == 0:
        runner.speculator.propose.assert_not_called()
    elif supports_dynamic_draft_shapes:
        assert (
            runner.speculator.propose.call_args.kwargs["num_speculative_tokens"]
            == scheduled_k
        )
    else:
        assert (
            "num_speculative_tokens" not in runner.speculator.propose.call_args.kwargs
        )
    published_tokens = runner.draft_tokens_handler.set_draft_tokens.call_args.args[1]
    assert published_tokens.shape == (2, scheduled_k)
    if scheduled_k > 0:
        torch.testing.assert_close(published_tokens, proposed_tokens[:, :scheduled_k])
