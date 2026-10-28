# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise capture through LLM.generate and verify the hidden/token alignment."""

import os
from dataclasses import asdict

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.hidden_states import (
    HiddenStateCapturePlan,
    capture_scheduled_hidden_states,
)
from vllm.sampling_params import SamplingParams


def test_window_logprobs_do_not_depend_on_other_selected_rows():
    generator = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn((8, 151936), generator=generator, device="cuda") * 5
    hidden = torch.zeros((8, 2), device="cuda")
    indices = torch.arange(8, device="cuda")

    def capture(start, end):
        plan = HiddenStateCapturePlan.from_window(
            "r", 1, start, end, "test", logprob_topk=16
        )
        return capture_scheduled_hidden_states(
            {"r": plan},
            ["r"],
            {"r": 8},
            {"r": 0},
            hidden,
            logits=logits,
            logits_indices=indices,
        )["r"].logprobs

    full = capture(0, 8)
    for index in range(8):
        torch.testing.assert_close(
            capture(index, index + 1)[0], full[index], atol=0, rtol=0
        )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_hidden_windows_match_full_capture_and_raw_logits(
    vllm_runner, monkeypatch, enforce_eager
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    with vllm_runner(
        model,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_model_len=256,
        max_num_seqs=2,
        enable_prefix_caching=False,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
        logprobs_mode="raw_logprobs",
    ) as runner:
        llm = runner.llm
        prompts = [{"prompt_token_ids": [100, 200, 300, 400]}] * 2
        params = SamplingParams(
            temperature=0, max_tokens=16, min_tokens=16, ignore_eos=True, logprobs=16
        )
        off = llm.generate(prompts, params, use_tqdm=False)

        def generate(start, end):
            plan = HiddenStateCapturePlan.from_window(
                "", 4, start, end, "test", min_rows=end - start, logprob_topk=16
            )
            selected = params.clone()
            selected.extra_args = {"hidden_state_capture": asdict(plan)}
            return llm.generate(prompts, selected, use_tqdm=False)

        full = generate(0, 16)
        for _ in range(2):
            window = generate(4, 8)
            for disabled, before, after in zip(off, full, window):
                assert after.outputs[0].token_ids == disabled.outputs[0].token_ids
                result = after.hidden_state_capture
                np.testing.assert_array_equal(result.hidden_positions, [7, 8, 9, 10])
                torch.testing.assert_close(
                    result.hidden_states,
                    before.hidden_state_capture.hidden_states[4:8],
                    atol=0,
                    rtol=0,
                )
        states = torch.stack([o.hidden_state_capture.hidden_states[5] for o in full])

        def project_hidden(model):
            logits = model.compute_logits(states.cuda()).float()
            values, ids = logits.log_softmax(-1).topk(16, dim=-1)
            return values.cpu().tolist(), ids.cpu().tolist()

        values, ids = llm.apply_model(project_hidden)[0]
        values = torch.tensor(values)
        ids = torch.tensor(ids)
        for index, output in enumerate(full):
            captured = output.hidden_state_capture
            torch.testing.assert_close(
                values[index], captured.logprobs[5], atol=2e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                ids[index], captured.logprob_token_ids[5], atol=0, rtol=0
            )
        never = generate(32, 40)
        assert all(
            o.hidden_state_capture is None
            and o.hidden_capture_skip_reason == "ended_before_window"
            for o in never
        )
