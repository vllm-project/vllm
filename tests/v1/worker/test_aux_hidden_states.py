# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise capture through LLM.generate and verify the hidden/token alignment."""

import os
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.hidden_states import (
    HiddenStateCapturePlan,
    capture_scheduled_hidden_states,
)
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class HiddenCaptureReferenceExtension:
    model_runner: "GPUModelRunner"

    @torch.no_grad()
    def capture_reference_logits(self, states):
        states = torch.tensor(states, device="cuda", dtype=torch.bfloat16)
        logits = self.model_runner.model.compute_logits(states).float()
        values, ids = logits.log_softmax(-1).topk(16, dim=-1)
        return values.cpu().tolist(), ids.cpu().tolist()


def test_window_logprobs_do_not_depend_on_other_selected_rows():
    generator = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn((8, 151936), generator=generator, device="cuda") * 5
    hidden = torch.zeros((8, 2), device="cuda")

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
            cu_num_logits=np.array([0, 8]),
        )["r"].logprobs

    full = capture(0, 8)
    for index in range(8):
        torch.testing.assert_close(
            capture(index, index + 1)[0], full[index], atol=0, rtol=0
        )


def test_recomputed_history_is_excluded_before_logprob_capture():
    plan = HiddenStateCapturePlan.from_window("r", 1, 0, 16, "test", logprob_topk=2)
    hidden = torch.arange(18, device="cuda").reshape(9, 2)
    logits = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    chunk = capture_scheduled_hidden_states(
        {"r": plan},
        ["r"],
        {"r": 9},
        {"r": 0},
        hidden,
        logits=logits,
        cu_num_logits=np.array([0, 1]),
        accepted_ends={"r": 8},
    )["r"]
    np.testing.assert_array_equal(chunk.positions, [8])
    torch.testing.assert_close(chunk.hidden_states, hidden[8:9])
    torch.testing.assert_close(
        chunk.logprob_token_ids, torch.tensor([[2, 1]], device="cuda")
    )
    torch.testing.assert_close(chunk.logprobs, logits.log_softmax(-1)[:, [2, 1]])


def test_hidden_logprobs_survive_kv_pressure_preemption(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    plan = HiddenStateCapturePlan.from_window(
        "", 4, 0, 128, "preemption", min_rows=128, logprob_topk=2
    )
    with vllm_runner(
        os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=144,
        max_num_seqs=2,
        block_size=16,
        num_gpu_blocks_override=10,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=False,
        aux_output_config={"enable_hidden_states": True},
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [100, 200, 300, 400]}] * 2,
            SamplingParams(
                temperature=0,
                max_tokens=128,
                min_tokens=128,
                ignore_eos=True,
                extra_args={"hidden_state_capture": asdict(plan)},
            ),
            use_tqdm=False,
        )
        assert (
            sum(
                m.value
                for m in runner.llm.get_metrics()
                if m.name == "vllm:num_preemptions"
            )
            > 0
        )
        for output in outputs:
            result = output.hidden_state_capture
            assert output.hidden_capture_skip_reason is None
            np.testing.assert_array_equal(result.hidden_positions, np.arange(3, 131))
            assert result.logprobs.shape == (128, 2)
            assert torch.isfinite(result.logprobs).all()
            assert len(output.outputs[0].token_ids) == 128


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_hidden_windows_match_full_capture_and_raw_logits(
    vllm_runner, monkeypatch, enforce_eager
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
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
        worker_extension_cls=f"{__name__}.HiddenCaptureReferenceExtension",
    ) as runner:
        llm = runner.llm
        prompts = [
            {"prompt_token_ids": ids}
            for ids in ([100, 200, 300, 400], [500, 600, 700, 800])
        ]
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
        invalid_specs = [
            {"start": 1.5, "end": 4, "collection_id": "test"},
            {"start": 0, "end": 4, "collection_id": "test", "unknown": 1},
            {"start": 0, "end": 4, "collection_id": "test", "logprob_topk": 21},
            {"start": 0, "end": 257, "collection_id": "test"},
            {
                "start": 0,
                "end": 256,
                "collection_id": "test",
                "aux_layer_ids": list(range(1000)),
                "hidden_layout": "aux_final",
            },
        ]
        from vllm.exceptions import VLLMValidationError

        for spec in invalid_specs:
            invalid = params.clone()
            invalid.extra_args = {"hidden_state_capture": spec}
            with pytest.raises(VLLMValidationError):
                llm.generate(prompts, invalid, use_tqdm=False)
        mixed_params = params.clone()
        mixed_params.extra_args = {
            "hidden_state_capture": {"start": 4, "end": 8, "collection_id": "mixed"}
        }
        mixed = llm.generate(prompts, [params, mixed_params], use_tqdm=False)
        assert mixed[0].hidden_state_capture is None
        torch.testing.assert_close(
            mixed[1].hidden_state_capture.hidden_states,
            full[1].hidden_state_capture.hidden_states[4:8],
            atol=0,
            rtol=0,
        )
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
        # Recompute each accepted prefix as prefill, independently of decode slicing.
        for response_index in range(4, 8):
            prefixes = [
                {
                    "prompt_token_ids": prompt["prompt_token_ids"]
                    + list(out.outputs[0].token_ids[:response_index])
                }
                for prompt, out in zip(prompts, off)
            ]
            reference_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                extra_args={
                    "hidden_state_capture": {
                        "start": 0,
                        "end": 1,
                        "collection_id": "teacher-forcing",
                    }
                },
            )
            references = llm.generate(prefixes, reference_params, use_tqdm=False)
            for reference, original in zip(references, full):
                assert (
                    reference.outputs[0].token_ids[0]
                    == original.outputs[0].token_ids[response_index]
                )
                torch.testing.assert_close(
                    reference.hidden_state_capture.hidden_states[0],
                    original.hidden_state_capture.hidden_states[response_index],
                    atol=0,
                    rtol=0,
                )
        states = torch.stack([o.hidden_state_capture.hidden_states[5] for o in full])

        values, ids = llm.collective_rpc(
            "capture_reference_logits", args=(states.tolist(),)
        )[0]
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
