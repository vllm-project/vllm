# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated-position hidden states through the real LLM entry point."""

import os
from typing import TYPE_CHECKING

import pytest
import torch

from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class HiddenCaptureReferenceExtension:
    model_runner: "GPUModelRunner"

    @torch.no_grad()
    def check_reference_logits(self, states):
        runner = self.model_runner
        states = torch.tensor(states, device="cuda", dtype=runner.dtype)
        model = runner.model
        logits = model.compute_logits(states)
        connector = runner.aux_output_connector
        assert connector is not None and connector._hidden is not None
        metadata = connector._hidden.metadata
        rebuilt = torch.nn.functional.linear(states, model.lm_head.weight)
        rebuilt = rebuilt[:, : metadata["vocab_size"]]
        if metadata["soft_cap"] is not None:
            cap = metadata["soft_cap"]
            rebuilt = cap * torch.tanh(rebuilt / cap)
        rebuilt *= metadata["logit_scale"]
        torch.testing.assert_close(rebuilt, logits, atol=0, rtol=0)
        return True


def capture_params(start=0, end=None, **kwargs):
    options = dict(
        temperature=0,
        max_tokens=16,
        min_tokens=16,
        ignore_eos=True,
        aux_outputs=("hidden_states",),
        hidden_states_start=start,
        hidden_states_end=end,
    )
    options.update(kwargs)
    return SamplingParams(**options)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_hidden_windows_match_full_capture_and_raw_logits(
    vllm_runner, monkeypatch, enforce_eager
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with vllm_runner(
        os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_model_len=192,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_return_hidden_states=True,
        worker_extension_cls=f"{__name__}.HiddenCaptureReferenceExtension",
    ) as runner:
        llm = runner.llm
        prompt = {"prompt_token_ids": list(range(100, 228))}
        full = llm.generate(prompt, capture_params(), use_tqdm=False)[0].outputs[0]
        assert full.hidden_states_start == 127
        assert full.hidden_states.shape == (
            16,
            llm.llm_engine.model_config.get_hidden_size(),
        )
        plain = llm.generate(
            prompt,
            SamplingParams(
                temperature=0,
                max_tokens=16,
                min_tokens=16,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0].outputs[0]
        assert plain.hidden_states is None
        assert plain.token_ids == full.token_ids
        mixed = llm.generate(
            [prompt, prompt],
            [
                capture_params(131, 135),
                SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            ],
            use_tqdm=False,
        )
        window = mixed[0].outputs[0]
        assert window.token_ids == full.token_ids
        torch.testing.assert_close(
            window.hidden_states, full.hidden_states[4:8], rtol=0, atol=0
        )
        assert mixed[1].outputs[0].hidden_states is None
        empty = llm.generate(prompt, capture_params(150, 155), use_tqdm=False)[
            0
        ].outputs[0]
        assert empty.hidden_states.shape == (0, full.hidden_states.shape[1])
        assert empty.hidden_states.dtype == full.hidden_states.dtype
        paired = llm.generate(prompt, capture_params(n=2), use_tqdm=False)[0].outputs
        assert len(paired) == 2
        for output in paired:
            torch.testing.assert_close(
                output.hidden_states, full.hidden_states, atol=0, rtol=0
            )
        assert all(
            llm.collective_rpc(
                "check_reference_logits", args=(full.hidden_states.tolist(),)
            )
        )
        tokenizer = llm.get_tokenizer()
        stop = tokenizer.decode(full.token_ids[4:6])
        assert stop
        stopped = llm.generate(
            prompt, capture_params(129, 133, stop=stop, min_tokens=0), use_tqdm=False
        )[0].outputs[0]
        assert stopped.finish_reason == "stop"
        count = max(0, min(133, 127 + len(stopped.token_ids)) - 129)
        assert len(stopped.hidden_states) == count
        torch.testing.assert_close(
            stopped.hidden_states, full.hidden_states[2 : 2 + count], atol=0, rtol=0
        )


def test_hidden_rows_survive_kv_pressure_preemption(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
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
        enable_return_hidden_states=True,
        disable_log_stats=False,
    ) as runner:
        prompt = {"prompt_token_ids": [100, 200, 300, 400]}
        params = SamplingParams(
            temperature=0,
            max_tokens=128,
            min_tokens=128,
            ignore_eos=True,
            aux_outputs=("hidden_states",),
        )
        reference = runner.llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
        outputs = runner.llm.generate([prompt, prompt], params, use_tqdm=False)
        assert (
            sum(
                m.value
                for m in runner.llm.get_metrics()
                if m.name == "vllm:num_preemptions"
            )
            > 0
        )
        for output in outputs:
            assert output.outputs[0].token_ids == reference.token_ids
            torch.testing.assert_close(
                output.outputs[0].hidden_states, reference.hidden_states, atol=0, rtol=0
            )


def test_speculative_hidden_windows_follow_accepted_rows(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with vllm_runner(
        os.getenv("VLLM_TEST_SPEC_TARGET", "Qwen/Qwen3-4B"),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=96,
        max_num_seqs=2,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_return_hidden_states=True,
        per_request_spec_decode_metrics="summary",
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 7,
            "model": os.getenv(
                "VLLM_TEST_SPEC_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7"
            ),
        },
    ) as runner:
        prompt = {"prompt_token_ids": [100, 200, 300, 400]}
        full = runner.llm.generate(prompt, capture_params(), use_tqdm=False)[0].outputs[
            0
        ]
        assert full.spec_decode_metrics.to_dict()["num_accepted_draft_tokens"] > 0
        window = runner.llm.generate(prompt, capture_params(7, 11), use_tqdm=False)[
            0
        ].outputs[0]
        assert window.token_ids == full.token_ids
        torch.testing.assert_close(
            window.hidden_states, full.hidden_states[4:8], atol=0, rtol=0
        )
