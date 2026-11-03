# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise capture through LLM.generate and verify the hidden/token alignment."""

import asyncio
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


def hidden_result(output):
    aux = output.aux_outputs
    return None if aux is None else aux.hidden_state_capture


def skip_reason(output):
    aux = output.aux_outputs
    return None if aux is None else aux.hidden_capture_skip_reason


class HiddenCaptureReferenceExtension:
    model_runner: "GPUModelRunner"

    def capture_contract(self):
        connector = self.model_runner.aux_output_connector
        assert connector is not None and connector._hidden is not None
        return connector._hidden.logits_contract

    def capture_pending_requests(self):
        connector = self.model_runner.aux_output_connector
        assert connector is not None and connector._hidden is not None
        hidden = connector._hidden
        return {
            "plans": len(hidden.plans),
            "settled": len(hidden.settled_ends),
            "carry_empty": hidden.carry.empty(),
        }

    @torch.no_grad()
    def capture_reference_logits(self, states):
        states = torch.tensor(states, device="cuda", dtype=torch.bfloat16)
        model = self.model_runner.model
        logits = model.compute_logits(states).float()
        contract = self.capture_contract()
        assert contract is not None
        rebuilt = torch.nn.functional.linear(states, model.lm_head.weight)
        if contract["soft_cap"] is not None:
            cap = contract["soft_cap"]
            rebuilt = cap * torch.tanh(rebuilt / cap)
        rebuilt = rebuilt[:, : contract["vocab_size"]] * contract["logit_scale"]
        torch.testing.assert_close(rebuilt.float(), logits, atol=0, rtol=0)
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
        settled_ends={"r": 8},
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
            result = hidden_result(output)
            assert skip_reason(output) is None
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
        served_model_name="hidden-test-teacher",
    ) as runner:
        llm = runner.llm
        contract = llm.collective_rpc("capture_contract")[0]
        assert contract["served_model_name"] == "hidden-test-teacher"
        assert contract["lm_head_weight"] == (
            "model.embed_tokens.weight"
            if contract["tie_word_embeddings"]
            else "lm_head.weight"
        )
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
        assert hidden_result(mixed[0]) is None
        torch.testing.assert_close(
            hidden_result(mixed[1]).hidden_states,
            hidden_result(full[1]).hidden_states[4:8],
            atol=0,
            rtol=0,
        )
        for _ in range(2):
            window = generate(4, 8)
            for disabled, before, after in zip(off, full, window):
                assert after.outputs[0].token_ids == disabled.outputs[0].token_ids
                result = hidden_result(after)
                np.testing.assert_array_equal(result.hidden_positions, [7, 8, 9, 10])
                torch.testing.assert_close(
                    result.hidden_states,
                    hidden_result(before).hidden_states[4:8],
                    atol=0,
                    rtol=0,
                )
        # Teacher scoring: prompt-segment rows of prompt + response match the
        # rows captured while generating that response.
        scoring_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            extra_args={
                "hidden_state_capture": {
                    "start": 3,
                    "end": 11,
                    "coordinate": "absolute",
                    "min_rows": 8,
                    "collection_id": "teacher-scoring",
                }
            },
        )
        scored = llm.generate(
            [
                {
                    "prompt_token_ids": prompt["prompt_token_ids"]
                    + list(out.outputs[0].token_ids[:8])
                }
                for prompt, out in zip(prompts, off)
            ],
            scoring_params,
            use_tqdm=False,
        )
        for teacher, original in zip(scored, full):
            result = hidden_result(teacher)
            np.testing.assert_array_equal(result.hidden_positions, np.arange(3, 11))
            assert [s.kind for s in result.segments] == ["prompt"]
            assert result.logits_contract["vocab_size"] > 0
            torch.testing.assert_close(
                result.hidden_states,
                hidden_result(original).hidden_states[:8],
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
                    hidden_result(reference).hidden_states[0],
                    hidden_result(original).hidden_states[response_index],
                    atol=0,
                    rtol=0,
                )
        states = torch.stack([hidden_result(o).hidden_states[5] for o in full])

        values, ids = llm.collective_rpc(
            "capture_reference_logits", args=(states.tolist(),)
        )[0]
        values = torch.tensor(values)
        ids = torch.tensor(ids)
        for index, output in enumerate(full):
            captured = hidden_result(output)
            torch.testing.assert_close(
                values[index], captured.logprobs[5], atol=2e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                ids[index], captured.logprob_token_ids[5], atol=0, rtol=0
            )
        never = generate(32, 40)
        assert all(
            hidden_result(o) is None and skip_reason(o) == "ended_before_window"
            for o in never
        )


@pytest.mark.parametrize("kv_pressure", [False, True])
def test_prompt_capture_spans_chunked_prefill(vllm_runner, monkeypatch, kv_pressure):
    """Rows from prefill chunks that sample nothing are delivered at the end."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    prompts = [
        {"prompt_token_ids": list(range(100, 148))},
        {"prompt_token_ids": list(range(200, 248))},
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=48 if kv_pressure else 1,
        min_tokens=48 if kv_pressure else 1,
        ignore_eos=True,
        extra_args={
            "hidden_state_capture": {
                "start": 2,
                "end": 47,
                "coordinate": "absolute",
                "min_rows": 45,
                "collection_id": "chunked",
            }
        },
    )
    with vllm_runner(
        model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=96,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enable_prefix_caching=False,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
    ) as runner:
        reference = [
            hidden_result(output).hidden_states.clone()
            for output in runner.llm.generate(prompts, params, use_tqdm=False)
        ]
    with vllm_runner(
        model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=96 if kv_pressure else 64,
        max_num_seqs=2,
        num_gpu_blocks_override=7 if kv_pressure else None,
        scheduler_reserve_full_isl=not kv_pressure,
        disable_log_stats=False,
        max_num_batched_tokens=2 if kv_pressure else 16,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
    ) as runner:
        outputs = runner.llm.generate(prompts, params, use_tqdm=False)
        for output, expected in zip(outputs, reference):
            result = hidden_result(output)
            assert skip_reason(output) is None
            np.testing.assert_array_equal(result.hidden_positions, np.arange(2, 47))
            torch.testing.assert_close(result.hidden_states, expected, atol=0, rtol=0)

        if kv_pressure:
            assert (
                sum(
                    m.value
                    for m in runner.llm.get_metrics()
                    if m.name == "vllm:num_preemptions"
                )
                > 0
            )


def test_prompt_capture_reuses_only_prefix_before_window(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    prompt = {"prompt_token_ids": list(range(100, 228))}
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        extra_args={
            "hidden_state_capture": {
                "start": 64,
                "end": 127,
                "coordinate": "absolute",
                "collection_id": "cached",
            }
        },
    )
    with vllm_runner(
        os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=144,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
    ) as runner:
        first = runner.llm.generate([prompt], params, use_tqdm=False)[0]
        cached = runner.llm.generate([prompt], params, use_tqdm=False)[0]
        assert first.num_cached_tokens == 0
        assert 0 < cached.num_cached_tokens <= 64
        np.testing.assert_array_equal(
            hidden_result(cached).hidden_positions, np.arange(64, 127)
        )
        torch.testing.assert_close(
            hidden_result(cached).hidden_states,
            hidden_result(first).hidden_states,
            atol=0,
            rtol=0,
        )


def test_speculative_aux_windows_match_full_capture(vllm_runner, monkeypatch):
    """DSpark auxiliary/final rows follow accepted tokens, including top-k."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    model = os.getenv("VLLM_TEST_SPEC_TARGET", "Qwen/Qwen3-4B")
    draft = os.getenv("VLLM_TEST_SPEC_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
    layers = (2, 10, 18, 26, 34)
    with vllm_runner(
        model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=96,
        max_num_seqs=2,
        enable_prefix_caching=False,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
        speculative_config={
            "method": "dspark",
            "model": draft,
            "num_speculative_tokens": 7,
        },
    ) as runner:
        prompts = [
            {"prompt_token_ids": [100, 200, 300, 400]},
            {"prompt_token_ids": [500, 600, 700, 800]},
        ]

        def generate(start, end):
            plan = HiddenStateCapturePlan.from_window(
                "",
                4,
                start,
                end,
                "spec",
                min_rows=end - start,
                aux_layer_ids=layers,
                hidden_layout="aux_final",
                logprob_topk=4,
            )
            return runner.llm.generate(
                prompts,
                SamplingParams(
                    temperature=0,
                    max_tokens=32,
                    min_tokens=32,
                    ignore_eos=True,
                    extra_args={"hidden_state_capture": asdict(plan)},
                ),
                use_tqdm=False,
            )

        full = generate(0, 32)
        window = generate(8, 16)
        for before, after in zip(full, window):
            assert before.outputs[0].token_ids == after.outputs[0].token_ids
            result = hidden_result(after)
            assert skip_reason(after) is None
            assert result.layer_ids == layers
            assert result.includes_final_layer
            np.testing.assert_array_equal(result.hidden_positions, np.arange(11, 19))
            for name, tensor in result.tensors.items():
                torch.testing.assert_close(
                    tensor, hidden_result(before).tensors[name][8:16], atol=0, rtol=0
                )


@pytest.mark.asyncio
async def test_aborted_capture_releases_worker_state(monkeypatch):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=512,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            async_scheduling=False,
            aux_output_config={"enable_hidden_states": True},
            worker_extension_cls=f"{__name__}.HiddenCaptureReferenceExtension",
            served_model_name="hidden-test-teacher",
        )
    )
    try:
        stream = engine.generate(
            {"prompt_token_ids": list(range(100, 356))},
            SamplingParams(
                temperature=0,
                max_tokens=128,
                min_tokens=128,
                ignore_eos=True,
                extra_args={
                    "hidden_state_capture": {
                        "start": 0,
                        "end": 255,
                        "coordinate": "absolute",
                        "collection_id": "abort",
                    }
                },
            ),
            "abort-capture",
        )
        first_output = asyncio.create_task(anext(stream))
        try:
            async with asyncio.timeout(10):
                while True:
                    states = await engine.collective_rpc("capture_pending_requests")
                    if not states[0]["carry_empty"]:
                        assert not first_output.done(), "prefill already emitted output"
                        break
                    assert not first_output.done(), "prefill carry was not observed"
            await engine.abort("abort-capture")
            first_output.cancel()
            await asyncio.gather(first_output, return_exceptions=True)
        finally:
            if not first_output.done():
                first_output.cancel()
                await asyncio.gather(first_output, return_exceptions=True)
        await stream.aclose()
        # A normal request flushes terminal metadata to the worker and proves
        # aborting capture does not disrupt other requests.
        async for output in engine.generate(
            {"prompt_token_ids": [100, 200, 300]},
            SamplingParams(temperature=0, max_tokens=1),
            "after-abort",
        ):
            if output.finished:
                assert len(output.outputs[0].token_ids) == 1
        assert await engine.collective_rpc("capture_pending_requests") == [
            {
                "plans": 0,
                "settled": 0,
                "carry_empty": True,
            }
        ]
    finally:
        engine.shutdown()
