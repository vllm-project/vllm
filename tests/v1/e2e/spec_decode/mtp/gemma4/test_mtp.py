# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams, TokensPrompt

from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_gemma4_skips_unused_draft_forward(monkeypatch, vllm_runner):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    prompt = TokensPrompt(prompt_token_ids=[42] * 257)
    with vllm_runner(
        "google/gemma-4-E4B-it",
        speculative_config={
            "method": "mtp",
            "model": "google/gemma-4-E4B-it-assistant",
            "num_speculative_tokens": 2,
        },
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        language_model_only=True,
        enforce_eager=True,
    ) as runner:
        worker = runner.llm.llm_engine.model_executor.driver_worker.worker
        predictor = worker.model_runner.speculator.model.model
        draft_rows: list[int] = []

        def record_rows(module, args):
            draft_rows.append(args[0].shape[0])

        with predictor.register_forward_pre_hook(record_rows):
            # No draft can be consumed after any chunk of a one-output request.
            runner.llm.generate(prompt, SamplingParams(temperature=0, max_tokens=1))
            assert draft_rows == []

            # A request that continues into decode still uses the same predictor.
            runner.llm.generate(
                prompt, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            )
            assert draft_rows
