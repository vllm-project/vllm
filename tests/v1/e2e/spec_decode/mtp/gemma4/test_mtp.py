# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams

from ...utils import get_spec_decode_metric_value, get_test_prompts
from .._correctness import check_mtp_correctness

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class DraftRowRecorder:
    """Worker extension that records how many rows each draft forward runs on."""

    model_runner: "GPUModelRunner"

    def start_draft_row_recording(self) -> None:
        from vllm.model_executor.models.gemma4_mtp import Gemma4MultiTokenPredictor
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

        self.draft_rows: list[int] = []

        def record_rows(_module, args):
            # args: (input_ids, positions, hidden_states, ...)
            self.draft_rows.append(args[1].shape[0])

        speculator = self.model_runner.speculator
        assert isinstance(speculator, Gemma4Speculator)
        # The torch.compile wrapper on Gemma4MTP calls forward() directly,
        # bypassing module hooks, so observe its inner predictor instead.
        predictor = speculator.model.model
        assert isinstance(predictor, Gemma4MultiTokenPredictor)
        self.draft_row_hook = predictor.register_forward_pre_hook(record_rows)

    def read_draft_rows(self) -> list[int]:
        self.draft_row_hook.remove()
        return self.draft_rows


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
def test_draft_prefill_runs_on_last_tokens_only(
    sampling_config: SamplingParams, vllm_runner, monkeypatch: pytest.MonkeyPatch
):
    """The Gemma4 drafter reads the target's KV and writes none, so draft step 0
    of a step that prefills prompts runs on one row per request, not on every
    prompt token, with no implicit GPU sync, and its drafts are still accepted."""
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompts = get_test_prompts(mm_enabled=False, num_prompts=8)
    with vllm_runner(
        "google/gemma-4-E4B-it",
        max_model_len=2048,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        worker_extension_cls=f"{__name__}.DraftRowRecorder",
        speculative_config={
            "method": "mtp",
            "model": "google/gemma-4-E4B-it-assistant",
            "num_speculative_tokens": 2,
        },
        disable_log_stats=False,
    ) as runner:
        runner.llm.collective_rpc("start_draft_row_recording")
        runner.llm.chat(prompts, sampling_config)
        (draft_rows,) = runner.llm.collective_rpc("read_draft_rows")
        metrics = runner.llm.get_metrics()

    assert draft_rows and max(draft_rows) <= len(prompts), draft_rows[:8]
    accepted = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    assert accepted > 0
