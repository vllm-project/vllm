# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
from pathlib import Path

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.transformers_utils.repo_utils import hf_api

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


@pytest.mark.parametrize("enforce_eager", [True, False])
@single_gpu_only
def test_prefill_with_only_assistant_config(
    vllm_runner, tmp_path: Path, enforce_eager: bool
):
    assistant = "google/gemma-4-E2B-it-assistant"
    shutil.copy(
        hf_api().hf_hub_download(assistant, "config.json"), tmp_path / "config.json"
    )

    with vllm_runner(
        "google/gemma-4-E2B-it",
        max_model_len=256,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        speculative_config={
            "method": "mtp",
            "model": str(tmp_path),
            "num_speculative_tokens": 2,
            "prefill_context_kv_only": True,
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path / "kv")},
        ),
    ) as producer:
        config = producer.llm.llm_engine.vllm_config
        assert config.num_speculative_tokens == 2
        assert config.runner_speculative_config is None
        assert config.num_scheduled_spec_tokens == config.num_lookahead_tokens == 0
        # The prefill worker runs the target alone: no drafter is loaded.
        assert producer.llm.collective_rpc("get_draft_model") == [None]
        (output,) = producer.llm.generate(
            ["Count from one to ten: one, two,"],
            SamplingParams(temperature=0, min_tokens=3, max_tokens=3),
        )
        assert len(output.outputs[0].token_ids) == 3
