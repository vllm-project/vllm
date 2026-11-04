# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams

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


def _update_draft_one_tensor_per_call(worker) -> list[str]:
    """A direct draft weight update with the assistant's own checkpoint, one
    tensor per ``load_weights`` call (as the NCCL engine sends it); returns
    the draft tensors whose bytes differ from the cold start."""
    import torch

    from vllm.model_executor.model_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    model = worker.get_draft_model()
    config = worker.speculative_config.draft_model_config
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    # only the update's model.embed_tokens.weight can restore the tied lm_head
    model.lm_head.weight.data.zero_()
    loader = DefaultModelLoader(worker.vllm_config.load_config)
    start_reload(model)
    for name, weight in loader.get_all_weights(config, model):
        model.load_weights([(name, weight)])
    finish_reload(model, config, worker.device)
    after = dict(model.named_parameters())
    return [n for n, p in before.items() if not torch.equal(p, after[n])]


@single_gpu_only
def test_draft_weight_update(monkeypatch: pytest.MonkeyPatch, vllm_runner):
    """The draft shares the target's (backbone-dim) embedding; an update
    writes the assistant's draft-dim embed_tokens into its tied lm_head."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "google/gemma-4-E2B-it",
        max_model_len=2048,
        speculative_config={
            "method": "mtp",
            "model": "google/gemma-4-E2B-it-assistant",
            "num_speculative_tokens": 2,
        },
    ) as llm:
        assert llm.collective_rpc(_update_draft_one_tensor_per_call) == [[]]
