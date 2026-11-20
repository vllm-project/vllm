# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise sidecar loading and checkpoint top-k overrides through real LLMs."""

import pytest
import torch
from transformers import AutoConfig

from vllm import SamplingParams

from ..utils import compute_acceptance_rate

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def _read_router(worker):
    head = worker.model_runner.speculator.model.masked_embedding
    return {
        "centroids": head.centroids.weight.detach().cpu(),
        "ordering": head.token_ordering.cpu(),
        "top_k": head.centroid_intermediate_top_k,
    }


def test_nemotron_sidecar_router(tmp_path, monkeypatch, vllm_runner):
    """Load both router tensors and generate with nonzero MTP acceptance."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = AutoConfig.from_pretrained(MODEL)
    centroids = torch.ones(2, config.hidden_size, dtype=torch.bfloat16)
    ordering = torch.arange(config.vocab_size - 1, -1, -1)
    router = tmp_path / "router.pt"
    torch.save(
        {
            "masked_embedding.centroids.weight": centroids,
            "masked_embedding.token_ordering": ordering,
            "metadata": {"step": 0},
        },
        router,
    )
    prompts = ["The capital of France is", "Count from one to ten:"]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=2,
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=None,
        speculative_config={
            "method": "mtp",
            "hierarchical_lm_head_router_path": str(router),
            "hierarchical_lm_head_top_k": 2,
            "num_speculative_tokens": 2,
            "use_local_argmax_reduction": True,
        },
    ) as runner:
        actual = runner.llm.collective_rpc(_read_router)
        assert len(actual) == 1
        assert actual[0]["top_k"] == 2
        torch.testing.assert_close(actual[0]["centroids"], centroids)
        torch.testing.assert_close(actual[0]["ordering"], ordering)
        outputs = runner.llm.generate(prompts, sampling)
        assert all(len(output.outputs[0].token_ids) == 32 for output in outputs)
        assert compute_acceptance_rate(runner.llm.get_metrics()) > 0


@pytest.mark.parametrize("top_k", [None, 16])
def test_gemma4_checkpoint_top_k_override(top_k, monkeypatch, vllm_runner):
    """Keep Gemma4's checkpoint default unless the user overrides top-k."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    speculative_config = {
        "method": "mtp",
        "model": "google/gemma-4-E4B-it-assistant",
        "num_speculative_tokens": 2,
        "use_local_argmax_reduction": True,
    }
    if top_k is not None:
        speculative_config["hierarchical_lm_head_top_k"] = top_k
    with vllm_runner(
        "google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=2,
        enforce_eager=True,
        speculative_config=speculative_config,
    ) as runner:
        actual = runner.llm.collective_rpc(_read_router)
        assert actual[0]["top_k"] == (32 if top_k is None else top_k)
        assert actual[0]["centroids"].shape[0] == 2048
        outputs = runner.llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        )
        assert len(outputs[0].outputs[0].token_ids) == 16
