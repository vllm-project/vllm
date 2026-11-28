# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise hierarchical router loading through normal draft checkpoints."""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm import SamplingParams
from vllm.transformers_utils.repo_utils import hf_api

from ..utils import compute_acceptance_rate

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def _read_router(worker):
    head = worker.model_runner.speculator.model.masked_embedding
    return {
        "centroids": head.centroids.weight.detach().cpu(),
        "ordering": head.token_ordering.cpu(),
        "top_k": head.centroid_intermediate_top_k,
    }


def test_nemotron_router_checkpoint(tmp_path, monkeypatch, vllm_runner):
    """Load router parameters and its ordering buffer with the MTP weights."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    api = hf_api()
    config = json.loads(Path(api.hf_hub_download(MODEL, "config.json")).read_text())
    index = json.loads(
        Path(api.hf_hub_download(MODEL, "model.safetensors.index.json")).read_text()
    )["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    for shard in {value for name, value in index.items() if name.startswith("mtp.")}:
        with safe_open(api.hf_hub_download(MODEL, shard), framework="pt") as tensors:
            names = tensors.keys()
            weights.update(
                (name, tensors.get_tensor(name))
                for name in names
                if name.startswith("mtp.")
            )
    centroids = torch.ones(2, config["hidden_size"], dtype=torch.bfloat16)
    ordering = torch.arange(config["vocab_size"] - 1, -1, -1)
    weights["masked_embedding.centroids.weight"] = centroids
    weights["masked_embedding.token_ordering"] = ordering
    save_file(weights, tmp_path / "model.safetensors")
    del weights
    config.update(
        use_ordered_embeddings=True, num_centroids=2, centroid_intermediate_top_k=2
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
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
            "model": str(tmp_path),
            "num_speculative_tokens": 2,
            "use_local_argmax_reduction": True,
        },
    ) as runner:
        actual = runner.llm.collective_rpc(_read_router)
        assert len(actual) == 1
        assert actual[0]["top_k"] == 2
        torch.testing.assert_close(actual[0]["centroids"], centroids)
        torch.testing.assert_close(actual[0]["ordering"], ordering)
        outputs = runner.llm.generate(
            ["The capital of France is", "Count from one to ten:"], sampling
        )
        assert all(len(output.outputs[0].token_ids) == 32 for output in outputs)
        assert compute_acceptance_rate(runner.llm.get_metrics()) > 0


@pytest.mark.parametrize("top_k", [None, 16])
def test_gemma4_checkpoint_top_k(top_k, tmp_path, monkeypatch, vllm_runner):
    """Use the official assistant or the same weights with a different config."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    draft = "google/gemma-4-E4B-it-assistant"
    if top_k is not None:
        original = Path(
            hf_api().snapshot_download(
                draft,
                allow_patterns=[
                    "config.json",
                    "model*.safetensors",
                    "generation_config.json",
                ],
            )
        )
        for path in original.iterdir():
            if path.name != "config.json" and path.is_file():
                (tmp_path / path.name).symlink_to(path.absolute())
        config = json.loads((original / "config.json").read_text())
        config["centroid_intermediate_top_k"] = top_k
        (tmp_path / "config.json").write_text(json.dumps(config))
        draft = str(tmp_path)
    with vllm_runner(
        "google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=2,
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=None,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "model": draft,
            "num_speculative_tokens": 2,
            "use_local_argmax_reduction": True,
        },
    ) as runner:
        actual = runner.llm.collective_rpc(_read_router)
        assert actual[0]["top_k"] == (32 if top_k is None else top_k)
        assert actual[0]["centroids"].shape[0] == 2048
        outputs = runner.llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        )
        assert len(outputs[0].outputs[0].token_ids) == 16
        assert compute_acceptance_rate(runner.llm.get_metrics()) > 0
