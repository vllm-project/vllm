# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@torch.inference_mode()
def _embed_dummy_video(worker) -> torch.Tensor:
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.utils import group_and_batch_mm_kwargs

    runner = worker.model_runner
    budget = MultiModalBudget(
        worker.vllm_config, MULTIMODAL_REGISTRY, enable_cache=False
    )
    items = budget.get_dummy_encoder_profile_inputs("video", 1)
    _, _, mm_kwargs = next(group_and_batch_mm_kwargs(items, device=runner.device))
    (embedding,) = runner.get_model().embed_multimodal(**mm_kwargs)
    return embedding.float().cpu()


@pytest.mark.core_model
def test_encoder_only_matches_full_model_for_pruned_video(vllm_runner, monkeypatch):
    """Pruned video outputs include the embeddings of timestamp and vision
    marker tokens, so an encoder-only instance must still produce them."""
    # collective_rpc sends a function, which needs the pickle fallback.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    embeddings = []
    for mm_encoder_only in (False, True):
        with vllm_runner(
            MODEL,
            mm_encoder_only=mm_encoder_only,
            video_pruning_rate=0.5,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 0, "video": 1},
        ) as model:
            (embedding,) = model.llm.collective_rpc(_embed_dummy_video)
            embeddings.append(embedding)

    full, encoder_only = embeddings
    torch.testing.assert_close(encoder_only, full)
