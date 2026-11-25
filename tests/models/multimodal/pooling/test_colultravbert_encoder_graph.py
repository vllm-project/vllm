# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image

from vllm.platforms import current_platform

from .test_colultravbert import (
    DTYPE,
    EMBEDDING_DIM,
    IMAGE_SEQ_LEN,
    MODEL_NAME,
    PAGE_PROMPT,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA graphs"
)
REVISION = "5a7209cf12bfdd9bd061884ea3dc792d693c6c30"


def _graph_counts(worker):
    manager = worker.model_runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    stats = manager.get_cumulative_stats()
    return stats["graph_hits"], stats["graph_misses"]


def test_colultravbert_encoder_graph_matches_eager(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    pages = [Image.new("RGB", (384, 64), "red"), Image.new("RGB", (720, 720), "blue")]
    batches = ((0,), (1,), (0, 1))
    reference = []
    for tile_budget in (None, 7, 8):
        use_graph = tile_budget is not None
        with vllm_runner(
            MODEL_NAME,
            revision=REVISION,
            runner="pooling",
            dtype=DTYPE,
            max_model_len=2048,
            max_num_seqs=2,
            max_num_batched_tokens=4096,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "NONE",
                "cudagraph_mm_encoder": use_graph,
                "encoder_cudagraph_token_budgets": (
                    [tile_budget * IMAGE_SEQ_LEN] if tile_budget is not None else []
                ),
                "encoder_cudagraph_max_vision_items_per_batch": 2,
            },
        ) as model:
            image_token = model.llm.get_tokenizer().convert_tokens_to_ids("<image>")
            if use_graph:
                (before,) = model.llm.collective_rpc(_graph_counts)
            for batch_idx, batch in enumerate(batches):
                prompts = model.get_inputs(
                    [PAGE_PROMPT] * len(batch), images=[pages[i] for i in batch]
                )
                for i, prompt in zip(batch, prompts, strict=True):
                    prompt["multi_modal_uuids"] = {"image": f"{batch_idx}-{i}"}
                outputs = model.llm.encode(prompts, pooling_task="token_embed")
                if use_graph:
                    (after,) = model.llm.collective_rpc(_graph_counts)
                    hits, misses = after[0] - before[0], after[1] - before[1]
                    if tile_budget == 8:
                        # Even the seven-tile page must fall back at this shape.
                        assert (hits, misses) == (0, len(batch))
                    elif len(batch) == 1:
                        assert (hits, misses) == ((1, 0) if batch == (0,) else (0, 1))
                    else:
                        # The engine can deliver these requests in separate steps.
                        assert hits + misses == len(batch) and misses >= 1
                    before = after
                for i, out in zip(batch, outputs, strict=True):
                    ids = out.prompt_token_ids
                    emb = torch.as_tensor(out.outputs.data).float().cpu()
                    assert ids.count(image_token) == (7, 26)[i] * IMAGE_SEQ_LEN
                    assert emb.shape == (len(ids), EMBEDDING_DIM)
                    assert torch.isfinite(emb).all()
                    if not use_graph and len(batch) == 1:
                        reference.append((ids, emb.clone()))
                    else:
                        expected_ids, expected = reference[i]
                        assert ids == expected_ids
                        torch.testing.assert_close(
                            emb, expected, atol=0.016, rtol=0.016
                        )


def test_colultravbert_graph_with_disabled_image_tower(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        revision=REVISION,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=256,
        limit_mm_per_prompt={"image": 0},
        enable_mm_embeds=True,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": True,
        },
    ) as model:
        (embedding,) = model.token_embed(["a text query"])
        vectors = torch.as_tensor(embedding)
        assert vectors.shape[1] == EMBEDDING_DIM
        assert torch.isfinite(vectors).all()
