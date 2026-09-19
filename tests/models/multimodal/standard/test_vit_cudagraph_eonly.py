# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-only ViT cudagraph capture, replay and output-ownership checks.

This file is deliberately outside ``vit_cudagraph/``. It monkeypatches
``VLLM_USE_V2_MODEL_RUNNER`` and ``VLLM_ALLOW_INSECURE_SERIALIZATION`` and builds an
encoder-only runner, and any ViT case that runs after it in the same pytest process
segfaults in CPU tensor allocation. Keeping it out of the hash-sharded directory
gives it its own process via its own Buildkite job.
"""

from functools import partial

import pytest

from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from .vit_cudagraph._vit_cudagraph import MODEL_CONFIGS, params_with_marks


def _check_eonly_encoder_outputs(worker, batches):
    """Compare real E-only replay with eager, retaining outputs across replay."""
    import torch

    from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

    runner = worker.model_runner
    assert isinstance(runner, MMEncoderModelRunner)
    manager = runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_error = 0.0
    stats = []
    with torch.inference_mode():
        for batch in batches:
            kwargs = {key: value.to(runner.device) for key, value in batch.items()}
            actual = manager.execute(kwargs)
            expected = runner.model.embed_multimodal(**kwargs)
            assert len(actual) == len(expected)
            for output, reference in zip(actual, expected):
                assert torch.isfinite(output).all()
                # Two BF16 ulps at unit scale, fixed before model validation.
                torch.testing.assert_close(output, reference, rtol=0.016, atol=0.016)
                max_error = max(max_error, (output - reference).abs().max().item())
            for output, snapshot in retained:
                torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
            retained.extend((output, output.clone()) for output in actual)
            stats.append(manager.get_cumulative_stats())
    assert stats[0]["graph_hits"] > 0
    assert stats[1]["graph_hits"] > stats[0]["graph_hits"]
    assert stats[2]["graph_misses"] > stats[1]["graph_misses"]
    assert stats[3]["graph_hits"] > stats[2]["graph_hits"]
    assert stats[4]["graph_hits"] > stats[3]["graph_hits"]
    assert stats[5]["graph_hits"] > stats[4]["graph_hits"]
    assert stats[6]["graph_hits"] > stats[5]["graph_hits"]
    return {"max_abs_error": max_error, "stats": stats}


@pytest.mark.parametrize(
    "model_id",
    params_with_marks({key: MODEL_CONFIGS[key] for key in ("qwen3_vl", "qwen3_5")}),
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_eonly_vit_cudagraph_outputs(
    model_id, vllm_runner, image_assets, video_assets, monkeypatch
):
    """Exercise startup capture, image/video replay, fallback and output ownership."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = MODEL_CONFIGS[model_id]
    token_budgets = [64, 256, 2048]
    processor = AutoProcessor.from_pretrained(config.model)
    first, second = [asset.pil_image for asset in image_assets]
    batches = [
        [first.resize((224, 224))],
        [second.resize((448, 224)), first.resize((224, 224))],
        [second.resize((1792, 1792))],
        [second.resize((224, 224))],
        [first.resize((1280, 720))],
        [first.resize((224, 224))],
    ]
    frames = sample_frames_from_video(video_assets[0].np_ndarrays, 2)
    video = np.stack(
        [np.asarray(Image.fromarray(frame).resize((224, 224))) for frame in frames]
    )
    with vllm_runner(
        config.model,
        dtype=config.dtype,
        mm_encoder_only=True,
        # FA padding NaNs are tracked separately in #57136.
        mm_encoder_attn_backend="FLASHINFER",
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 2, "video": 1},
        compilation_config={
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": token_budgets,
            "encoder_cudagraph_max_vision_items_per_batch": 2,
            "encoder_cudagraph_max_frames_per_batch": 2,
        },
    ) as model:
        mm_config = model.llm.llm_engine.vllm_config.model_config.multimodal_config
        processor_kwargs = mm_config.merge_mm_processor_kwargs({})
        inputs = [
            dict(
                processor.image_processor(
                    images=images, return_tensors="pt", **processor_kwargs
                )
            )
            for images in batches
        ]
        inputs.append(
            dict(
                processor.video_processor(
                    videos=[video],
                    do_sample_frames=False,
                    return_tensors="pt",
                    **processor_kwargs,
                )
            )
        )
        results = model.llm.collective_rpc(
            partial(_check_eonly_encoder_outputs, batches=inputs)
        )
        for result in results:
            assert result["stats"][0]["token_budgets"] == token_budgets
        print(f"E-only {model_id}: {results}")
