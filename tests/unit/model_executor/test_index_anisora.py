import os
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required for diffusion inference"
)
def test_index_anisora_loads():
    try:
        from vllm.model_executor.models.index_anisora import (
            IndexAniSoraForVideoGeneration,
        )
    except Exception as e:
        pytest.skip(f"Import failed (diffusers likely missing): {e}")

    # Do not trigger a full generation in unit test - just load pipeline
    model = IndexAniSoraForVideoGeneration(
        model_id=os.environ.get("ANISORA_MODEL", "IndexTeam/Index-anisora"),
        torch_dtype=torch.float16,
        device="cuda",
    )
    assert model.pipe is not None


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required for diffusion inference"
)
def test_index_anisora_forward_smoke():
    try:
        from vllm.model_executor.models.index_anisora import (
            IndexAniSoraForVideoGeneration,
        )
    except Exception as e:
        pytest.skip(f"Import failed (diffusers likely missing): {e}")

    model = IndexAniSoraForVideoGeneration(
        model_id=os.environ.get("ANISORA_MODEL", "IndexTeam/Index-anisora"),
        torch_dtype=torch.float16,
        device="cuda",
    )

    # Keep params tiny for smoke test; real quality not validated here
    video = model.forward(
        prompt="A cat is running",
        num_frames=2,
        num_inference_steps=1,
        height=128,
        width=128,
        return_type="tensor",
    )
    assert isinstance(video, torch.Tensor)
    assert video.ndim == 4 and video.shape[-1] == 3
