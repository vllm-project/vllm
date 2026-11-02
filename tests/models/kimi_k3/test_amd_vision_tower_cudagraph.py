# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    CompilationConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 vision tower encoder cudagraph tests require ROCm",
)


@pytest.fixture(scope="module")
def should_do_global_cleanup_after_test():
    """The module-scoped distributed state and captured graphs must outlive
    each test, so skip the per-test global cleanup from tests/conftest.py."""
    return False


@pytest.fixture(scope="module")
def vllm_config_with_dist():
    """VllmConfig with an initialized single-rank dist env."""
    cfg = VllmConfig()
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield cfg
        destroy_model_parallel()
        destroy_distributed_environment()
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        torch.set_default_dtype(prev_dtype)


def _init_weights(module: nn.Module):
    """Initialize empty parameter tensors to avoid NaN in random tests."""
    for p in module.parameters():
        if p.data.numel() > 0:
            if p.data.ndim >= 2:
                p.data.normal_(0.0, 0.02)
            else:
                p.data.fill_(0.01)


# ---------------------------------------------------------------------------
# Encoder CUDA graph tests
#
# KimiK3ForConditionalGeneration cannot be built from a tiny config (its
# KimiLinear text layers need a full MLA/KDA config), so the harnesses below
# subclass it and only replace __init__: every method under test is the real
# one, driven by a real ModelConfig, VllmConfig and EncoderCudaGraphManager.
# ---------------------------------------------------------------------------

_TINY_K3_CONFIG = {
    "architectures": ["KimiK3ForConditionalGeneration"],
    "model_type": "kimi_k3",
    "text_config": {
        "model_type": "kimi_linear",
        "hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "vocab_size": 1024,
    },
    "vision_config": {"num_hidden_layers": 1},
    "max_position_embeddings": 4096,
}
_BUDGETS = [64, 128]
_MAX_VISION_ITEMS = 2


def _model_config(path, max_model_len: int = 4096) -> ModelConfig:
    return ModelConfig(
        model=str(path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 4, "video": 0},
        trust_remote_code=True,
    )


def _vllm_config(
    model_config: ModelConfig,
    budgets: list[int],
) -> VllmConfig:
    return VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=2048,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=budgets,
            encoder_cudagraph_max_vision_items_per_batch=_MAX_VISION_ITEMS,
        ),
    )


class _EncoderHarness(KimiK3ForConditionalGeneration):
    """Adds the real vision tower and projectors with dummy weights."""

    def __init__(self, model_config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = model_config.hf_config
        self.model_config = model_config
        self.hidden_size = self.config.text_config.hidden_size
        v_cfg = self.config.vision_config
        self.device = torch.device("cuda:0")
        self.vision_tower = MoonViT3dPretrainedModel(v_cfg)
        self.mm_projector = KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=self.hidden_size,
            use_data_parallel=False,
            prefix="mm_projector",
        )
        self.to(device=self.device, dtype=torch.bfloat16)


def _build_encoder(path):
    """Real model plus a manager that has already captured its graphs."""
    cfg = _vllm_config(_model_config(path), _BUDGETS)
    with set_current_vllm_config(cfg):
        model = _EncoderHarness(cfg.model_config)
        _init_weights(model)
        model.eval()
        mgr = EncoderCudaGraphManager(cfg, model.device, torch.bfloat16, model)
        mgr.capture(torch.cuda.graph_pool_handle())
    return model, mgr


@pytest.fixture(scope="module")
def tiny_k3_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_kimi_k3")
    (path / "config.json").write_text(json.dumps(_TINY_K3_CONFIG))
    return path


@pytest.fixture(scope="module")
def encoder_stack(vllm_config_with_dist, tiny_k3_dir):
    return _build_encoder(tiny_k3_dir)


def _mm_kwargs(model, grids: list[list[int]], seed: int = 0) -> dict:
    patch = model.config.vision_config.patch_size
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_patches = sum(t * h * w for t, h, w in grids)
    pixel_values = torch.randn(
        num_patches,
        3,
        patch,
        patch,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return {"pixel_values": pixel_values, "grid_thws": grids}


def _execute(mgr, mm_kwargs) -> list[torch.Tensor]:
    with torch.inference_mode():
        return mgr.execute(mm_kwargs)


def _eager(model, mm_kwargs, tokens: list[int]) -> list[torch.Tensor]:
    with torch.inference_mode():
        return list(model.encoder_eager_forward(mm_kwargs).split(tokens))


def _legacy(model, mm_kwargs) -> torch.Tensor:
    """Reference that bypasses the encoder-cudagraph helpers entirely."""
    with torch.inference_mode():
        features = model.vision_tower(mm_kwargs["pixel_values"], mm_kwargs["grid_thws"])
        out = model.mm_projector(torch.cat(features))
    return out.reshape(-1, out.shape[-1])


def _graph_stats(mgr) -> tuple[int, int]:
    stats = mgr.get_cumulative_stats()
    return stats["graph_hits"], stats["graph_misses"]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow 2 bf16 ulps at the output scale (graph replay runs padded GEMMs).

    The outputs must be large enough for that bound to still catch real errors.
    """
    assert actual.shape == expected.shape
    bound = 2 * expected.abs().max().item() * 2**-7
    assert expected.float().std().item() > 10 * bound, "output too small to test"
    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= bound, f"max abs diff {diff} > {bound}"


def _out_tokens(model, grid: list[int]) -> int:
    kh, kw = model.config.vision_config.merge_kernel_size
    return (grid[1] // kh) * (grid[2] // kw)


@pytest.mark.parametrize(
    "grids",
    [
        [[1, 16, 16]],  # 64 tokens fill the 64 budget
        [[1, 14, 14]],  # 49 tokens replay it padded
        [[1, 14, 14], [1, 14, 28]],  # two images share one replay
    ],
    ids=["exact", "under", "two_images"],
)
def test_replay_matches_eager(encoder_stack, grids):
    """Graph replay matches the eager encoder and the plain vision tower."""
    model, mgr = encoder_stack
    mm_kwargs = _mm_kwargs(model, grids)
    tokens = [_out_tokens(model, g) for g in grids]
    hits, misses = _graph_stats(mgr)

    outs = _execute(mgr, mm_kwargs)

    assert _graph_stats(mgr) == (hits + len(grids), misses)
    for out, expected in zip(outs, _eager(model, mm_kwargs, tokens)):
        _assert_close(out, expected)
    _assert_close(torch.cat(outs), _legacy(model, mm_kwargs))


def test_oversize_item_falls_back_to_eager(encoder_stack):
    """196 tokens exceed the largest budget (128), alone and next to a graphed item."""
    model, mgr = encoder_stack
    big = _mm_kwargs(model, [[1, 28, 28]], seed=3)
    hits, misses = _graph_stats(mgr)

    (out_big,) = _execute(mgr, big)

    assert _graph_stats(mgr) == (hits, misses + 1)
    assert torch.equal(out_big, _eager(model, big, [196])[0])

    small = _mm_kwargs(model, [[1, 14, 14]], seed=4)
    mixed = {
        "pixel_values": torch.cat([small["pixel_values"], big["pixel_values"]]),
        "grid_thws": small["grid_thws"] + big["grid_thws"],
    }
    hits, misses = _graph_stats(mgr)

    out_small, out_big_again = _execute(mgr, mixed)

    assert _graph_stats(mgr) == (hits + 1, misses + 1)
    _assert_close(out_small, _eager(model, small, [49])[0])
    assert torch.equal(out_big_again, out_big)
