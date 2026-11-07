# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import MuseGlimmerConfig

import vllm.utils.gpu_sync_debug as gpu_sync_debug
from vllm.model_executor.models.muse_glimmer import MuseGlimmerVisionEncoder
from vllm.models.muse_spark.common.vision import (
    NaViT,
    _get_sparse_attention_perm,
    _seqlens_to_cu_seqlens,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.muse_spark import MuseSparkVisionConfig
from vllm.utils.torch_utils import set_default_torch_dtype

from ..registry import HF_EXAMPLE_MODELS
from ..utils import dummy_hf_overrides

GRIDS = [(2, 2), (16, 16), (32, 32), (34, 34), (64, 64), (74, 74), (22, 286), (96, 170)]


def _block_major_order(grid_height: int, grid_width: int, block: int) -> np.ndarray:
    rows, cols = np.divmod(np.arange(grid_height * grid_width), grid_width)
    return np.lexsort((cols % block, rows % block, cols // block, rows // block))


def _window_lengths(grid_height: int, grid_width: int) -> list[int]:
    return [
        min(32, grid_height - row) * min(32, grid_width - col)
        for row in range(0, grid_height, 32)
        for col in range(0, grid_width, 32)
    ]


def _check_without_gpu_sync(monkeypatch, fn, *args):
    monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", False)
    gpu_sync_debug.enable_gpu_sync_check()
    return gpu_sync_debug.with_gpu_sync_check(fn)(*args)


def _tiny_glimmer_encoder(
    layer_types: list[str], dtype: torch.dtype, pos_emb_size: int = 4
) -> MuseGlimmerVisionEncoder:
    config = MuseGlimmerConfig(
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            pos_emb_height=pos_emb_size,
            pos_emb_width=pos_emb_size,
        ),
        out_hidden_size=256,
    )
    with torch.device("cuda"), set_default_torch_dtype(dtype):
        encoder = MuseGlimmerVisionEncoder(config)
    with torch.no_grad():
        for param in encoder.parameters():
            param.normal_(std=0.1)
    return encoder


@pytest.fixture
def glimmer_layout_encoder(default_vllm_config, dist_init) -> MuseGlimmerVisionEncoder:
    return _tiny_glimmer_encoder(["window_attention"], torch.float32, pos_emb_size=32)


def _pixel_values(grid_thw: torch.Tensor, encoder: MuseGlimmerVisionEncoder):
    return torch.randn(
        int(grid_thw.prod(-1).sum()),
        encoder.conv1_linear.in_features,
        device="cuda",
        dtype=encoder.conv1_linear.weight.dtype,
    )


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("grid_height", "grid_width"), GRIDS)
def test_glimmer_sparse_permutation_orders_tokens_by_window(
    glimmer_layout_encoder: MuseGlimmerVisionEncoder, grid_height: int, grid_width: int
) -> None:
    permutation, seq_lens = glimmer_layout_encoder._get_sparse_permutation(
        grid_height,
        grid_width,
    )

    np.testing.assert_array_equal(
        permutation, _block_major_order(grid_height, grid_width, block=32)
    )
    assert seq_lens == _window_lengths(grid_height, grid_width)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("grid_height", "grid_width"), GRIDS + [(128, 128)])
def test_glimmer_pos_emb_layout_is_bilinear_interpolation(
    glimmer_layout_encoder: MuseGlimmerVisionEncoder, grid_height: int, grid_width: int
) -> None:
    table = torch.randn(32, 32, 8, dtype=torch.float64)

    indices, weights = glimmer_layout_encoder._get_pos_emb_layout(
        grid_height,
        grid_width,
    )
    actual = (
        table.reshape(32 * 32, 8)[torch.from_numpy(indices)]
        * torch.from_numpy(weights).double()[..., None]
    ).sum(0)

    y = (torch.arange(grid_height, dtype=torch.float64) + 0.5) / grid_height * 2 - 1
    x = (torch.arange(grid_width, dtype=torch.float64) + 0.5) / grid_width * 2 - 1
    grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)[None]
    expected = F.grid_sample(
        table.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False
    )[0].reshape(8, -1)
    torch.testing.assert_close(actual, expected.T, rtol=0, atol=5e-5)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("grid_height", "grid_width"), [(2, 2), (6, 10), (74, 74)])
def test_glimmer_pixel_shuffle_order_groups_merge_blocks(
    glimmer_layout_encoder: MuseGlimmerVisionEncoder, grid_height: int, grid_width: int
) -> None:
    order = glimmer_layout_encoder._get_pixel_shuffle_order(
        grid_height,
        grid_width,
    )

    np.testing.assert_array_equal(
        order, _block_major_order(grid_height, grid_width, block=2)
    )


@pytest.mark.cpu_test
@pytest.mark.core_model
@pytest.mark.parametrize(("grid_height", "grid_width"), GRIDS)
def test_navit_sparse_permutation_orders_tokens_by_window(
    grid_height: int, grid_width: int
) -> None:
    offset = 7

    permutation, seq_lens, next_offset = _get_sparse_attention_perm(
        offset, grid_height, grid_width, 32, 32
    )

    np.testing.assert_array_equal(
        permutation, offset + _block_major_order(grid_height, grid_width, block=32)
    )
    assert seq_lens.tolist() == _window_lengths(grid_height, grid_width)
    assert next_offset == offset + grid_height * grid_width


@pytest.mark.cpu_test
@pytest.mark.core_model
def test_navit_cu_seqlens() -> None:
    cu_seqlens, max_seqlen = _seqlens_to_cu_seqlens([3, 1, 4], torch.device("cpu"))

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 3, 4, 8]
    assert max_seqlen.device.type == "cpu" and max_seqlen.item() == 4


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "layer_types",
    [["window_attention", "full_attention"], ["full_attention", "full_attention"]],
)
def test_glimmer_vision_encoder_is_batch_invariant(
    default_vllm_config, dist_init, layer_types: list[str]
) -> None:
    encoder = _tiny_glimmer_encoder(layer_types, torch.float32)
    items = [
        torch.tensor([[1, 6, 10]]),
        torch.tensor([[1, 4, 8]]),
        torch.tensor([[2, 4, 4]]),
    ]
    pixel_values = [_pixel_values(grid_thw, encoder) for grid_thw in items]

    together = encoder(torch.cat(pixel_values), torch.cat(items))
    alone = torch.cat(
        [encoder(pv, grid_thw) for pv, grid_thw in zip(pixel_values, items)]
    )

    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-4)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glimmer_vision_encoder_rejects_grid_not_divisible_by_merge_size(
    default_vllm_config, dist_init
) -> None:
    encoder = _tiny_glimmer_encoder(["window_attention"], torch.float32)
    grid_thw = torch.tensor([[1, 6, 5]])

    with pytest.raises(ValueError, match="divisible by the merge size"):
        encoder(_pixel_values(grid_thw, encoder), grid_thw)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glimmer_vision_encoder_projects_patches_per_group(
    default_vllm_config, dist_init
) -> None:
    """One matmul over mixed-size groups is not bit-identical on all GPUs."""
    encoder = _tiny_glimmer_encoder(["window_attention"], torch.bfloat16)
    grid_thw = torch.tensor([[1, 6, 10], [2, 4, 4]])
    num_patches: list[int] = []
    encoder.conv1_linear.register_forward_hook(
        lambda _, args, __: num_patches.append(args[0].shape[-2])
    )

    encoder(_pixel_values(grid_thw, encoder), grid_thw)

    assert num_patches == [60, 16, 16]


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glimmer_engine_has_no_gpu_sync(monkeypatch, vllm_runner) -> None:
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    arch = "MuseGlimmerForConditionalGeneration"
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")
    video = np.random.default_rng(0).integers(0, 255, (8, 720, 1280, 3), dtype=np.uint8)

    with vllm_runner(
        model_info.default,
        trust_remote_code=False,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1, "video": 1},
        hf_overrides=partial(dummy_hf_overrides, model_arch=arch),
    ) as runner:
        outputs = runner.llm.generate(
            [
                {
                    "prompt": "<|patch|>Describe.",
                    "multi_modal_data": {"image": Image.new("RGB", (1024, 768))},
                },
                {
                    "prompt": "<|video|>Describe.",
                    "multi_modal_data": {"video": video},
                },
            ],
            SamplingParams(temperature=0, max_tokens=4),
        )

    assert [len(output.outputs[0].token_ids) for output in outputs] == [4, 4]


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glimmer_vision_encoder_has_no_gpu_sync(
    monkeypatch, default_vllm_config, dist_init
) -> None:
    encoder = _tiny_glimmer_encoder(
        ["window_attention", "full_attention"], torch.bfloat16
    )
    grid_thw = torch.tensor([[1, 6, 10], [2, 4, 4]])
    pixel_values = _pixel_values(grid_thw, encoder)
    encoder(pixel_values, grid_thw)

    features = _check_without_gpu_sync(monkeypatch, encoder, pixel_values, grid_thw)

    assert features.shape == (60 // 4 + 2 * 16 // 4, 256)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_navit_has_no_gpu_sync(monkeypatch, default_vllm_config, dist_init) -> None:
    config = MuseSparkVisionConfig(
        hidden_act="gelu",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        layer_types=["window_attention", "full_attention"],
        layer_norm_eps=1e-5,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        navit = NaViT(config, projector_hidden_size=32)
    visions = [
        torch.randn(3, 6 * 14, 10 * 14, device="cuda", dtype=torch.bfloat16),
        torch.randn(4, 3, 4 * 14, 4 * 14, device="cuda", dtype=torch.bfloat16),
    ]
    navit(visions)

    features = _check_without_gpu_sync(monkeypatch, navit, visions)

    assert features.shape == (60 // 4 + 2 * 16 // 4, 32)
