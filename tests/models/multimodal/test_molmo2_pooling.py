# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.molmo2 import _prepare_molmo2_pooling

pytestmark = pytest.mark.skip_global_cleanup


def _reference_pooling_preparation(
    image_features: torch.Tensor,
    token_pooling: torch.Tensor,
    masked_average: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, _, dim = image_features.shape
    valid = token_pooling >= 0
    batch_idx = torch.arange(
        token_pooling.shape[0],
        dtype=torch.long,
        device=token_pooling.device,
    )
    batch_idx = torch.tile(
        batch_idx.view(batch_size, 1, 1),
        [1, token_pooling.shape[1], token_pooling.shape[2]],
    )
    to_pool = image_features.reshape(batch_size, -1, dim)[
        batch_idx, torch.clamp(token_pooling, min=0)
    ]
    to_pool = to_pool * valid.to(image_features.dtype)[..., None]
    to_pool = to_pool.reshape(-1, token_pooling.shape[-1], dim)

    if masked_average:
        denom = valid.reshape(-1, valid.shape[-1]).float().sum(-1).clamp_min(1)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
    else:
        query = to_pool.mean(-2, keepdim=True)

    return (
        to_pool,
        query,
        valid.reshape(-1, 1, 1, valid.shape[-1]),
        valid.any(-1),
    )


@pytest.mark.parametrize("masked_average", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_molmo2_pooling_native_contract(
    default_vllm_config,
    masked_average: bool,
    index_dtype: torch.dtype,
) -> None:
    image_features = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(
        2, 3, 5, 7
    )
    token_pooling = torch.tensor(
        [
            [[0, 1, 2, 3], [5, -1, 5, 14], [-1, -1, -1, -1]],
            [[14, 0, 7, 7], [3, 4, -1, 6], [8, 9, 10, 11]],
        ],
        dtype=index_dtype,
    )

    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average
    )
    actual = _prepare_molmo2_pooling(
        image_features,
        token_pooling,
        masked_average=masked_average,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_molmo2_pooling_preserves_non_finite_mask_semantics(
    default_vllm_config,
) -> None:
    image_features = torch.zeros(1, 1, 4, 8)
    image_features[0, 0, 0, 0] = torch.nan
    token_pooling = torch.full((1, 1, 4), -1, dtype=torch.int32)

    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average=True
    )
    actual = _prepare_molmo2_pooling(image_features, token_pooling, masked_average=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("masked_average", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("batch_size", "num_crops", "num_patches", "num_groups", "pool_size", "dim"),
    [
        pytest.param(1, 2, 16, 8, 4, 2304, id="k4-image"),
        pytest.param(2, 3, 27, 17, 9, 2304, id="k9-video"),
        pytest.param(2, 1, 16, 5, 4, 128, id="k4-small"),
    ],
)
def test_molmo2_pooling_cuda_matches_reference(
    default_vllm_config,
    masked_average: bool,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    batch_size: int,
    num_crops: int,
    num_patches: int,
    num_groups: int,
    pool_size: int,
    dim: int,
) -> None:
    torch.manual_seed(0)
    image_features = torch.randn(
        batch_size,
        num_crops,
        num_patches,
        dim,
        device="cuda",
        dtype=dtype,
    )
    token_pooling = torch.randint(
        0,
        num_crops * num_patches,
        (batch_size, num_groups, pool_size),
        device="cuda",
        dtype=index_dtype,
    )
    token_pooling[:, 0] = -1
    token_pooling[:, 1, 0] = -1
    token_pooling[:, 2, 1] = token_pooling[:, 2, 0]

    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average
    )
    actual = _prepare_molmo2_pooling(
        image_features, token_pooling, masked_average=masked_average
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    tolerance = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual[1], expected[1], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("masked_average", [False, True])
def test_molmo2_pooling_cuda_supports_strided_features(
    default_vllm_config, masked_average: bool
) -> None:
    padded_features = torch.randn(2, 3, 18, 128, device="cuda", dtype=torch.float16)
    image_features = padded_features[:, :, 1:17, :]
    assert not image_features.is_contiguous()
    token_pooling = torch.tensor(
        [
            [[0, 16, 31, -1], [15, 15, 17, 18]],
            [[47, 32, 1, 0], [-1, -1, -1, -1]],
        ],
        device="cuda",
    )

    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average
    )
    actual = _prepare_molmo2_pooling(
        image_features, token_pooling, masked_average=masked_average
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_molmo2_pooling_cuda_preserves_non_finite_mask_semantics(
    default_vllm_config,
) -> None:
    image_features = torch.zeros(1, 1, 4, 8, device="cuda", dtype=torch.float16)
    image_features[0, 0, 0, 0] = torch.nan
    token_pooling = torch.full((1, 1, 4), -1, device="cuda")
    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average=True
    )
    actual = _prepare_molmo2_pooling(image_features, token_pooling, masked_average=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_molmo2_pooling_cuda_supports_fullgraph_compile(
    default_vllm_config,
) -> None:
    image_features = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    token_pooling = torch.randint(0, 32, (1, 8, 4), device="cuda")
    token_pooling[:, 0] = -1
    expected = _prepare_molmo2_pooling(
        image_features, token_pooling, masked_average=True
    )
    compiled = torch.compile(_prepare_molmo2_pooling, fullgraph=True)
    actual = compiled(image_features, token_pooling, masked_average=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_molmo2_pooling_cuda_supports_graph_replay(
    default_vllm_config,
) -> None:
    image_features = torch.randn(1, 3, 27, 128, device="cuda", dtype=torch.bfloat16)
    token_pooling = torch.randint(0, 81, (1, 16, 9), device="cuda")
    for _ in range(3):
        _prepare_molmo2_pooling(image_features, token_pooling, masked_average=True)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _prepare_molmo2_pooling(
            image_features, token_pooling, masked_average=True
        )

    image_features.copy_(torch.randn_like(image_features))
    token_pooling.copy_(torch.randint_like(token_pooling, 0, 81))
    token_pooling[:, 0] = -1
    graph.replay()
    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average=True
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2)
