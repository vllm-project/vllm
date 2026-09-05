# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 fused HIP compressor tests."""

import pytest
import torch

from tests.kernels.attention.dsv4_compress_utils import (
    BYTE_EXACT_SHAPES,
    H512_NOPE,
    H512_TOKEN_STRIDE,
    KV_BLOCK_SIZE,
    SCENARIO_NAMES,
    build_scenario,
    build_shared_input,
    compare_to_triton,
    detect_gfx942,
    hip_available,
    run_hip,
    run_triton,
)

pytestmark = pytest.mark.skipif(
    not detect_gfx942(), reason="DSV4 fused compressor requires gfx942"
)


@pytest.fixture(autouse=True)
def _default_cuda_device():
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


@pytest.fixture(scope="module", autouse=True)
def _require_ops():
    assert hip_available(), "dsv4 compressor ops not registered in _rocm_C on gfx942"


def _native_op_and_args(ctx, kv_3d):
    shape = ctx.shape
    args = [
        ctx.state_cache_bf16,
        ctx.num_tokens,
        ctx.ape,
        ctx.token_to_req_t,
        ctx.positions_t,
        ctx.slot_mapping_t,
        ctx.block_table,
        shape.state_block_size,
        ctx.rms_weight.to(torch.float32),
        1e-6,
        ctx.cos_sin_cache,
        kv_3d,
        ctx.kv_slot_mapping_t,
        kv_3d.shape[1],
        shape.scale_dim,
    ]
    if shape.ratio == 128:
        plan_capacity = ctx.num_tokens // shape.ratio + ctx.block_table.shape[0] + 2
        args.extend(
            [
                torch.empty(plan_capacity, dtype=torch.int32),
                torch.empty(1, dtype=torch.int32),
            ]
        )
        return torch.ops._rocm_C.dsv4_hca_compress, args
    return torch.ops._rocm_C.dsv4_csa_compress, args


@pytest.mark.parametrize("shape", BYTE_EXACT_SHAPES, ids=lambda s: s.label)
@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_compress_matches_triton(shape, scenario):
    ctx = build_scenario(shape, scenario)
    ctx.build()
    build_shared_input(ctx)

    ref_flat, ref_3d = ctx.new_kv_cache()
    run_triton(ctx, ref_3d)
    torch.accelerator.synchronize()

    hip_flat, hip_3d = ctx.new_kv_cache()
    run_hip(ctx, hip_3d)
    torch.accelerator.synchronize()

    aligned, detail = compare_to_triton(ref_flat.cpu(), hip_flat.cpu(), ctx)
    assert aligned, (
        f"{shape.label}/{scenario}: {detail} (expected reference-equivalent)"
    )


@pytest.mark.parametrize("shape", BYTE_EXACT_SHAPES, ids=lambda s: s.label)
@pytest.mark.parametrize(
    "weight",
    [0.0, -0.0, float("nan"), -float("nan")],
    ids=["positive_zero", "negative_zero", "positive_nan", "negative_nan"],
)
def test_native_ocp_transform_edge_values(shape, weight):
    ctx = build_scenario(shape, "prefill_256")
    ctx.build()
    build_shared_input(ctx)
    ctx.rms_weight.fill_(weight)

    ref_flat, ref_3d = ctx.new_kv_cache()
    run_triton(ctx, ref_3d)
    hip_flat, hip_3d = ctx.new_kv_cache()
    run_hip(ctx, hip_3d)
    torch.accelerator.synchronize()

    aligned, detail = compare_to_triton(ref_flat.cpu(), hip_flat.cpu(), ctx)
    for ci in range(ctx.num_compress):
        block, offset = divmod(ci, KV_BLOCK_SIZE)
        base = offset * H512_TOKEN_STRIDE
        encoded = hip_flat[block, base : base + H512_NOPE]
        if weight == weight:
            expected = ref_flat[block, base : base + H512_NOPE]
            assert torch.equal(encoded, expected), (
                f"{shape.label}/{weight}: OCP zero bytes differ: {detail}"
            )
        else:
            assert set(encoded.cpu().tolist()) == {0x7F}
    assert aligned or weight != 0.0, f"{shape.label}/{weight}: {detail}"


@pytest.mark.parametrize("shape", BYTE_EXACT_SHAPES, ids=lambda s: s.label)
@pytest.mark.parametrize(
    "invalid",
    ["negative_count", "short_metadata", "cpu_input", "block_table", "kv_layout"],
)
def test_native_wrapper_rejects_invalid_inputs(shape, invalid):
    ctx = build_scenario(shape, "prefill_256")
    ctx.build()
    build_shared_input(ctx)
    _, kv_3d = ctx.new_kv_cache()
    op, args = _native_op_and_args(ctx, kv_3d)
    if invalid == "negative_count":
        args[1] = -1
    elif invalid == "short_metadata":
        args[4] = ctx.positions_t[:-1]
    elif invalid == "cpu_input":
        args[2] = ctx.ape.cpu()
    elif invalid == "block_table":
        args[6] = ctx.block_table.T
    else:
        args[11] = kv_3d[:, :, :H512_TOKEN_STRIDE]

    with pytest.raises(RuntimeError):
        op(*args)


@pytest.mark.parametrize("shape", BYTE_EXACT_SHAPES, ids=lambda s: s.label)
def test_native_compressor_graph_replay(shape):
    ctx = build_scenario(shape, "prefill_256")
    ctx.build()
    build_shared_input(ctx)
    _, kv_3d = ctx.new_kv_cache()
    op, args = _native_op_and_args(ctx, kv_3d)

    op(*args)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    torch.accelerator.synchronize()
    expected = kv_3d.clone()

    kv_3d.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(kv_3d, expected)
