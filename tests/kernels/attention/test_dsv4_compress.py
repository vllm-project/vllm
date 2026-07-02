# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the DeepSeek-V4 fused HIP compressors (gfx950 / CDNA4).

Three kernels share one front-end (RMSNorm + GPT-J RoPE over a softmax-pooled
window) and differ only in the quant tail / cache layout:

  * CSA          head_dim=512, ratio=4    NOPE 448 FP8 + ROPE 64 bf16 + ue8m0
  * HCA          head_dim=512, ratio=128  (same packed layout, wider window)
  * indexer FP8  head_dim=128, ratio=4    whole-row FP8 + single fp32 scale
  * indexer MXFP4 head_dim=128, ratio=4   E2M1 nibbles + per-32 ue8m0

The vLLM Triton FP32 path is the byte-exact oracle for the FP8 outputs. The
MXFP4 tail is checked against a faithful torch front-end + RNE MXFP4 reference
(the Triton MXFP4 kernel uses NVIDIA PTX and cannot run on AMD).

gfx950-only: the kernels use native FP8/FP4 cvt builtins and are built into
_rocm_C only when gfx950 is a target arch. Skipped on every other platform.
"""

import pytest
import torch

from tests.kernels.attention.dsv4_compress_utils import (
    BYTE_EXACT_SHAPES,
    INDEXER_MXFP4,
    SCENARIO_NAMES,
    build_scenario,
    build_shared_input,
    compare_to_triton,
    detect_gfx950,
    hip_available,
    mxfp4_oracle_diff,
    run_hip,
    run_triton,
)

pytestmark = pytest.mark.skipif(
    not detect_gfx950(), reason="DSV4 fused compressor requires gfx950 (CDNA4)"
)


@pytest.fixture(autouse=True)
def _default_cuda_device():
    """The scenario/state-cache builders create CUDA tensors via the default
    device; pin it here (only runs on gfx950, where CUDA is present)."""
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


@pytest.fixture(scope="module", autouse=True)
def _require_ops():
    assert hip_available(), "dsv4 compressor ops not registered in _rocm_C on gfx950"


@pytest.mark.parametrize("shape", BYTE_EXACT_SHAPES, ids=lambda s: s.label)
@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_compress_matches_triton(shape, scenario):
    """HIP packed cache is reference-equivalent to the Triton FP32 oracle."""
    ctx = build_scenario(shape, scenario)
    ctx.build()
    build_shared_input(ctx)

    ref_flat, ref_3d = ctx.new_kv_cache()
    run_triton(ctx, ref_3d)
    torch.cuda.synchronize()

    hip_flat, hip_3d = ctx.new_kv_cache()
    run_hip(ctx, hip_3d)
    torch.cuda.synchronize()

    aligned, detail = compare_to_triton(ref_flat.cpu(), hip_flat.cpu(), ctx)
    assert aligned, (
        f"{shape.label}/{scenario}: {detail} (expected reference-equivalent)"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_indexer_mxfp4_matches_torch_oracle(scenario):
    """HIP MXFP4 tail matches a faithful torch front-end + RNE MXFP4 reference."""
    ctx = build_scenario(INDEXER_MXFP4, scenario)
    ctx.build()
    build_shared_input(ctx)

    hip_flat, hip_3d = ctx.new_kv_cache()
    run_hip(ctx, hip_3d)
    torch.cuda.synchronize()

    dmean = mxfp4_oracle_diff(ctx, hip_flat.cpu())
    assert dmean < 1e-2, (
        f"indexer_mxfp4/{scenario}: dmean={dmean:.5f} "
        f"(expected FP4 RNE-tie reference-equivalent)"
    )
