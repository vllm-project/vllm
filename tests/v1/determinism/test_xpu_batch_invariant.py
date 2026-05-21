# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant kernel overrides on Intel XPU.

Verifies correctness (vs torch reference) and the batch-invariance property
(result for one item is bitwise identical regardless of other items in the
batch) for the ops registered with the "XPU" dispatch key:
  - aten::mm (via matmul_persistent Triton kernel)
  - aten::bmm
  - aten::_log_softmax
  - aten::softmax / aten::_softmax
  - aten::mean.dim

Note: aten::addmm, aten::matmul, and aten::linear all delegate to the same
matmul_persistent kernel tested via aten::mm.

Also tests the Triton rms_norm kernel (called directly, not via aten dispatch),
verifies that registering the XPU overrides correctly routes standard torch ops
through the batch-invariant implementations, and includes an end-to-end LLM
logprobs test demonstrating bitwise batch invariance at the inference level
(using Triton Attention backend).
"""

import os
import random

import pytest
import torch
from utils import (
    TEST_MODEL,
    _extract_step_logprobs,
    _random_prompt,
    skip_unsupported_xpu,
)

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.batch_invariant import (
    _register_common_overrides,
    _register_matmul_overrides,
    bmm_batch_invariant,
    log_softmax,
    matmul_persistent,
    mean_batch_invariant,
    mean_dim,
    mm_batch_invariant,
    rms_norm,
    softmax_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# BMM tests — bmm_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "B,M,K,N",
    [
        (1, 32, 64, 16),
        (4, 64, 128, 64),
        (8, 512, 1024, 512),
        (2, 1, 64, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_correctness(B, M, K, N, dtype):
    """bmm_batch_invariant matches torch.bmm within tolerance."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(B, M, K, dtype=dtype, device=device)
    b = torch.randn(B, K, N, dtype=dtype, device=device)

    expected = torch.bmm(a, b)
    actual = bmm_batch_invariant(a, b)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_batch_invariance(dtype):
    """Same slice gives bitwise-identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a_single = torch.randn(1, 64, 32, dtype=dtype, device=device)
    b_single = torch.randn(1, 32, 128, dtype=dtype, device=device)

    out_single = bmm_batch_invariant(a_single, b_single)

    # Embed the same slice in a larger batch with random neighbors
    a_batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    b_batch = torch.randn(8, 32, 128, dtype=dtype, device=device)
    a_batch[5] = a_single[0]
    b_batch[5] = b_single[0]

    out_batch = bmm_batch_invariant(a_batch, b_batch)

    assert torch.equal(out_single[0], out_batch[5])


# ---------------------------------------------------------------------------
# Matmul (mm) tests — matmul_kernel_persistent Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "M,K,N",
    [
        (128, 128, 128),
        (256, 512, 256),
        (1024, 1024, 1024),
        (2048, 4096, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_persistent_correctness(M, K, N, dtype):
    """matmul_persistent matches torch.matmul within tolerance."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    expected = torch.matmul(a.float(), b.float()).to(dtype)
    actual = matmul_persistent(a, b)

    rtol, atol = (2e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 2e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "K,N",
    [
        (4096, 4096),
        (4096, 11008),
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_persistent_batch_invariance(K, N, dtype):
    """Same row produces bitwise-identical result regardless of batch size.

    This is the core property: result for a fixed input row must not change
    when the M dimension (number of rows) changes. Non-batch-invariant BLAS
    implementations (e.g. oneMKL with split-k) fail this test.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    weight = torch.randn(K, N, dtype=dtype, device=device)
    probe_row = torch.randn(1, K, dtype=dtype, device=device)

    # Reference: single-row matmul
    ref = matmul_persistent(probe_row, weight)

    # Embed the probe row into batches of varying sizes
    batch_sizes = [2, 8, 32, 64, 128]
    for bs in batch_sizes:
        filler = torch.randn(bs - 1, K, dtype=dtype, device=device)
        # Place probe at a non-trivial position
        pos = min(3, bs - 1)
        if pos == 0:
            batch = torch.cat([probe_row, filler], dim=0)
        else:
            batch = torch.cat([filler[:pos], probe_row, filler[pos:]], dim=0)

        result = matmul_persistent(batch, weight)
        assert torch.equal(ref[0], result[pos]), (
            f"Batch invariance violated at batch_size={bs}, pos={pos}: "
            f"max_diff={(ref[0] - result[pos]).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Log-softmax tests — _log_softmax_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_correctness(rows, cols, dtype):
    """Triton log_softmax matches torch.log_softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.log_softmax(x, dim=-1)
    actual = log_softmax(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = log_softmax(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = log_softmax(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Softmax tests — pure PyTorch (exp/sum, no Triton kernel)
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_correctness(rows, cols, dtype):
    """Deterministic softmax matches torch.softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.softmax(x, dim=-1)
    actual = softmax_batch_invariant(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = softmax_batch_invariant(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = softmax_batch_invariant(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Mean reduction tests — mean_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((16, 128), 1),
        ((4, 64, 32), 1),
        ((8, 256), 0),
        ((2, 16, 64), 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_dim_correctness(shape, dim, dtype):
    """Triton mean_dim matches torch.mean."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(shape, dtype=dtype, device=device)

    expected = torch.mean(x.float(), dim=dim).to(dtype)
    actual = mean_dim(x, dim=dim)
    if actual.dtype != dtype:
        actual = actual.to(dtype)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.float32 else (1e-1, 1e-1)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_batch_invariance(dtype):
    """Same slice gives identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 512, dtype=dtype, device=device)
    out_single = mean_dim(row, dim=1)

    batch = torch.randn(8, 512, dtype=dtype, device=device)
    batch[3] = row[0]
    out_batch = mean_dim(batch, dim=1)

    assert torch.equal(out_single[0], out_batch[3])


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_multi_dim_batch_invariance(dtype):
    """mean_batch_invariant over multiple dims is batch-invariant."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 64, 32, dtype=dtype, device=device)
    out_single = mean_batch_invariant(row, dim=[1, 2])

    batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    batch[5] = row[0]
    out_batch = mean_batch_invariant(batch, dim=[1, 2])

    assert torch.equal(out_single[0], out_batch[5])


# ---------------------------------------------------------------------------
# RMS norm tests — _rms_norm_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_correctness_xpu(batch_size, hidden_size, dtype):
    """Triton rms_norm produces results close to a reference implementation."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Reference: manual RMS norm in float32
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = ((x_f32 / rms) * weight.float()).to(dtype)

    actual = rms_norm(x, weight, eps=eps)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_batch_invariance_xpu(dtype):
    """Same row gives identical rms_norm result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    hidden_size = 2048
    eps = 1e-6

    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    row = torch.randn(1, hidden_size, dtype=dtype, device=device)

    out_single = rms_norm(row, weight, eps=eps)

    batch = torch.randn(8, hidden_size, dtype=dtype, device=device)
    batch[4] = row[0]
    out_batch = rms_norm(batch, weight, eps=eps)

    assert torch.equal(out_single[0], out_batch[4])


# ---------------------------------------------------------------------------
# Dispatch override tests: verify that registering XPU overrides causes
# standard torch ops to route through batch-invariant implementations.
# ---------------------------------------------------------------------------


@pytest.fixture
def xpu_batch_invariant_lib():
    """Register XPU dispatch overrides and clean up after the test."""
    lib = torch.library.Library("aten", "IMPL")
    _register_matmul_overrides(lib, "XPU")
    _register_common_overrides(lib, "XPU")
    yield lib
    del lib


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_mm(xpu_batch_invariant_lib, dtype):
    """torch.mm routes to mm_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(64, 1024, dtype=dtype, device=device)
    b = torch.randn(1024, 512, dtype=dtype, device=device)

    via_torch = torch.mm(a, b)
    via_direct = mm_batch_invariant(a, b)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_bmm(xpu_batch_invariant_lib, dtype):
    """torch.bmm routes to bmm_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(4, 64, 32, dtype=dtype, device=device)
    b = torch.randn(4, 32, 128, dtype=dtype, device=device)

    via_torch = torch.bmm(a, b)
    via_direct = bmm_batch_invariant(a, b)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_log_softmax(xpu_batch_invariant_lib, dtype):
    """torch.log_softmax routes to batch-invariant impl after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.log_softmax(x, dim=-1)
    via_direct = log_softmax(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_softmax(xpu_batch_invariant_lib, dtype):
    """torch.softmax routes to softmax_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.softmax(x, dim=-1)
    via_direct = softmax_batch_invariant(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_override_mean(xpu_batch_invariant_lib, dtype):
    """torch.mean routes to mean_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(8, 512, dtype=dtype, device=device)

    via_torch = torch.mean(x, dim=1)
    via_direct = mean_batch_invariant(x, dim=[1])

    assert torch.equal(via_torch, via_direct)


# ---------------------------------------------------------------------------
# End-to-end LLM generation: batch invariance at inference level
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.timeout(600)
def test_e2e_logprobs_bs1_vs_bsN():
    """Logprobs for each prompt must be bitwise identical between BS=1 and BS=N.

    Runs prompts individually (BS=1) and batched (BS=N), then compares
    per-token logprobs to verify the batch composition does not affect results.

    This test validates that with batch-invariant mode enabled (matmul overrides
    + attention 2D kernel path), the full pipeline produces identical logprobs
    regardless of what other sequences are in the batch.

    Note:
    Only tested to work with the TRITON_ATTN backend.
    Other backends may have non-deterministic attention implementations that fail
    this test, even with individual batch-invariant kernels enabled.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    llm = LLM(
        model=TEST_MODEL,
        max_num_seqs=32,
        max_model_len=4096,
        dtype="auto",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        attention_config={"backend": "TRITON_ATTN"},
    )

    try:
        prompts = [_random_prompt(10, 50) for _ in range(16)]

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            logprobs=5,
        )

        # BS=1 runs
        bs1_logprobs = []
        bs1_tokens = []
        for p in prompts:
            outs = llm.generate([p], sp, use_tqdm=False)
            step_logprobs, token_ids = _extract_step_logprobs(outs[0])
            if step_logprobs is None:
                pytest.skip("Logprobs not available on this configuration.")
            bs1_logprobs.append(step_logprobs)
            bs1_tokens.append(token_ids)

        # BS=N run
        outs_batched = llm.generate(prompts, sp, use_tqdm=False)
        bsN_logprobs = []
        bsN_tokens = []
        for o in outs_batched:
            step_logprobs, token_ids = _extract_step_logprobs(o)
            if step_logprobs is None:
                pytest.skip("Logprobs not available on this configuration.")
            bsN_logprobs.append(step_logprobs)
            bsN_tokens.append(token_ids)

        # Compare
        failed = []
        for i, (lp1, lpN, t1, tN) in enumerate(
            zip(bs1_logprobs, bsN_logprobs, bs1_tokens, bsN_tokens)
        ):
            if t1 != tN:
                failed.append(f"Prompt {i}: different tokens bs1={t1} bsN={tN}")
                continue
            if not torch.equal(lp1, lpN):
                max_diff = torch.abs(lp1 - lpN).max().item()
                failed.append(f"Prompt {i}: logprob mismatch (max_diff={max_diff:.6e})")

        if failed:
            pytest.fail(
                f"BS=1 vs BS=N logprob mismatch in {len(failed)}/{len(prompts)} "
                f"prompts:\n" + "\n".join(failed[:5])
            )

    finally:
        llm.shutdown()
