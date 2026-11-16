# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact parity of the fused state gather/scatter kernels with eager."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="The fused state gather/scatter kernels are Triton/CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (  # noqa: E402
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import (  # noqa: E402
    scatter_states,
)

# A small GDN-like state: (slots, heads, rows, cols).
SLOTS, HEADS, ROWS, COLS = 512 + 3, 4, 128, 128


def _make(batch, pool_dtype, strided, gen):
    # Mamba pages may pad the row stride; build the pool over padded storage.
    row_size = HEADS * ROWS * COLS
    storage = torch.randn(SLOTS, row_size + 256, dtype=pool_dtype, device="cuda")
    pool = storage[:, :row_size].view(SLOTS, HEADS, ROWS, COLS)
    assert not pool.is_contiguous() and pool[0].is_contiguous()
    has = torch.rand(batch, device="cuda", generator=gen) < 0.5
    if strided:
        # Spec-decode block table: rows hold 1 + num_spec state slots, so
        # column 0 is a strided view (degenerate but valid at batch == 1).
        table = torch.randperm(SLOTS, device="cuda", generator=gen)[: 2 * batch]
        indices = table.view(batch, 2).to(torch.int32)[:, 0]
        assert batch == 1 or not indices.is_contiguous()
    else:
        indices = torch.randperm(SLOTS, device="cuda", generator=gen)[:batch].to(
            torch.int32
        )
    return pool, indices, has


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("batch", [1, 3, 64, 256])
@pytest.mark.parametrize("pool_dtype", [torch.float32, torch.bfloat16])
def test_gather_correctness(pool_dtype, batch, strided):
    gen = torch.Generator(device="cuda").manual_seed(0)
    pool, indices, has = _make(batch, pool_dtype, strided, gen)

    want = pool[indices]
    want[~has] = 0
    got = gather_initial_states(pool, indices, has)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("batch", [1, 3, 64, 256])
@pytest.mark.parametrize(
    "pool_dtype,src_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_scatter_correctness(pool_dtype, src_dtype, batch, strided):
    gen = torch.Generator(device="cuda").manual_seed(0)
    pool, indices, _ = _make(batch, pool_dtype, strided, gen)
    src = torch.randn(batch, HEADS, ROWS, COLS, dtype=src_dtype, device="cuda")

    want = pool.clone()
    want[indices] = src.to(pool_dtype)
    got = pool.clone()
    scatter_states(got, src, indices)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
