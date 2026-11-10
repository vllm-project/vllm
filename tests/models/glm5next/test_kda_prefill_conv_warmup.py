# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup JIT warmup of the planar KDA prefill conv (`kda_prefill_conv`: `auto`,
the default, takes it on SM10x).

`planar_conv_warmup_plan` lists one layer configuration's kernels (the layer
registers each entry); compiling their keys must leave nothing for the first
prefill step to compile: a real call
with that configuration's shapes (the model's row stride and the dense row of
a spec-decode step's gathered rows, the conv-state layout, cache indices from
block tables of any width) must hit Triton's kernel cache for all three
launches: the per-sequence kernels do not specialize on the strides that vary
between steps, so the plan holds three or four compile keys.
"""

import pytest
import torch

from vllm.models.glm5next.nvidia.ops import kda_prefill_conv as planar
from vllm.platforms import current_platform
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128


def _kernel_caches(dispatchers) -> list[dict]:
    """Triton's compiled-variant caches of the kernels for the current device."""
    device = triton.runtime.driver.active.get_current_device()
    return [d.kernel.device_caches[device][0] for d in dispatchers]


def _compiled(dispatchers) -> list[int]:
    return [len(c) for c in _kernel_caches(dispatchers)]


def _step(dim: int, row: int, dim_first: bool, cache_indices: torch.Tensor, state_len):
    lens = [1, 1, 300, 40]
    x = torch.randn(sum(lens), row, device="cuda", dtype=torch.bfloat16)[:, :dim]
    weight = torch.randn(dim, 4, device="cuda")
    num_slots = int(cache_indices.max()) + 1  # every index must own a slot
    shape = (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim)
    pool = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return planar.causal_conv1d_planar(
        x,
        weight,
        None,
        pool if dim_first else pool.transpose(-1, -2),
        query_start_loc=torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        ),
        cache_indices=cache_indices,
        has_initial_state=torch.tensor([True, True, False, True], device="cuda"),
    )


# 4 heads: the fused projection's row is not 16-byte aligned, so the register
# kernel; 24 heads: TMA. The probe below starts from cleared kernel caches, so
# the file's position in the test run does not matter.
@pytest.mark.parametrize("num_heads", [4, 24])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
# 3 state positions (no speculative tokens) and 16 (13 speculative tokens):
# the two Triton stride classes of the conv-state pool in the DS layout.
@pytest.mark.parametrize("state_len", [3, 16])
@torch.inference_mode()
def test_planar_conv_warmup_covers_the_runtime_variants(
    num_heads, dim_first, state_len
):
    dim = 3 * num_heads * D
    row = dim + num_heads + 2 * D  # the fused projection's row
    plan = planar.planar_conv_warmup_plan(
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        bias_dtype=None,
        dim=dim,
        num_planes=3,
        # The fused projection's row and the dense row of the gathered
        # non-spec rows of a spec-decode step.
        row_strides=(row, dim),
        conv_state_dim_first=dim_first,
        state_len=state_len,
    )
    # The dense row is 16-byte aligned at every head count, the projection's
    # row only at multiples of 8 heads.
    dense = "tma" if current_platform.has_device_capability(90) else "regs"
    strided = dense if num_heads % 8 == 0 else "regs"

    def stencil(kind):
        return (
            planar._planar_stencil_tma if kind == "tma" else planar._planar_stencil_regs
        )

    # One stencil case per row stride, then the fixup and the state update
    # once: three or four compile keys per layer.
    assert [d for d, _ in plan] == [
        stencil(strided),
        stencil(dense),
        planar._planar_fixup,
        planar._planar_state_update,
    ]
    dispatchers = list(
        dict.fromkeys(
            [
                stencil(strided),
                stencil(dense),
                planar._planar_fixup,
                planar._planar_state_update,
            ]
        )
    )
    # Start from empty kernel caches so the probe below sees this case's
    # compiles only (another case or test may have compiled the same variants;
    # the disk cache keeps the recompile cheap).
    for cache in _kernel_caches(dispatchers):
        cache.clear()
    for dispatcher, kwargs in plan:
        keys = dispatcher.get_warmup_keys(**kwargs)
        assert keys, (dispatcher.__name__, kwargs)
        dispatcher.compile_many(keys)
    before = _compiled(dispatchers)
    assert all(before)
    # Cache indices as column 0 of block tables of three widths (strides 16, 7
    # and 1): the per-sequence kernels do not specialize on that stride, so
    # none of them may compile anything.
    for width in (16, 7, 1):
        table = torch.arange(1, 1 + 4 * width, dtype=torch.int32, device="cuda")
        out = _step(dim, row, dim_first, table.view(4, width)[:, 0], state_len)
        torch.accelerator.synchronize()
        assert out.shape == (3, 342, dim // 3)
        assert _compiled(dispatchers) == before, (
            f"the call with cache-index stride {width} compiled a variant the "
            "warmup did not cover"
        )
    # The dense rows of a spec-decode step.
    table = torch.arange(1, 5, dtype=torch.int32, device="cuda")
    out = _step(dim, dim, dim_first, table, state_len)
    torch.accelerator.synchronize()
    assert out.shape == (3, 342, dim // 3)
    assert _compiled(dispatchers) == before, (
        "the dense-row call compiled a variant the warmup did not cover"
    )
    # Positive control for the probe: int64 cache indices are a new variant
    # of the two per-sequence kernels (the stencil never sees them).
    table = torch.arange(1, 5, dtype=torch.int64, device="cuda")
    _step(dim, row, dim_first, table, state_len)
    torch.accelerator.synchronize()
    after = _compiled(dispatchers)
    assert after[:-2] == before[:-2]
    assert after[-2:] == [n + 1 for n in before[-2:]]
