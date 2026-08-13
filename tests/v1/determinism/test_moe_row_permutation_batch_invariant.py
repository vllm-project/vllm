# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused-MoE expert GEMM must not depend on row order within an expert.

All2all dispatch kernels assign each token a slot in the destination expert's
buffer with an atomic increment (MoRI's ``intranode.hpp``, DeepEP low-latency's
``internode_ll.cu``), so whichever warp arrives first gets the low slot and the
row order inside an expert's buffer is nondeterministic run to run. The routing
is still correct -- a map records each token's slot -- so the whole scheme rests
on the expert GEMM being invariant to a permutation of the rows it is handed.

The permutation applied here is a global derangement of the token buffer, which
is exactly a permutation of the rows within every expert's group: per-expert
membership and counts are untouched, only the arrival order changes.

Two guards keep the test from passing vacuously:
  * ``_slot_movement`` asserts the permutation really does move rows to
    different blocks and lanes of the ``moe_align_block_size`` layout; and
  * the same comparison without the un-permute must fail.
"""

import warnings

import pytest
import torch

from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

from .utils import rows_that_differ, skip_if_not_cuda_alike

DEVICE_TYPE = current_platform.device_type

E = 8
N = 512
K = 256
TOP_K = 2


def _derangement(m: int, generator: torch.Generator) -> torch.Tensor:
    """A permutation with no fixed point, so no row is trivially left alone."""
    ar = torch.arange(m, device=generator.device)
    for _ in range(64):
        pi = torch.randperm(m, generator=generator, device=generator.device)
        if bool((pi != ar).all()):
            return pi
    return torch.roll(ar, 1)


def _slot_movement(topk_ids, permuted_topk_ids, pi, block_size) -> float:
    """Fraction of (token, slot) pairs that land in a different absolute slot
    of the padded expert buffer once the rows are permuted."""
    m, topk = topk_ids.shape
    invalid = m * topk
    sorted0, _, _ = moe_align_block_size(topk_ids, block_size, E)
    sorted1, _, _ = moe_align_block_size(permuted_topk_ids, block_size, E)

    def positions(sorted_ids, relabel=False):
        pos = torch.full((invalid,), -1, dtype=torch.long, device=sorted_ids.device)
        flat = sorted_ids.long()
        idx = torch.nonzero(flat < invalid).flatten()
        val = flat[idx]
        if relabel:
            val = pi[val // topk].long() * topk + val % topk
        pos[val] = idx
        return pos

    p0, p1 = positions(sorted0), positions(sorted1, relabel=True)
    seen = (p0 >= 0) & (p1 >= 0)
    return float(((p0 != p1) & seen).sum().item()) / max(int(seen.sum().item()), 1)


def _quantize_blockwise(w: torch.Tensor, block: int = 128):
    fp8_min, fp8_max = get_fp8_min_max()
    e, x, y = w.shape
    view = w.float().view(e, x // block, block, y // block, block)
    scale = view.abs().amax(dim=(2, 4)).clamp(1e-4) / fp8_max
    q = (
        (view / scale[:, :, None, :, None])
        .clamp(fp8_min, fp8_max)
        .to(current_platform.fp8_dtype())
    )
    return q.view(e, x, y), scale.contiguous()


def _make_case(m: int, seed: int, scheme: str):
    device = torch.device(DEVICE_TYPE)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((m, K), generator=gen, device=device) / 5).bfloat16()
    score = torch.randn((m, E), generator=gen, device=device)
    topk_weights, topk_ids, _ = fused_topk(x, score, TOP_K, renormalize=False)
    w1 = (torch.randn((E, 2 * N, K), generator=gen, device=device) / 10).bfloat16()
    w2 = (torch.randn((E, K, N), generator=gen, device=device) / 10).bfloat16()

    if scheme == "bf16":
        quant_config = FusedMoEQuantConfig.make(None)
    else:
        w1q, w1s = _quantize_blockwise(w1)
        w2q, w2s = _quantize_blockwise(w2)
        w1, w2 = w1q, w2q
        quant_config = FusedMoEQuantConfig.make(
            current_platform.fp8_dtype(),
            per_act_token_quant=False,
            block_shape=[128, 128],
            w1_scale=w1s,
            w2_scale=w2s,
        )
    return x, w1, w2, topk_weights, topk_ids, quant_config, gen


@skip_if_not_cuda_alike
@pytest.mark.parametrize("m", [17, 65, 129, 512])
@pytest.mark.parametrize(
    "scheme",
    [
        "bf16",
        pytest.param(
            "fp8_block",
            marks=pytest.mark.skipif(
                not current_platform.supports_fp8(), reason="needs fp8"
            ),
        ),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_expert_gemm_is_row_permutation_invariant(
    default_vllm_config, m: int, scheme: str, seed: int
):
    """Permuting the rows within each expert's buffer must not move a bit."""
    x, w1, w2, topk_weights, topk_ids, quant_config, gen = _make_case(m, seed, scheme)

    def run(a, weights, ids):
        return fused_experts(
            a, w1, w2, weights, ids, global_num_experts=E, quant_config=quant_config
        )

    baseline = run(x, topk_weights, topk_ids)
    assert rows_that_differ(baseline, run(x, topk_weights, topk_ids)).numel() == 0, (
        "the expert GEMM is not even run-to-run stable; nothing below is interpretable"
    )

    pi = _derangement(m, gen)
    permuted_ids = topk_ids[pi].contiguous()
    assert torch.equal(
        torch.bincount(topk_ids.flatten().long(), minlength=E),
        torch.bincount(permuted_ids.flatten().long(), minlength=E),
    ), "the permutation changed per-expert counts; it is not a row permutation"

    # Guard against a vacuous pass: the rows must genuinely relocate.  The
    # batch-invariant config uses BLOCK_SIZE_M=64.
    moved = _slot_movement(topk_ids, permuted_ids, pi, 64)
    assert moved > 0.1, (
        f"only {moved:.1%} of rows changed slot -- this M does not exercise "
        "row reordering, so a pass would mean nothing"
    )

    permuted = run(x[pi].contiguous(), topk_weights[pi].contiguous(), permuted_ids)
    inverse = torch.empty_like(pi)
    inverse[pi] = torch.arange(m, device=pi.device)

    bad = rows_that_differ(baseline, permuted[inverse])
    assert bad.numel() == 0, (
        f"{bad.numel()}/{m} output rows changed when the rows were permuted "
        f"within their experts (first offenders: {bad[:8].tolist()}). All2all "
        "dispatch orders rows nondeterministically, so this is a batch-"
        "invariance defect, not a layout detail."
    )

    # Positive control: without the un-permute the very same comparison must
    # fail, otherwise it is blind to row identity.
    assert rows_that_differ(baseline, permuted).numel() > 0


@skip_if_not_cuda_alike
@pytest.mark.parametrize("m", [512, 1024])
@pytest.mark.parametrize(
    "scheme",
    [
        "bf16",
        pytest.param(
            "fp8_block",
            marks=pytest.mark.skipif(
                not current_platform.supports_fp8(), reason="needs fp8"
            ),
        ),
    ],
)
@torch.inference_mode()
def test_expert_gemm_tolerates_native_sort_nondeterminism(
    default_vllm_config, m: int, scheme: str
):
    """The same reordering happens without any all2all at all.

    ``moe_align_block_size`` picks the large-batch path once
    ``topk_ids.numel() >= 1024`` (or ``num_experts > 64``), and there
    ``_count_and_sort_expert_tokens`` takes each row's rank inside its expert
    from an ``atomicAdd``. So ``sorted_token_ids`` is already nondeterministic
    run to run on a single GPU. Assert the expert GEMM does not notice.
    """
    x, w1, w2, topk_weights, topk_ids, quant_config, _ = _make_case(m, 0, scheme)
    assert topk_ids.numel() >= 1024, "M too small to reach the atomic sort path"

    def run():
        return fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=E,
            quant_config=quant_config,
        )

    baseline = run()
    sorted_baseline, _, _ = moe_align_block_size(topk_ids, 64, E)
    sorted_baseline = sorted_baseline.clone()

    reordered = 0
    for _ in range(16):
        bad = rows_that_differ(baseline, run())
        assert bad.numel() == 0, (
            f"{bad.numel()}/{m} output rows are not run-to-run stable "
            f"(first offenders: {bad[:8].tolist()})"
        )
        sorted_ids, _, _ = moe_align_block_size(topk_ids, 64, E)
        reordered += int(not torch.equal(sorted_ids, sorted_baseline))

    # Warn rather than skip: the 16 stability assertions above did run, and the
    # only thing not exercised is invariance under a reordering that did not
    # occur.
    if reordered == 0:
        warnings.warn(
            "moe_align_block_size produced a stable order in all 16 trials, so "
            "run-to-run stability was verified but invariance *under row "
            "reordering* was not exercised. Not a failure, and not a skip "
            "either: the assertions above did run.",
            stacklevel=2,
        )


@skip_if_not_cuda_alike
@pytest.mark.parametrize("m", [17, 65, 129])
@torch.inference_mode()
def test_naive_and_aligned_block_assignment_agree(default_vllm_config, m: int):
    """`_prepare_expert_assignment` flips row-assignment strategy at
    ``num_tokens * top_k * 4 <= global_num_experts``. Both sides must produce
    the same bits at a fixed M: the naive path pins each row to lane 0 of its
    own tile, the aligned path puts it at an arbitrary lane of a shared tile."""
    x, w1, _, _, topk_ids, _, _ = _make_case(m, 0, "bf16")
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
    }

    def gemm(force_naive: bool):
        # num_tokens only feeds the branch predicate here, so lying about it is
        # how the naive side is forced at a fixed M.
        sorted_ids, expert_ids, num_post_pad = _prepare_expert_assignment(
            topk_ids,
            config,
            0 if force_naive else m,
            TOP_K,
            E,
            None,
            ignore_invalid_experts=True,
        )
        assert (sorted_ids is None) == force_naive
        out = torch.empty((m, TOP_K, 2 * N), device=x.device, dtype=x.dtype)
        invoke_fused_moe_triton_kernel(
            x,
            w1,
            out,
            None,
            None,
            None,
            sorted_ids,
            expert_ids,
            num_post_pad,
            False,
            TOP_K,
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        return out.reshape(m, -1)

    bad = rows_that_differ(gemm(False), gemm(True))
    assert bad.numel() == 0, (
        f"{bad.numel()}/{m} rows differ between the naive and aligned block "
        f"assignments (first offenders: {bad[:8].tolist()})"
    )
