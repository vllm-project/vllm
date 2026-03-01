# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import Generator

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

CUDA_DEVICE = "cuda" if current_platform.is_cuda() else None
DEVICE = current_platform.device_type

BATCH_SIZE = 1024
VOCAB_SIZE = 128 * 1024


@pytest.fixture(autouse=True)
def reset_default_device():
    """
    Explicitly set the default device, which can affect subsequent tests.
    Adding this fixture helps avoid this problem.
    """
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)


def test_topk_impl_equivalence():
    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(33)

    logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

    # Random top-k values between 1 and 9.
    k = torch.randint(1, 10, (BATCH_SIZE,), generator=generator)

    # Set k=vocab_size for ~50% of requests in the batch (top-k disabled).
    k.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=bool), VOCAB_SIZE
    )

    # Top-k only implementation
    result1 = apply_top_k_top_p_pytorch(logits=logits.clone(), k=k, p=None)

    # Top-p + top-k
    no_op_top_p = torch.tensor([1.0])
    result2 = apply_top_k_top_p_pytorch(logits=logits.clone(), k=k, p=no_op_top_p)

    assert torch.allclose(result1, result2)


@pytest.mark.skip(
    reason="FlashInfer top-k/top-p renorm comparison fails; "
    "needs investigation of tolerance threshold or "
    "interface differences between Python and FlashInfer implementations"
)
def test_flashinfer_sampler():
    """
    This test verifies that the FlashInfer top-k and top-p sampling
    implementation produces the same results as the Python implementation.

    NOTE: FlashInfer did not directly expose an interface for fused top-k and
    top-p prob renorm (it did provide fused sampling but we cannot compare
    sampling results due to randomness), so we will compare the probability
    renormed consequently by top-k and then top-p of FlashInfer implementation.
    """
    try:
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

        is_flashinfer_available = True
    except ImportError:
        is_flashinfer_available = False

    FLASHINFER_ENABLED = current_platform.is_cuda() and is_flashinfer_available

    if not FLASHINFER_ENABLED:
        pytest.skip("FlashInfer not installed or not available on this platform.")

    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(42)

    # Generate random logits
    logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

    # Generate various top-k and top-p values
    k_values = torch.randint(1, 1000, (BATCH_SIZE,), generator=generator)
    p_values = (
        torch.rand((BATCH_SIZE,), generator=generator) * 0.5 + 0.5
    )  # range in [0.5, 1.0]

    # Sometimes disable top-k (k=vocab_size)
    k_values.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool),
        VOCAB_SIZE,
    )

    # Sometimes disable top-p (p=1.0)
    p_values.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool), 1.0
    )

    python_logits = apply_top_k_top_p_pytorch(
        logits=logits.clone(),
        k=k_values,
        p=p_values,
    )
    python_probs = torch.softmax(python_logits, dim=-1)

    # FlashInfer only exposed renorm interfaces for probs so convert first
    flashinfer_probs = torch.softmax(logits.clone(), dim=-1)
    flashinfer_probs = top_k_renorm_probs(
        probs=flashinfer_probs,
        top_k=k_values,
    )
    flashinfer_probs = top_p_renorm_probs(
        probs=flashinfer_probs,
        top_p=p_values,
    )

    # Compare the results
    assert torch.allclose(python_probs, flashinfer_probs, atol=2e-2), (
        "FlashInfer and Python sampling implementations do not match!"
    )


# =============================================================================
# Triton kernel tests
# =============================================================================


@pytest.mark.skipif(CUDA_DEVICE is None, reason="CUDA not available")
class TestTritonTopkTopp:
    """Tests for the Triton top-k/top-p kernel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        torch.set_default_device(CUDA_DEVICE)
        self.generator = Generator(device=CUDA_DEVICE).manual_seed(42)

    def _compare_results(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ):
        """Compare Triton kernel results with PyTorch sorting implementation.

        For top-k only, we expect exact match.
        For top-p (with or without top-k), we allow small differences due to
        floating-point precision in probability sum calculations.
        """
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        # Clone logits for both implementations
        logits_pytorch = logits.clone()
        logits_triton = logits.clone().to(torch.float32)

        # Apply PyTorch sorting implementation
        result_pytorch = apply_top_k_top_p_pytorch(logits_pytorch, k, p)

        # Apply Triton kernel
        k_i32 = k.to(torch.int32) if k is not None else None
        p_f32 = p.to(torch.float32) if p is not None else None
        result_triton = apply_top_k_top_p_triton(logits_triton, k_i32, p_f32)

        # Compare kept counts per row
        pytorch_kept = (result_pytorch != float("-inf")).sum(dim=-1)
        triton_kept = (result_triton != float("-inf")).sum(dim=-1)

        if p is None:
            # Top-k only: expect exact match
            assert torch.equal(pytorch_kept, triton_kept), (
                f"Top-k mask mismatch: PyTorch kept {pytorch_kept.tolist()}, "
                f"Triton kept {triton_kept.tolist()}"
            )
        else:
            # Top-p involved: allow small differences
            # Either < 1% of kept values OR < 5 values absolute
            max_diff = (pytorch_kept - triton_kept).abs().max().item()
            max_kept = pytorch_kept.max().item()
            if max_kept > 0 and max_diff > 3:
                diff_pct = max_diff / max_kept * 100
                assert diff_pct < 0.5, (
                    f"Top-p mask difference too large: {diff_pct:.2f}% "
                    f"(max diff {max_diff} values out of {max_kept})"
                )

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topk_only(self, batch_size: int, vocab_size: int):
        """Test top-k only (p=None)."""
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        k = torch.randint(
            1, min(100, vocab_size), (batch_size,), generator=self.generator
        )
        # Randomly disable top-k for some rows (~25%)
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        k.masked_fill_(disable_mask, vocab_size)

        self._compare_results(logits, k, p=None)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topp_only(self, batch_size: int, vocab_size: int):
        """Test top-p only (k=None)."""
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]
        # Randomly disable top-p for some rows (~25%)
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        p.masked_fill_(disable_mask, 1.0)

        self._compare_results(logits, k=None, p=p)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topk_and_topp(self, batch_size: int, vocab_size: int):
        """Test combined top-k and top-p."""
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        k = torch.randint(
            1, min(100, vocab_size), (batch_size,), generator=self.generator
        )
        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]

        # Randomly disable top-k for some rows (~25%)
        disable_k = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        k.masked_fill_(disable_k, vocab_size)
        # Randomly disable top-p for some rows (~25%)
        disable_p = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        p.masked_fill_(disable_p, 1.0)

        self._compare_results(logits, k, p)

    def test_both_disabled(self):
        """Test when both k and p are None (should be no-op)."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        logits = torch.randn(32, 1024, generator=self.generator, dtype=torch.float32)
        logits_clone = logits.clone()

        result = apply_top_k_top_p_triton(logits_clone, k=None, p=None)

        assert torch.equal(result, logits), "Should be no-op when both k and p are None"

    def test_extreme_k_values(self):
        """Test edge cases for k values."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )

        # k=1 (keep only top 1)
        k = torch.ones(batch_size, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

        # k=vocab_size (keep all)
        k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

        # Mixed extreme values
        k = torch.tensor([1, vocab_size, 2, vocab_size - 1] * 4, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

    def test_extreme_p_values(self):
        """Test edge cases for p values."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )

        # p close to 0 (very restrictive)
        p = torch.full((batch_size,), 0.01, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

        # p=1.0 (keep all)
        p = torch.ones(batch_size, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

        # Mixed values
        p = torch.tensor([0.1, 0.5, 0.9, 1.0] * 4, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

    def test_large_batch(self):
        """Test with a large batch size."""
        batch_size, vocab_size = 512, 32000
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        k = torch.randint(1, 50, (batch_size,), generator=self.generator)
        p = torch.rand(batch_size, generator=self.generator) * 0.5 + 0.5

        self._compare_results(logits, k, p)

    # -----------------------------------------------------------------
    # Tests for -inf logits (e.g. from grammar / structured output masks)
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topk_with_neginf_logits(self, inf_fraction: float):
        """Top-k with many -inf logits (simulating grammar bitmask).

        The kernel must not produce NaN when most logits are -inf, which
        can happen when structured-output grammar masks are applied before
        sampling.
        """
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        # Mask a fraction of logits to -inf.
        mask = (
            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        )
        logits[mask] = float("-inf")

        k = torch.randint(
            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
        )
        result = apply_top_k_top_p_triton(logits.clone(), k, None)

        assert not result.isnan().any(), "NaN found in top-k result with -inf logits"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"
            # At least one value should survive unless the row was all -inf.
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topp_with_neginf_logits(self, inf_fraction: float):
        """Top-p with many -inf logits."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        mask = (
            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        )
        logits[mask] = float("-inf")

        p = (
            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
            + 0.1
        )
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN found in top-p result with -inf logits"
        for i in range(batch_size):
            finite_in = (logits[i] > float("-inf")).sum().item()
            kept = (result[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topk_topp_with_neginf_logits(self, inf_fraction: float):
        """Combined top-k + top-p with many -inf logits."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        mask = (
            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        )
        logits[mask] = float("-inf")

        k = torch.randint(
            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
        )
        p = (
            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
            + 0.1
        )
        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), (
            "NaN found in top-k+top-p result with -inf logits"
        )
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"

    def test_all_neginf_logits(self):
        """All logits are -inf (fully masked). Kernel should be a no-op."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 16, 128256
        logits = torch.full(
            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
        )

        k = torch.randint(
            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
        )
        p = torch.full((batch_size,), 0.9, dtype=torch.float32)

        # top-k only
        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert not result.isnan().any(), "NaN from all-inf top-k"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        # top-p only
        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert not result.isnan().any(), "NaN from all-inf top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        # top-k + top-p
        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN from all-inf top-k+top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

    def test_few_valid_tokens_with_neginf(self):
        """Only a handful of tokens are finite per row (strict grammar)."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.full(
            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
        )
        # Allow only 5 random tokens per row to be finite.
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:5]
            logits[i, indices] = torch.randn(
                5, generator=self.generator, dtype=torch.float32
            )

        k = torch.full((batch_size,), 50, dtype=torch.int32)
        p = torch.full((batch_size,), 0.9, dtype=torch.float32)

        # top-k only (k=50 but only 5 finite → keep all 5)
        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept == 5, f"Row {i}: expected 5 kept, got {kept}"

        # top-k with k < num_finite
        k_small = torch.full((batch_size,), 3, dtype=torch.int32)
        result = apply_top_k_top_p_triton(logits.clone(), k_small, None)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= 3, f"Row {i}: expected <=3 kept, got {kept}"

        # top-p only
        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, f"Row {i}: no tokens kept"

    @pytest.mark.parametrize("num_valid", [1, 2, 5, 10, 50])
    @pytest.mark.parametrize(
        "mode",
        ["topk_only", "topp_only", "topk_and_topp"],
    )
    def test_equal_logits_few_valid(self, num_valid: int, mode: str):
        """Few valid tokens all sharing the same logit value.

        This is the pattern produced by grammar bitmask filtering when
        the model assigns similar scores to the few allowed tokens.
        The ternary search can converge to a pivot equal to max_logit,
        causing the strict `>` keep_mask to exclude everything.
        Regression test for the `final_pivot >= max_logit` guard.
        """
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.full(
            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
        )
        # Set exactly `num_valid` tokens per row to the SAME finite value.
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
            logits[i, indices] = 1.0  # all equal

        k: torch.Tensor | None = None
        p: torch.Tensor | None = None
        if mode in ("topk_only", "topk_and_topp"):
            k = torch.full((batch_size,), max(1, num_valid - 1), dtype=torch.int32)
        if mode in ("topp_only", "topk_and_topp"):
            p = torch.full((batch_size,), 0.95, dtype=torch.float32)

        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), "NaN in equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            # The key invariant: at least one token must survive.
            # With all-equal logits the pivot search can't differentiate
            # tokens, so the guard may keep more than k — that is the
            # intended safe fallback.
            assert kept > 0, (
                f"Row {i}: all tokens masked with {num_valid} equal-valued "
                f"finite logits ({mode})"
            )

    @pytest.mark.parametrize("num_valid", [2, 5, 10])
    def test_nearly_equal_logits_topp(self, num_valid: int):
        """Few valid tokens with very similar (but not identical) logits.

        Ensures the kernel handles near-degenerate probability
        distributions where the ternary search range collapses.
        """
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 128256
        logits = torch.full(
            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
        )
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
            # Tiny spread: values in [1.0, 1.0 + 1e-6]
            logits[i, indices] = (
                1.0
                + torch.rand(num_valid, generator=self.generator, dtype=torch.float32)
                * 1e-6
            )

        p = torch.full((batch_size,), 0.95, dtype=torch.float32)
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN in nearly-equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, (
                f"Row {i}: all tokens masked with {num_valid} "
                f"nearly-equal finite logits"
            )

    def test_mixed_neginf_and_normal_rows(self):
        """Batch with a mix of normal rows and heavily-masked rows."""
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        batch_size, vocab_size = 32, 32000
        logits = torch.randn(
            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
        )
        # Mask even rows heavily (99% -inf), leave odd rows normal.
        for i in range(0, batch_size, 2):
            mask = torch.rand(vocab_size, generator=self.generator) < 0.99
            logits[i][mask] = float("-inf")

        k = torch.randint(
            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
        )
        p = (
            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
            + 0.1
        )

        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN in mixed normal/-inf batch"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item()
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept"
