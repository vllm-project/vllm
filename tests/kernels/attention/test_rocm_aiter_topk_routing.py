"""Tests for the gfx950 decode Top-K kernel routing policy.

Tests the two-dimensional (rows, live_compressed_ctx) routing policy that
replaced the one-dimensional max_valid_seq_len threshold in
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py.

Run inside the container:
    python3 -m pytest pr/b2-topk-routing/test_rocm_aiter_topk_routing.py -v

or standalone:
    python3 pr/b2-topk-routing/test_rocm_aiter_topk_routing.py
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip if not on gfx950 (cdna3+). Mirror the marker used in PR #56638.
# ---------------------------------------------------------------------------
try:
    _is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    if _is_rocm:
        _arch = torch.cuda.get_device_properties(0).gcnArchName
        _is_cdna3_or_newer = "gfx950" in _arch or "gfx942" in _arch
    else:
        _is_cdna3_or_newer = False
except Exception:
    _is_cdna3_or_newer = False

requires_rocm_cdna3_or_newer = pytest.mark.skipif(
    not _is_cdna3_or_newer,
    reason="Requires ROCm on cdna3+ (gfx942/gfx950)",
)

# ---------------------------------------------------------------------------
# Import the policy (from the patched file in the container, or standalone).
# ---------------------------------------------------------------------------
try:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # type: ignore
        _GFX950_TOPK_AITER_POLICY,
        _get_aiter_top_k_kernel,
    )
    _IMPORT_OK = True
except ImportError:
    # Fall back to the local copy so tests can be run from this directory.
    from routing_change import (  # type: ignore
        _GFX950_TOPK_AITER_POLICY,
        _get_aiter_top_k_kernel,
    )
    _IMPORT_OK = True


# ---------------------------------------------------------------------------
# 1. Policy unit tests — no GPU required
# ---------------------------------------------------------------------------

class TestRoutingPolicy:
    """Verify the policy table logic without touching the GPU."""

    # Full calibration grid (rows, live_ctx, expected_kernel).
    # Source: eval/eval_b2_decode_topk_policy.py on gfx950 MI350X,
    #         image nightly-eed1f3d0c6, 20/20 correct decisions.
    CALIBRATION_GRID = [
        # rows   live_ctx   expected
        (  48,   32_768, "aiter"),
        (  48,   65_536, "aiter"),
        (  48,  131_072, "native"),
        (  48,  262_144, "native"),
        (  48,  524_288, "native"),
        (  96,   32_768, "aiter"),
        (  96,   65_536, "aiter"),
        (  96,  131_072, "aiter"),
        (  96,  262_144, "native"),
        (  96,  524_288, "native"),
        ( 192,   32_768, "aiter"),
        ( 192,   65_536, "aiter"),
        ( 192,  131_072, "aiter"),
        ( 192,  262_144, "aiter"),   # C32 AgentX operating point
        ( 192,  524_288, "native"),
        ( 384,   32_768, "aiter"),
        ( 384,   65_536, "aiter"),
        ( 384,  131_072, "aiter"),
        ( 384,  262_144, "aiter"),
        ( 384,  524_288, "native"),
    ]

    @pytest.mark.parametrize("rows,live_ctx,expected", CALIBRATION_GRID)
    def test_policy_matches_calibration(self, rows, live_ctx, expected):
        """Policy must match the measured faster kernel for every calibration shape."""
        kernel = _get_aiter_top_k_kernel(
            num_rows=rows,
            logits_width=524_288,
            live_compressed_ctx=live_ctx,
        )
        got = "native" if kernel is None else "aiter"
        assert got == expected, (
            f"rows={rows}, live_ctx={live_ctx}: expected {expected}, got {got}"
        )

    def test_full_graph_always_native(self):
        """None live_compressed_ctx (FULL-graph replay) must always return native."""
        for rows in (48, 96, 192, 384):
            kernel = _get_aiter_top_k_kernel(
                num_rows=rows,
                logits_width=524_288,
                live_compressed_ctx=None,
            )
            assert kernel is None, (
                f"rows={rows}: expected native (None) for FULL graph, "
                f"got {kernel}"
            )

    def test_c32_agentx_operating_point_uses_aiter(self):
        """C32 AgentX: 192 rows, context up to 262k must route AITER."""
        for ctx in (32_768, 65_536, 131_072, 262_144):
            kernel = _get_aiter_top_k_kernel(
                num_rows=192,
                logits_width=524_288,
                live_compressed_ctx=ctx,
            )
            assert kernel is not None, (
                f"C32 with live_ctx={ctx}: expected AITER, got native"
            )

    def test_full_context_always_native(self):
        """Full 524288-token context must always route native at all row counts."""
        for rows in (48, 96, 192, 384, 512):
            kernel = _get_aiter_top_k_kernel(
                num_rows=rows,
                logits_width=524_288,
                live_compressed_ctx=524_288,
            )
            assert kernel is None, (
                f"rows={rows}, full context: expected native, got AITER"
            )

    def test_policy_table_has_correct_row_count(self):
        """Policy table must have exactly 3 entries (one per concurrency tier)."""
        assert len(_GFX950_TOPK_AITER_POLICY) == 3

    def test_policy_table_rows_are_descending(self):
        """Policy table must be sorted descending by min_rows for first-match logic."""
        rows = [r for r, _ in _GFX950_TOPK_AITER_POLICY]
        assert rows == sorted(rows, reverse=True), (
            "Policy table rows must be in descending order so the most "
            "specific (largest batch) rule matches first."
        )

    def test_logits_width_does_not_affect_routing(self):
        """logits_width is not part of the routing decision (it's the capture-time buffer)."""
        for logits_w in (131_072, 262_144, 524_288, 1_048_576):
            k1 = _get_aiter_top_k_kernel(192, logits_w, 131_072)
            assert k1 is not None, f"logits_width={logits_w} should not force native"


# ---------------------------------------------------------------------------
# 2. Correctness tests — require GPU
# ---------------------------------------------------------------------------

def _run_topk(fn, q_score: torch.Tensor, topk: int) -> torch.Tensor:
    """Invoke a top_k_per_row kernel or the torch reference and return indices."""
    if fn is None:
        # Native path: torch.topk per row
        _, idx = torch.topk(q_score, topk, dim=-1, sorted=False)
        return idx.sort(dim=-1).values
    else:
        # AITER kernel path — call signature matches aiter top_k_per_row_decode
        out = torch.empty(
            (q_score.shape[0], topk), dtype=torch.int32, device=q_score.device
        )
        fn(q_score, out, topk)
        return out.sort(dim=-1).values


# Shapes: (rows, live_ctx, logits_width, topk)
_CORRECTNESS_SHAPES = [
    (  48,  32_768, 524_288,  512),   # C8 short context
    (  96, 131_072, 524_288,  512),   # C16 medium context
    ( 192, 131_072, 524_288,  512),   # C32 AgentX typical
    ( 192, 262_144, 524_288,  512),   # C32 AgentX max AITER context
    ( 384, 262_144, 524_288, 2048),   # C64 medium context
]


@requires_rocm_cdna3_or_newer
@pytest.mark.parametrize("rows,live_ctx,logits_w,topk", _CORRECTNESS_SHAPES)
def test_aiter_and_native_index_sets_match(rows, live_ctx, logits_w, topk):
    """AITER and native must produce bitwise-identical sorted Top-K indices.

    This is the key correctness requirement: switching kernels must not change
    which blocks are selected, only how fast they are found.
    """
    torch.manual_seed(42)
    # Random logit scores in [0, 1); shape [rows, logits_w]
    q_score = torch.rand((rows, logits_w), dtype=torch.float32, device="cuda")

    native_idx = _run_topk(None, q_score, topk)

    aiter_kernel = _get_aiter_top_k_kernel(rows, logits_w, live_ctx)
    if aiter_kernel is None:
        pytest.skip(f"Policy routes native for rows={rows}, live_ctx={live_ctx}")

    aiter_idx = _run_topk(aiter_kernel, q_score, topk)

    assert torch.equal(native_idx, aiter_idx), (
        f"Index mismatch at rows={rows}, live_ctx={live_ctx}, topk={topk}. "
        f"Max diff index: {(native_idx != aiter_idx).nonzero()[:5]}"
    )


@requires_rocm_cdna3_or_newer
def test_aiter_kernel_available_on_gfx950():
    """AITER top_k_per_row_decode must be importable on gfx950."""
    kernel = _get_aiter_top_k_kernel(
        num_rows=192,
        logits_width=524_288,
        live_compressed_ctx=131_072,
    )
    assert kernel is not None, (
        "AITER top_k_per_row_decode not available — "
        "check that module_top_k_per_row.so is present in the aiter JIT cache"
    )


@requires_rocm_cdna3_or_newer
def test_native_path_used_at_full_context_on_gpu():
    """At 524288 live_ctx the policy returns None and native torch.topk produces correct results."""
    torch.manual_seed(0)
    rows, logits_w, topk = 192, 524_288, 512
    q_score = torch.rand((rows, logits_w), dtype=torch.float32, device="cuda")

    kernel = _get_aiter_top_k_kernel(rows, logits_w, live_compressed_ctx=524_288)
    assert kernel is None, "Expected native (None) at full 524288 context"

    _, idx = torch.topk(q_score, topk, dim=-1, sorted=False)
    assert idx.shape == (rows, topk)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== Policy unit tests (no GPU required) ===")
    suite = TestRoutingPolicy()
    passed = failed = 0

    for rows, live_ctx, expected in TestRoutingPolicy.CALIBRATION_GRID:
        try:
            suite.test_policy_matches_calibration(rows, live_ctx, expected)
            print(f"  PASS  rows={rows:4d}  live_ctx={live_ctx:>8}  -> {expected}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1

    for name, fn in [
        ("full_graph_always_native", suite.test_full_graph_always_native),
        ("c32_agentx_uses_aiter",    suite.test_c32_agentx_operating_point_uses_aiter),
        ("full_context_native",      suite.test_full_context_always_native),
        ("table_row_count",          suite.test_policy_table_has_correct_row_count),
        ("table_descending",         suite.test_policy_table_rows_are_descending),
        ("logits_width_neutral",     suite.test_logits_width_does_not_affect_routing),
    ]:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")

    if _is_cdna3_or_newer:
        print("\n=== GPU correctness tests ===")
        for rows, live_ctx, logits_w, topk in _CORRECTNESS_SHAPES:
            try:
                test_aiter_and_native_index_sets_match(rows, live_ctx, logits_w, topk)
                print(f"  PASS  rows={rows}  live_ctx={live_ctx}  topk={topk}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  rows={rows}  live_ctx={live_ctx}: {e}")
                failed += 1
    else:
        print("\n(Skipping GPU tests — not on gfx950/gfx942)")

    sys.exit(1 if failed else 0)
