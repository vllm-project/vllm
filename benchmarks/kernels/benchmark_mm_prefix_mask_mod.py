# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latency of the FA4 mm_prefix mask_mod as the number of ranges varies.

Models the Gemma4 video-pooling shape that motivated the optimization: a text
prefix followed by N video frames, each frame contributing one bidirectional
mm_prefix range. Real batches see N anywhere from 1 to 64.

Four generations of the mask_mod are compared, all producing the same mask:

  range-scan-constexpr  Upstream before #50294. ``(batch, max_ranges, 2)``
                        scanned with ``cutlass.range_constexpr(max_ranges)``,
                        so N is baked into the CuTe compile key: every new N
                        is a cold JIT of the whole FA4 forward.
  range-scan-dynamic    Same scan, but N arrives as an ``aux_scalar`` and the
                        loop is a dynamic ``range``, so N no longer
                        re-specializes the JIT. Still O(N) per score.
  range-id              #50294. ``(num_seqs, max_seq_len)`` per-token range
                        ids; membership is one equality. O(1) per score but
                        two loads that depend on both q and kv.
  query-range           vhagor/vllm#2. ``(num_actual_tokens, 2)`` absolute
                        bounds of the range holding each query token. Both
                        loads depend only on q, so they hoist out of the
                        per-element loop, and kv is never used as an index.

``triton`` runs the same mask through vLLM's Triton reference path
(``unified_attention`` with the ``(batch, max_ranges, 2)`` tensor it still
uses), which is the alternative a deployment would pick if FA4 mm_prefix were
not viable. ``none`` (plain causal + sliding window, no mask_mod) is available
via ``--variants`` as a floor but is off by default: attaching any mask_mod
makes FA4 resolve causal and local to False, so it stops pruning tiles and
walks the full Q x K grid, and that cost is common to every mask_mod variant.

Examples:
    # Steady-state latency vs frame count.
    python benchmark_mm_prefix_mask_mod.py

    # Cold-start cost, i.e. the recompile #50294 was about.
    python benchmark_mm_prefix_mask_mod.py --measure-jit

    # Hold seqlen fixed to separate mask cost from the O(L^2) growth.
    python benchmark_mm_prefix_mask_mod.py --shape fixed
"""

import argparse
import time

import numpy as np
import torch

from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    fill_mm_prefix_query_ranges,
)

# Gemma4 26B A4B: sliding layers use head_dim=256 with a 1024 window, global
# layers use global_head_dim=512 with no window. Sliding layers dominate.
DEFAULT_HEAD_SIZE = 256
DEFAULT_SLIDING_WINDOW = 1024
DEFAULT_NUM_HEADS = 16
DEFAULT_NUM_KV_HEADS = 8

# Workload from the #50294 measurement: ~885 text tokens, 280 soft tokens per
# frame, frame count between 8 and 64 with an average of 48.
DEFAULT_TEXT_LEN = 885
DEFAULT_TOKENS_PER_RANGE = 280
DEFAULT_RANGE_COUNTS = [1, 8, 16, 32, 46, 64]


# --------------------------------------------------------------------------- #
# mask_mod variants
# --------------------------------------------------------------------------- #


def make_range_scan_mask_mod(sliding_window_left, max_ranges=None):
    """Pre-#50294 scan. ``max_ranges=None`` uses the dynamic-N variant."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32

    # `vllm.vllm_flash_attn.cute` is a build artifact, so isort sorts it as
    # third-party in a clean checkout.
    # isort: split
    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa

    # `range_constexpr` is rewritten by the CuTe source preprocessor, so it has
    # to appear literally in the for-statement. Hence two bodies rather than a
    # shared one parameterized by the loop object.
    if max_ranges is not None:

        @cute.jit
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            keep = kv_idx <= q_abs
            if sliding_window_left is not None:
                sw = scalar_to_ssa(Int32(sliding_window_left), Int32)
                keep = keep & ((q_abs - kv_idx) < sw)
            ranges = aux_tensors[0]
            b = batch_idx[0]
            for i in cutlass.range_constexpr(max_ranges):
                r_start = scalar_to_ssa(ranges[b, i, 0], Int32)
                r_end = scalar_to_ssa(ranges[b, i, 1], Int32)
                valid = r_start < r_end
                q_in = (q_abs >= r_start) & (q_abs <= r_end) & valid
                k_in = (kv_idx >= r_start) & (kv_idx <= r_end) & valid
                keep = keep | (q_in & k_in)
            return keep

    else:

        @cute.jit
        def mask_mod(
            batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars
        ):
            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            keep = kv_idx <= q_abs
            if sliding_window_left is not None:
                sw = scalar_to_ssa(Int32(sliding_window_left), Int32)
                keep = keep & ((q_abs - kv_idx) < sw)
            ranges = aux_tensors[0]
            n = Int32(aux_scalars[0])
            b = batch_idx[0]
            for i in range(n):
                r_start = scalar_to_ssa(ranges[b, i, 0], Int32)
                r_end = scalar_to_ssa(ranges[b, i, 1], Int32)
                valid = r_start < r_end
                q_in = (q_abs >= r_start) & (q_abs <= r_end) & valid
                k_in = (kv_idx >= r_start) & (kv_idx <= r_end) & valid
                keep = keep | (q_in & k_in)
            return keep

    mask_mod.use_fast_sampling = True
    return mask_mod


def make_range_id_mask_mod(sliding_window_left):
    """#50294: per-token range ids, one equality per score."""
    import cutlass.cute as cute
    from cutlass import Int32

    # isort: split
    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa

    @cute.jit
    def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
        q_abs = q_idx + ctx_off
        keep = kv_idx <= q_abs
        if sliding_window_left is not None:
            sw = scalar_to_ssa(Int32(sliding_window_left), Int32)
            keep = keep & ((q_abs - kv_idx) < sw)
        range_ids = aux_tensors[0]
        b = batch_idx[0]
        q_range_id = scalar_to_ssa(range_ids[b, q_abs[0]], Int32)
        k_range_id = scalar_to_ssa(range_ids[b, kv_idx[0]], Int32)
        keep = keep | ((q_range_id >= Int32(0)) & (q_range_id == k_range_id))
        return keep

    mask_mod.use_fast_sampling = True
    return mask_mod


def make_query_range_mask_mod(sliding_window_left):
    """vhagor/vllm#2, imported from the tree so the benchmark tracks the code."""
    from vllm.v1.attention.backends.flash_attn import _make_mm_prefix_mask_mod

    return _make_mm_prefix_mask_mod(
        sliding_window=0, sliding_window_left=sliding_window_left
    )


# Mirrors `flash_attn._MM_PREFIX_MASK_MOD_CACHE`: production builds one mask_mod
# per sliding-window value and reuses it for every batch. Rebuilding per call
# would misreport compile cost, because the CuTe compile key is a hash of the
# callable's source plus `repr()` of each closure cell, and some variants close
# over freshly created function objects whose repr embeds an address.
_MASK_MOD_CACHE: dict = {}


def get_mask_mod(variant, sliding_window_left, num_ranges):
    # range-scan-constexpr genuinely cannot be cached across N: the frame count
    # is a compile-time constant of the kernel. That is the point of #50294.
    key = (
        variant,
        sliding_window_left,
        num_ranges if variant == "range-scan-constexpr" else None,
    )
    if key not in _MASK_MOD_CACHE:
        if variant == "range-scan-constexpr":
            mm = make_range_scan_mask_mod(sliding_window_left, max_ranges=num_ranges)
        elif variant == "range-scan-dynamic":
            mm = make_range_scan_mask_mod(sliding_window_left)
        elif variant == "range-id":
            mm = make_range_id_mask_mod(sliding_window_left)
        elif variant == "query-range":
            mm = make_query_range_mask_mod(sliding_window_left)
        else:
            raise ValueError(variant)
        _MASK_MOD_CACHE[key] = mm
    return _MASK_MOD_CACHE[key]


# --------------------------------------------------------------------------- #
# aux tensor construction
# --------------------------------------------------------------------------- #


def build_ranges(num_ranges, text_len, tokens_per_range, batch_size):
    """One range per video frame, laid out after the text prefix."""
    per_req = [
        (text_len + i * tokens_per_range, text_len + (i + 1) * tokens_per_range - 1)
        for i in range(num_ranges)
    ]
    return {b: list(per_req) for b in range(batch_size)}


def range_id_tensor(mm_ranges, num_seqs, max_seq_len, device):
    ids = torch.full((num_seqs, max_seq_len), -1, dtype=torch.int32)
    for seq_idx in range(num_seqs):
        for rid, (start, end) in enumerate(sorted(mm_ranges.get(seq_idx, []))):
            ids[seq_idx, start : end + 1] = rid
    return ids.to(device)


def make_aux(variant, mm_ranges, seq_lens, cu_seqlens, device):
    """Return (aux_tensors, aux_scalars) for a variant."""
    num_seqs = len(seq_lens)
    if variant.startswith("range-scan"):
        ranges = compute_mm_prefix_range_tensor(mm_ranges, num_seqs, device)
        n = int(ranges.shape[1])
        if variant.endswith("dynamic"):
            from cutlass import Int32

            return [ranges], (Int32(n),)
        return [ranges], None
    if variant == "range-id":
        return [range_id_tensor(mm_ranges, num_seqs, max(seq_lens), device)], None
    if variant == "query-range":
        total = int(cu_seqlens[-1])
        staging = np.empty((total, 2), dtype=np.int32)
        rows = fill_mm_prefix_query_ranges(
            staging,
            mm_ranges,
            cu_seqlens,
            torch.tensor(seq_lens, dtype=torch.int32),
        )
        assert rows > 0
        q_ranges = torch.from_numpy(staging[:rows]).to(device)
        return [q_ranges, cu_seqlens.to(device)], None
    raise ValueError(variant)


def build_paged_kv(k, v, seq_lens, block_size, num_kv_heads, head_size, device):
    """Pack varlen k/v into the paged layout `unified_attention` expects.

    Cache is ``(num_blocks, block_size, num_kv_heads, head_size)``; each
    sequence gets a contiguous run of blocks, which is the best case for the
    Triton path and keeps the comparison about the mask, not the page walk.
    """
    blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = blocks_per_seq * len(seq_lens)
    k_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size, dtype=k.dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        len(seq_lens), blocks_per_seq
    )

    off = 0
    for i, s_len in enumerate(seq_lens):
        flat_k = k_cache[i * blocks_per_seq : (i + 1) * blocks_per_seq].view(
            -1, num_kv_heads, head_size
        )
        flat_v = v_cache[i * blocks_per_seq : (i + 1) * blocks_per_seq].view(
            -1, num_kv_heads, head_size
        )
        flat_k[:s_len] = k[off : off + s_len]
        flat_v[:s_len] = v[off : off + s_len]
        off += s_len
    return k_cache, v_cache, block_table


def make_block_sparse(
    mm_ranges, seq_lens, sliding_window, head_size, num_heads, device
):
    from vllm.v1.attention.backends.flash_attn import _fa4_sm90_tile_size
    from vllm.v1.attention.backends.utils import (
        compute_mm_prefix_block_sparse_tensors,
    )

    tiles = _fa4_sm90_tile_size(head_size)
    if tiles is None:
        return None
    return compute_mm_prefix_block_sparse_tensors(
        mm_ranges,
        np.array(seq_lens),
        sliding_window,
        tiles[0],
        tiles[1],
        num_heads,
        device,
    )


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def time_kernel(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "p10_ms": samples[max(0, int(0.1 * len(samples)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    }


def build_call(args, variant, num_ranges, device):
    """Materialize tensors and return a zero-arg callable running one forward."""
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    if args.shape == "realistic":
        tokens_per_range = args.tokens_per_range
        seq_len = args.text_len + num_ranges * tokens_per_range
    else:
        # Hold seqlen at the largest realistic value and shrink each range so
        # total mm coverage stays constant; only N changes.
        seq_len = args.text_len + max(args.range_counts) * args.tokens_per_range
        tokens_per_range = (seq_len - args.text_len) // max(num_ranges, 1)

    seq_lens = [seq_len] * args.batch_size
    cu = torch.zeros(args.batch_size + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    total = int(cu[-1])

    torch.manual_seed(0)
    q = torch.randn(
        total, args.num_heads, args.head_size, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        total, args.num_kv_heads, args.head_size, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        total, args.num_kv_heads, args.head_size, dtype=torch.bfloat16, device=device
    )
    out = torch.empty_like(q)
    cu_gpu = cu.to(device)
    scale = args.head_size**-0.5
    sw = args.sliding_window
    window = [sw - 1, 0] if sw > 0 else None
    sw_left = sw if sw > 0 else None

    if variant == "triton":
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        mm_ranges = build_ranges(
            num_ranges, args.text_len, tokens_per_range, args.batch_size
        )
        k_cache, v_cache, block_table = build_paged_kv(
            k, v, seq_lens, 16, args.num_kv_heads, args.head_size, device
        )
        mm_prefix_range = compute_mm_prefix_range_tensor(
            mm_ranges, args.batch_size, device
        )
        seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        tri_window = [sw - 1, 0] if sw > 0 else [-1, -1]

        def call():
            unified_attention(
                q=q,
                k=k_cache,
                v=v_cache,
                out=out,
                cu_seqlens_q=cu_gpu,
                max_seqlen_q=seq_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                window_size=tri_window,
                block_table=block_table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                mm_prefix_range=mm_prefix_range,
            )

        return call, seq_len, out

    if variant == "none":
        mask_mod, aux, aux_scalars, bs = None, None, None, None
    else:
        mm_ranges = build_ranges(
            num_ranges, args.text_len, tokens_per_range, args.batch_size
        )
        mask_mod = get_mask_mod(variant, sw_left, num_ranges)
        aux, aux_scalars = make_aux(variant, mm_ranges, seq_lens, cu, device)
        bs = (
            make_block_sparse(
                mm_ranges, seq_lens, sw, args.head_size, args.num_heads, device
            )
            if args.block_sparse
            else None
        )

    def call():
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_gpu,
            cu_seqlens_k=cu_gpu,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            fa_version=4,
            mask_mod=mask_mod,
            aux_tensors=aux,
            aux_scalars=aux_scalars,
            block_sparse_tensors=bs,
        )

    return call, seq_len, out


def check_equivalence(args, device, num_ranges):
    """All mm_prefix variants must produce the same mask before timing them.

    Without this the table is just four different kernels, not four
    implementations of one mask.
    """

    variants = [v for v in args.variants if v != "none"]
    if len(variants) < 2:
        return
    outs = {}
    for variant in variants:
        if variant == "triton":
            continue  # different kernel; compared with a tolerance below
        call, _, out = build_call(args, variant, num_ranges, device)
        call()
        torch.accelerator.synchronize()
        outs[variant] = out.clone()
        del call, out
        torch.accelerator.empty_cache()

    ref_name, ref = next(iter(outs.items()))
    for name, out in outs.items():
        if name == ref_name:
            continue
        if not torch.equal(ref, out):
            delta = (ref.float() - out.float()).abs().max().item()
            raise SystemExit(
                f"variant '{name}' does not match '{ref_name}' at N={num_ranges} "
                f"(max abs diff {delta}); latency comparison would be meaningless"
            )
    print(f"equivalence check at N={num_ranges}: {', '.join(outs)} all bit-identical")

    if "triton" in args.variants:
        # A different kernel entirely, so only the mask semantics can be
        # checked, not the bits. Loose enough for bf16 accumulation order,
        # tight enough that a wrong mask shows up.
        call, _, tri_out = build_call(args, "triton", num_ranges, device)
        call()
        torch.accelerator.synchronize()
        delta = (ref.float() - tri_out.float()).abs().max().item()
        del call, tri_out
        torch.accelerator.empty_cache()
        if delta > 0.05:
            raise SystemExit(
                f"triton output diverges from {ref_name} at N={num_ranges} "
                f"(max abs diff {delta}); mask semantics differ"
            )
        print(
            f"  triton vs {ref_name}: max abs diff {delta:.4f} (same mask, "
            f"different kernel)"
        )


def run_steady_state(args, device):
    print(f"\nSteady-state latency (median of {args.iters}, ms)")
    print(
        f"shape={args.shape} batch={args.batch_size} head_size={args.head_size} "
        f"heads={args.num_heads}/{args.num_kv_heads} "
        f"sliding_window={args.sliding_window} block_sparse={args.block_sparse}"
    )

    header = f"{'variant':<22}" + "".join(f"{f'N={n}':>12}" for n in args.range_counts)
    print(header)
    print("-" * len(header))

    results: dict[str, dict[int, float]] = {}
    for variant in args.variants:
        results[variant] = {}
        cells = []
        for n in args.range_counts:
            call, _, _ = build_call(args, variant, n, device)
            stats = time_kernel(call, args.warmup, args.iters)
            results[variant][n] = stats["median_ms"]
            cells.append(stats["median_ms"])
            del call
            torch.accelerator.empty_cache()
        print(f"{variant:<22}" + "".join(f"{c:>12.3f}" for c in cells))

    if "query-range" in results and len(results) > 1:
        print("\nRatio vs query-range (>1 = query-range is faster)")
        print(header)
        print("-" * len(header))
        base = results["query-range"]
        for variant, by_n in results.items():
            if variant == "query-range":
                continue
            print(
                f"{variant:<22}"
                + "".join(f"{by_n[n] / base[n]:>11.2f}x" for n in args.range_counts)
            )


def run_jit(args, device):
    """Wall time of the first call at each N, i.e. what a live server pays.

    The scan-constexpr variant bakes N into the compile key, so every new frame
    count is a fresh CuTe compile of the whole FA4 forward with the GPU idle.
    The others compile once and reuse.
    """
    print("\nCold-start wall time of the first call at each N (seconds)")
    header = f"{'variant':<22}" + "".join(f"{f'N={n}':>12}" for n in args.range_counts)
    print(header)
    print("-" * len(header))

    for variant in args.variants:
        # Fresh process-level cache state is not reachable here, so warm the
        # variant on N=range_counts[0] first and report the *incremental* cost
        # of each subsequent N.
        cells = []
        for n in args.range_counts:
            call, _, _ = build_call(args, variant, n, device)
            torch.accelerator.synchronize()
            t0 = time.perf_counter()
            call()
            torch.accelerator.synchronize()
            cells.append(time.perf_counter() - t0)
            del call
            torch.accelerator.empty_cache()
        print(f"{variant:<22}" + "".join(f"{c:>12.2f}" for c in cells))
    print(
        "\nNote: run with a single --variants entry per process for a clean "
        "reading; CuTe caches compiled kernels for the process lifetime."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=[
            "range-scan-constexpr",
            "range-scan-dynamic",
            "range-id",
            "query-range",
            "triton",
        ],
        help="'none' (no mask_mod at all) is also accepted as a floor",
    )
    ap.add_argument("--range-counts", type=int, nargs="+", default=DEFAULT_RANGE_COUNTS)
    ap.add_argument(
        "--shape",
        choices=["realistic", "fixed"],
        default="realistic",
        help="realistic: seqlen grows with N (the actual workload). "
        "fixed: seqlen pinned at the largest N so only the "
        "mask cost varies.",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--text-len", type=int, default=DEFAULT_TEXT_LEN)
    ap.add_argument("--tokens-per-range", type=int, default=DEFAULT_TOKENS_PER_RANGE)
    ap.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE)
    ap.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    ap.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    ap.add_argument(
        "--sliding-window",
        type=int,
        default=DEFAULT_SLIDING_WINDOW,
        help="0 for a global-attention layer",
    )
    ap.add_argument(
        "--block-sparse",
        action="store_true",
        help="also pass the block-sparse K-block lists (fork-only)",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--measure-jit", action="store_true")
    ap.add_argument(
        "--skip-check",
        action="store_true",
        help="skip the cross-variant bit-equality gate",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

    if not is_fa_version_supported(4):
        raise SystemExit("FA4 not supported on this device")

    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}")
    if args.measure_jit:
        run_jit(args, device)
    else:
        if not args.skip_check:
            check_equivalence(args, device, args.range_counts[-1])
        run_steady_state(args, device)


if __name__ == "__main__":
    main()
