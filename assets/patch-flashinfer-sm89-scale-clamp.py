#!/usr/bin/env python3
"""Clamp the emulated block-scale factor in the FlashInfer `+sm89` build.

Ada has no hardware block-scaled MMA. The `+sm89` port replaces

    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32

with "plain FP8 MMA, then multiply by the scale". The exponent arithmetic is
correct (E8M0 bias 127, so two scales multiply to 2^(e_a + e_b - 254)), but the
substitution drops the range clamp the hardware instruction applies to
degenerate encodings:

    scale_exp == -1  ->  (uint32)(-1) << 23  ==  0xFF800000  ==  -Inf
    scale_exp >= 255 ->  exponent all ones   ==  Inf / NaN

`scale_exp <= 0` holds whenever `e_a + e_b < 128`, which every all-zero or
padded block hits (E8M0 encodes such a scale as 0), and zero blocks are
everywhere on this path: padding slots, and KV slots not chosen by top-k (the
kernel dequantises padded slots to 0 by design). One -Inf in the accumulator
turns into NaN and spreads across attention, so output is garbage from the
first token -- without a crash and without a warning.

This is JIT source text shipped inside the wheel; FlashInfer compiles it at
runtime. Point `FLASHINFER_CACHE_DIR` at a fresh directory after patching so
stale cached kernels are not reused. No build toolchain is required.

Usage:
    python patch-flashinfer-sm89-scale-clamp.py --check   # dry run, no writes
    python patch-flashinfer-sm89-scale-clamp.py           # apply
    python patch-flashinfer-sm89-scale-clamp.py --path /path/to/site-packages/flashinfer

The script is idempotent: it keys off the `[SM89_SCALE_CLAMP]` sentinel, not off
the anchor text, so re-running it is a no-op rather than a double application.
It validates before it writes -- if the anchor is missing or matches more than
once it exits non-zero and touches nothing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RELATIVE = Path(
    "data/include/flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh"
)

SENTINEL = "[SM89_SCALE_CLAMP]"

ANCHOR = """  const int scale_exp = static_cast<int>(scale_a) + static_cast<int>(scale_b) - 127;
  const float scale = __uint_as_float(static_cast<uint32_t>(scale_exp) << 23);
"""

REPLACEMENT = """  // [SM89_SCALE_CLAMP] The hardware block_scale MMA handles degenerate E8M0
  // encodings; this manual substitution must too. Without clamping,
  // scale_exp <= 0 (any zero/padded block) wraps through uint32 and
  // (uint32)(-1) << 23 == 0xFF800000 == -Inf, poisoning the accumulator.
  const int scale_exp = static_cast<int>(scale_a) + static_cast<int>(scale_b) - 127;
  const float scale =
      (scale_a == 0 || scale_b == 0 || scale_exp <= 0)
          ? 0.0f
          : __uint_as_float(static_cast<uint32_t>(scale_exp > 254 ? 254 : scale_exp) << 23);
"""


def locate_package(explicit: str | None) -> Path:
    if explicit:
        pkg = Path(explicit)
        if pkg.name != "flashinfer":
            pkg = pkg / "flashinfer"
        return pkg
    try:
        import flashinfer  # noqa: PLC0415
    except ImportError:
        sys.exit(
            "flashinfer is not importable; install the wheel first or pass --path"
        )
    return Path(flashinfer.__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--path", help="site-packages dir, or the flashinfer package dir")
    args = ap.parse_args()

    pkg = locate_package(args.path)
    target = pkg / RELATIVE

    if not target.is_file():
        sys.exit(f"not found: {target}")

    # newline="" on both sides: never translate line endings. The vendored
    # source is LF; rewriting it as CRLF would churn the whole file.
    with open(target, encoding="utf-8", newline="") as fh:
        text = fh.read()

    if SENTINEL in text:
        print(f"already patched: {target}")
        return 0

    hits = text.count(ANCHOR)
    if hits != 1:
        sys.exit(
            f"anchor matched {hits} times in {target} (expected exactly 1); "
            "the vendored build does not look like flashinfer 0.6.14+sm89 -- "
            "nothing was written"
        )

    if args.check:
        print(f"CHECK OK: 1 anchor, 0 already applied -- {target}")
        return 0

    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(target, backup)

    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(ANCHOR, REPLACEMENT))
    print(f"patched: {target}")
    print(f"backup:  {backup}")
    print("point FLASHINFER_CACHE_DIR at a fresh directory before restarting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
