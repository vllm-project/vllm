#!/usr/bin/env python3
"""Fix the emulated block-scaled MMA in the FlashInfer `+sm89` 0.6.14 build.

Ada has no hardware block-scaled MMA, so the `+sm89` port replaces

    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32

with a plain FP8 MMA followed by a scale multiply. The 0.6.14 port gets two
things wrong, and both are silent -- no crash, no warning:

1. **One scale for four accumulators.** In the m16n8k32 fragment layout d0/d1
   belong to row `gid` and d2/d3 to row `gid+8`, while adjacent columns carry
   distinct B scales. The `scale_vec::1X` instruction gathers those distributed
   scale operands per accumulator; the manual substitution applied the calling
   lane's single `scale_a`/`scale_b` product to all four. Three of the four
   accumulators are scaled with the wrong factor whenever scales differ across
   the fragment. The error is invisible when every block in a tile shares a
   scale, and grows with the number of distinct KV blocks in flight -- so it
   surfaces as long-context quality decay rather than as an obvious failure.

2. **No range handling for degenerate E8M0 encodings.** `scale_exp <= 0` wraps
   through `uint32` to `0xFF800000`, i.e. `-Inf`, which becomes NaN and spreads
   across attention. `0xff` (the E8M0 NaN encoding) and exponent overflow are
   likewise unhandled.

Both are fixed upstream in `flashinfer_python-0.6.17+sm89.1`. This script
backports that file: `assets/mma_sm120.cuh` is the 0.6.17+sm89.1 revision of
`mma_sm120.cuh`, taken verbatim -- outside the two hunks above it is byte for
byte identical to the 0.6.14 one, so installing it is exactly equivalent to
applying them.

This is JIT source text shipped inside the wheel and compiled by FlashInfer at
runtime, so no build toolchain is needed. Point `FLASHINFER_CACHE_DIR` at a
fresh directory afterwards, otherwise stale cached kernels are reused and the
fix does nothing.

Usage:
    python patch-flashinfer-sm89-mma-scales.py --check   # dry run, writes nothing
    python patch-flashinfer-sm89-mma-scales.py           # apply
    python patch-flashinfer-sm89-mma-scales.py --path /path/to/site-packages
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

RELATIVE = Path("data/include/flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh")

# Every state we know how to move to FIXED, so a wrong or unexpected file is
# refused rather than silently overwritten.
PRISTINE_0614 = "8cff4db04e3dbb41839cf1329872a703"  # 0.6.14+sm89 as released
CLAMP_ONLY = "beab1e4ae9921b19b261e685887f866f"     # + the earlier clamp-only fix
FIXED = "594b3362ffdd9ad4e15e92e836a2bf5f"          # 0.6.17+sm89.1 revision

ACCEPTED = {PRISTINE_0614: "0.6.14+sm89 (as released)",
            CLAMP_ONLY: "0.6.14+sm89 + clamp-only fix (superseded)"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def locate(explicit: str | None) -> Path:
    if explicit:
        pkg = Path(explicit)
        return pkg if pkg.name == "flashinfer" else pkg / "flashinfer"
    try:
        import flashinfer  # noqa: PLC0415
    except ImportError:
        sys.exit("flashinfer is not importable; install the wheel first or pass --path")
    return Path(flashinfer.__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--path", help="site-packages dir, or the flashinfer package dir")
    args = ap.parse_args()

    source = Path(__file__).with_name("mma_sm120.cuh")
    if not source.is_file():
        sys.exit(f"missing vendored replacement: {source}")
    if md5(source) != FIXED:
        sys.exit(f"vendored replacement has unexpected md5 {md5(source)}; refusing to install")

    target = locate(args.path) / RELATIVE
    if not target.is_file():
        sys.exit(f"not found: {target}")

    current = md5(target)
    if current == FIXED:
        print(f"already fixed: {target}")
        return 0
    if current not in ACCEPTED:
        sys.exit(
            f"{target}\n  has md5 {current}, which is not a state this script knows.\n"
            "  Expected one of:\n"
            + "".join(f"    {h}  {w}\n" for h, w in ACCEPTED.items())
            + "  Refusing to overwrite an unrecognised file."
        )

    print(f"found: {ACCEPTED[current]}")
    if args.check:
        print(f"CHECK OK: would install the 0.6.17+sm89.1 revision into {target}")
        return 0

    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copyfile(source, target)

    after = md5(target)
    if after != FIXED:
        sys.exit(f"post-check failed: {target} is {after}, expected {FIXED}")

    print(f"patched: {target}")
    print(f"backup:  {backup}")
    print("verified: md5 matches the 0.6.17+sm89.1 revision")
    print("now point FLASHINFER_CACHE_DIR at a fresh directory before restarting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
