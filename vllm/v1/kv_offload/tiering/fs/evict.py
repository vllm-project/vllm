# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep a shared FS-tier ``root_dir`` under a byte budget.

Run one instance per ``root_dir``, alongside any number of vLLM engines:

    python -m vllm.v1.kv_offload.tiering.fs.evict /mnt/kv --max-bytes 500G

Blocks are deleted oldest-first by mtime. The FS tier never rewrites an
existing block, so mtime is its first-store time and eviction is FIFO. Engines
need no coordination: a block deleted after lookup fails to load and is
recomputed.
"""

import argparse
import os
import time

_UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(value: str) -> int:
    unit = _UNITS.get(value[-1:].upper())
    return int(float(value[:-1]) * unit) if unit else int(value)


def evict(
    root_dir: str,
    max_bytes: int,
    low_watermark: float = 0.9,
    tmp_grace_s: float = 3600.0,
) -> tuple[int, int]:
    """Run one eviction pass over ``root_dir``.

    Args:
        root_dir: The FS tier's ``root_dir``.
        max_bytes: Byte budget for all ``.bin`` block files under ``root_dir``.
        low_watermark: Once over budget, evict down to this fraction of it.
        tmp_grace_s: Delete ``.tmp`` files older than this, left by writers
            that crashed mid-store.

    Returns:
        (bytes_used_after, files_deleted).

    """
    now = time.time()
    blocks = []
    deleted = 0
    # ponytail: full tree scan per pass, O(blocks) time and memory. Fine for
    # millions of blocks on an interval; past that, track usage incrementally.
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            if name.endswith(".bin"):
                blocks.append((st.st_mtime, st.st_size, path))
            elif name.endswith(".tmp") and now - st.st_mtime > tmp_grace_s:
                deleted += _unlink(path)

    used = sum(size for _, size, _ in blocks)
    if used <= max_bytes:
        return used, deleted
    target = max_bytes * low_watermark
    blocks.sort()
    for _, size, path in blocks:
        if used <= target:
            break
        # Missing means another evictor already removed it; count it freed.
        deleted += _unlink(path)
        used -= size
    return used, deleted


def _unlink(path: str) -> bool:
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root_dir")
    parser.add_argument("--max-bytes", type=parse_size, required=True, help="e.g. 500G")
    parser.add_argument("--low-watermark", type=float, default=0.9)
    parser.add_argument("--tmp-grace-s", type=float, default=3600.0)
    parser.add_argument(
        "--interval-s",
        type=float,
        default=60.0,
        help="Seconds between passes; 0 runs a single pass.",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.root_dir):
        parser.error(f"not a directory: {args.root_dir}")
    if not 0 < args.low_watermark <= 1:
        parser.error("--low-watermark must be in (0, 1]")

    while True:
        used, deleted = evict(
            args.root_dir, args.max_bytes, args.low_watermark, args.tmp_grace_s
        )
        print(f"used={used} limit={args.max_bytes} deleted={deleted}", flush=True)
        if args.interval_s <= 0:
            return
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
