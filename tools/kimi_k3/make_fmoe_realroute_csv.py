#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the merged aiter fused-MoE tuning table with the Kimi-K3.1 real-route rows.

Interim fallback for the stock aiter experts path (the amd_kernels `moe` route does not
read it).

aiter's `AITER_CONFIG_FMOE` REPLACES the shipped tables: when it is set, aiter reads
only the files it names and no longer globs `configs/model_configs/*tuned_fmoe*.csv`.
This script therefore writes (a) `kimik3_a4w4_tuned_fmoe.csv` with
`third_party/aiter_kimik3_a4w4_fmoe_realroute.diff` applied and (b) one merged CSV =
configs/tuned_fmoe.csv + every other model_configs/*tuned_fmoe*.csv + (a), with the same
column handling as aiter's own merge, under `<tmpdir>/aiter_fmoe_realroute/` (or
`--out`; never aiter's shared /tmp/aiter_configs/tuned_fmoe.csv). It prints the path to
export:

    export AITER_CONFIG_FMOE=$(python tools/kimi_k3/make_fmoe_realroute_csv.py \
        --patch /path/to/aiter_kimik3_a4w4_fmoe_realroute.diff)

Rows changed by the patch: token buckets 16 and 128 of the TP8 shape (3584 / 384 / 896
experts / top-16, Situv2 a4w4): block_m 16 inline-quant stage1 + nt stage2 (-3..-9% at
M 12 and 65-128). Re-run after upgrading aiter.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

PATCHED_NAME = "kimik3_a4w4_tuned_fmoe.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--patch",
        required=True,
        help="aiter_kimik3_a4w4_fmoe_realroute.diff (applied to the shipped table)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(tempfile.gettempdir(), "aiter_fmoe_realroute"),
        help="output directory",
    )
    args = ap.parse_args()
    patch, out_dir = os.path.abspath(args.patch), args.out
    import aiter.jit.core as core

    root = core.AITER_ROOT_DIR
    cfg = os.path.join(root, "aiter", "configs")
    os.makedirs(out_dir, exist_ok=True)
    patched = os.path.join(out_dir, PATCHED_NAME)
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "aiter", "configs", "model_configs")
        os.makedirs(d)
        shutil.copy(os.path.join(cfg, "model_configs", PATCHED_NAME), d)
        subprocess.run(["patch", "-s", "-p1", "-i", patch], cwd=tmp, check=True)
        shutil.copy(os.path.join(d, PATCHED_NAME), patched)
    others = sorted(
        p
        for p in glob.glob(os.path.join(cfg, "model_configs", "*tuned_fmoe*.csv"))
        if "untuned" not in os.path.basename(p) and os.path.basename(p) != PATCHED_NAME
    )
    files = [os.path.join(cfg, "tuned_fmoe.csv"), *others, patched]
    # Same merge as aiter's AITER_CONFIG.update_config_files (text cells, union of
    # columns, missing columns filled with defaults), written to OUR directory: aiter's
    # own routine always writes /tmp/aiter_configs/tuned_fmoe.csv, which every aiter
    # process shares.
    import pandas as pd

    dfs = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
    cols = list(dfs[0].columns)
    for df in dfs[1:]:
        for c in df.columns:
            if c not in cols:
                before = "tflops" if "tflops" in cols else cols[-1]
                cols.insert(cols.index(before), c)
    fill = {"xbf16": 0, "run_1stage": 0, "ksplit": 0}
    aligned = []
    for f, df in zip(files, dfs):
        for c in cols:
            if c not in df.columns:
                if c == "gfx" and "cu_num" in df.columns:
                    from aiter.jit.utils.chip_info import gfx_from_cu_num

                    df[c] = pd.to_numeric(df["cu_num"], errors="coerce").map(
                        gfx_from_cu_num
                    )
                else:
                    df[c] = str(fill.get(c, 0))
        aligned.append(df[cols])
    merged = pd.concat([d for d in aligned if not d.empty], ignore_index=True)
    # (aiter checks duplicate shape keys itself when it reads the file named by
    # AITER_CONFIG_FMOE)
    out = os.path.join(out_dir, "tuned_fmoe_merged_realroute.csv")
    merged.to_csv(out, index=False)
    print(f"merged {len(files)} tables, {len(merged)} rows -> {out}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
