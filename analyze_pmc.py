"""Aggregate rocprofv3 PMC counter_collection.csv files per kernel.

Usage:  python3 analyze_pmc.py <pmc_root_dir>

Walks every counter_collection.csv under the root, groups by Kernel_Name +
Grid_Size (so HS=64 vs HS=128 stay separate when grids differ), and prints
mean / min / max for each counter alongside VGPR/LDS/SGPR/Workgroup.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
data = defaultdict(lambda: defaultdict(list))  # (kernel, grid) -> counter -> [values]
shapes = {}  # (kernel, grid) -> {VGPR, LDS, SGPR, Workgroup, AccumVGPR}

for csv_path in sorted(root.rglob("*counter_collection.csv")):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (
                row["Kernel_Name"].strip('"'),
                row["Grid_Size"],
                row["LDS_Block_Size"],  # separates HS=64 (16896 B) vs HS=128 (33280 B)
            )
            shapes[key] = {
                "VGPR": row["VGPR_Count"],
                "Accum_VGPR": row["Accum_VGPR_Count"],
                "SGPR": row["SGPR_Count"],
                "LDS_bytes": row["LDS_Block_Size"],
                "Workgroup": row["Workgroup_Size"],
            }
            counter = row["Counter_Name"].strip('"')
            try:
                data[key][counter].append(float(row["Counter_Value"]))
            except ValueError:
                pass

for (kernel, grid, lds), counters in sorted(data.items()):
    sh = shapes[(kernel, grid, lds)]
    head_size = "HS=64" if int(lds) < 20000 else "HS=128"
    print(f"\n=== {kernel}  grid={grid}  {head_size} (LDS={lds}B) ===")
    print(f"  VGPR={sh['VGPR']}  Accum_VGPR={sh['Accum_VGPR']}  "
          f"SGPR={sh['SGPR']}  LDS={int(sh['LDS_bytes'])} B  "
          f"Workgroup={sh['Workgroup']}")
    for ctr in sorted(counters):
        vals = counters[ctr]
        mean = sum(vals) / len(vals)
        print(f"    {ctr:24s}  mean={mean:14.4f}  min={min(vals):14.4f}  "
              f"max={max(vals):14.4f}  n={len(vals)}")
