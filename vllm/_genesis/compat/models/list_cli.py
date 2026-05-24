# SPDX-License-Identifier: Apache-2.0
"""Genesis models — list CLI.

Usage:
  python3 -m vllm._genesis.compat.models.list_cli
  python3 -m vllm._genesis.compat.models.list_cli --status PROD
  python3 -m vllm._genesis.compat.models.list_cli --json

Author: Sandermage(Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import json
import sys


def _format_table(models) -> list[str]:
    rows = [("KEY", "TITLE", "SIZE", "QUANT", "STATUS", "HW")]
    for m in models:
        rows.append((
            m.key,
            (m.title[:40] + "…") if len(m.title) > 40 else m.title,
            f"{m.size_gb:.1f}GB",
            m.quant_format,
            m.status,
            ", ".join(m.tested_hardware[:3]) + (" …" if len(m.tested_hardware) > 3 else ""),
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
        out.append(line)
        if i == 0:  # separator after header
            out.append("  ".join("─" * widths[i] for i in range(len(rows[0]))))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.models.list_cli",
        description="List Genesis-supported models",
    )
    p.add_argument("--status", choices=["PROD", "SUPPORTED", "EXPERIMENTAL", "PLANNED"],
                   default=None, help="Filter by status")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    args = p.parse_args(argv)

    from vllm._genesis.compat.models.registry import list_models

    models = list_models(status_filter=args.status)
    if not models:
        print("(no models)", file=sys.stderr)
        return 0

    if args.json:
        out = [
            {
                "key": m.key, "hf_id": m.hf_id, "title": m.title,
                "size_gb": m.size_gb, "quant_format": m.quant_format,
                "status": m.status, "tested_hardware": list(m.tested_hardware),
                "recommended_workloads": list(m.recommended_workloads),
                "license": m.license, "gated": m.gated,
            }
            for m in models
        ]
        print(json.dumps(out, indent=2))
        return 0

    print("=" * 72)
    print(f"Genesis-supported models — {len(models)} entries")
    if args.status:
        print(f"Filter: status={args.status}")
    print("=" * 72)
    for line in _format_table(models):
        print(line)
    print()
    print("To download a model:")
    print("  python3 -m vllm._genesis.compat.models.pull <key>")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
