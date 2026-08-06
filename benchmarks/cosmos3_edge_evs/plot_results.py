#!/usr/bin/env python3
"""Plot Cosmos3-Edge EVS concurrent + accuracy results.

Default: uses the checked-in results.json from the Nebius run.
Optionally rebuild from a directory of bench_evs_concurrent / accuracy JSONs.

Examples:
  python3 plot_results.py
  python3 plot_results.py --data results.json --out-dir plots
  python3 plot_results.py --from-dir ~/evs_eval/results/cosmos --out-dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_data(path: Path) -> dict:
    return json.loads(path.read_text())


def load_from_dir(d: Path) -> dict:
    """Rebuild summary dict from individual bench JSON files."""
    concurrent = []
    for c in [1, 2, 4, 8, 16, 32]:
        b = json.loads((d / f"baseline-c{c}.json").read_text())
        e = json.loads((d / f"evs-c{c}.json").read_text())
        concurrent.append(
            {
                "concurrency": c,
                "baseline": {
                    "prompt_tokens": b["prompt_tokens_mean"],
                    "throughput_rps": b["throughput_rps"],
                    "mean_ms": b["mean_ms"],
                    "success_rate": b["success_rate"],
                },
                "evs": {
                    "prompt_tokens": e["prompt_tokens_mean"],
                    "throughput_rps": e["throughput_rps"],
                    "mean_ms": e["mean_ms"],
                    "success_rate": e["success_rate"],
                },
            }
        )
    acc_path = d / "acc-summary.json"
    if acc_path.exists():
        acc = json.loads(acc_path.read_text())
        accuracy = {
            "baseline_correct": acc.get("baseline_correct"),
            "evs_correct": acc.get("evs_correct"),
            "n": acc.get("n"),
            "baseline_acc_pct": acc["baseline_acc"],
            "evs_acc_pct": acc["evs_acc"],
            "acc_delta_pp": acc["acc_delta_pp"],
        }
    else:
        b = json.loads((d / "acc-baseline-120.json").read_text())
        e = json.loads((d / "acc-evs-120.json").read_text())
        accuracy = {
            "baseline_correct": b["correct"],
            "evs_correct": e["correct"],
            "n": b["n"],
            "baseline_acc_pct": b["accuracy"] * 100,
            "evs_acc_pct": e["accuracy"] * 100,
            "acc_delta_pp": (e["accuracy"] - b["accuracy"]) * 100,
        }
    return {
        "protocol": {"source_dir": str(d)},
        "concurrent": concurrent,
        "accuracy": accuracy,
    }


def markdown_table(data: dict) -> str:
    rows = data["concurrent"]
    lines = [
        "| C | Base tok | EVS tok | Ratio | Base RPS | EVS RPS | Base ms | EVS ms |",
        "|---|----------|---------|-------|----------|---------|---------|--------|",
    ]
    for r in rows:
        b, e = r["baseline"], r["evs"]
        ratio = b["prompt_tokens"] / e["prompt_tokens"] if e["prompt_tokens"] else 0
        lines.append(
            f"| {r['concurrency']} | {b['prompt_tokens']} | {e['prompt_tokens']} | "
            f"{ratio:.2f}× | {b['throughput_rps']:.3f} | {e['throughput_rps']:.3f} | "
            f"{b['mean_ms']:.0f} | {e['mean_ms']:.0f} |"
        )
    acc = data["accuracy"]
    lines += [
        "",
        f"**Accuracy:** baseline {acc['baseline_acc_pct']:.1f}% "
        f"({acc['baseline_correct']}/{acc['n']}) · "
        f"EVS {acc['evs_acc_pct']:.1f}% ({acc['evs_correct']}/{acc['n']}) · "
        f"Δ {acc['acc_delta_pp']:+.1f} pp",
    ]
    return "\n".join(lines)


def plot(data: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = data["concurrent"]
    cs = [r["concurrency"] for r in rows]
    base_rps = [r["baseline"]["throughput_rps"] for r in rows]
    evs_rps = [r["evs"]["throughput_rps"] for r in rows]
    base_ms = [r["baseline"]["mean_ms"] for r in rows]
    evs_ms = [r["evs"]["mean_ms"] for r in rows]
    base_tok = rows[0]["baseline"]["prompt_tokens"]
    evs_tok = rows[0]["evs"]["prompt_tokens"]
    acc = data["accuracy"]

    # 1) Prompt tokens
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(
        ["Baseline", "EVS q=0.5"],
        [base_tok, evs_tok],
        color=["#6b7280", "#2563eb"],
    )
    ax.set_ylabel("Prompt tokens")
    ax.set_title("Prompt tokens (same 10s@3fps clip)")
    for bar, v in zip(bars, [base_tok, evs_tok]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(base_tok, evs_tok) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_tokens.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out_dir / 'prompt_tokens.png'}")

    # 2) Throughput vs concurrency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cs, base_rps, "o-", color="#6b7280", label="Baseline", linewidth=2)
    ax.plot(cs, evs_rps, "s-", color="#2563eb", label="EVS q=0.5", linewidth=2)
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs concurrency")
    ax.set_xticks(cs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_concurrency.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out_dir / 'throughput_vs_concurrency.png'}")

    # 3) Latency vs concurrency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cs, base_ms, "o-", color="#6b7280", label="Baseline", linewidth=2)
    ax.plot(cs, evs_ms, "s-", color="#d97706", label="EVS q=0.5", linewidth=2)
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Mean end-to-end latency (ms)")
    ax.set_title("Mean latency vs concurrency (256 fixed output tokens)")
    ax.set_xticks(cs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_concurrency.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out_dir / 'latency_vs_concurrency.png'}")

    # 4) Accuracy
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vals = [acc["baseline_acc_pct"], acc["evs_acc_pct"]]
    bars = ax.bar(
        ["Baseline", "EVS q=0.5"],
        vals,
        color=["#6b7280", "#16a34a"],
    )
    drop_floor = acc["baseline_acc_pct"] - 3.0
    ax.axhline(
        drop_floor,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"−3pp drop floor ({drop_floor:.1f}%)",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"VideoMME accuracy ({acc['n']} MCQs)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out_dir / 'accuracy.png'}")

    # 5) Combined overview figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.bar(["Baseline", "EVS"], [base_tok, evs_tok], color=["#6b7280", "#2563eb"])
    ax.set_title("Prompt tokens")
    ax.set_ylabel("tokens")

    ax = axes[0, 1]
    ax.bar(
        ["Baseline", "EVS"],
        [acc["baseline_acc_pct"], acc["evs_acc_pct"]],
        color=["#6b7280", "#16a34a"],
    )
    ax.axhline(acc["baseline_acc_pct"] - 3, color="#dc2626", linestyle="--", lw=1)
    ax.set_title(f"Accuracy (Δ {acc['acc_delta_pp']:+.1f} pp)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    ax = axes[1, 0]
    ax.plot(cs, base_rps, "o-", color="#6b7280", label="Baseline")
    ax.plot(cs, evs_rps, "s-", color="#2563eb", label="EVS")
    ax.set_title("Throughput vs C")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("req/s")
    ax.set_xticks(cs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(cs, base_ms, "o-", color="#6b7280", label="Baseline")
    ax.plot(cs, evs_ms, "s-", color="#d97706", label="EVS")
    ax.set_title("Latency vs C")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("ms")
    ax.set_xticks(cs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cosmos3-Edge · EVS q=0.5 · 10s@3fps 720p · 256 out · Nebius H200",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out_dir / 'overview.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=HERE / "results.json",
        help="Summary JSON (default: results.json next to this script)",
    )
    p.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Optional dir of baseline-c*.json / evs-c*.json / acc-*.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "plots",
        help="Output directory for PNGs + table.md",
    )
    args = p.parse_args()

    if args.from_dir is not None:
        data = load_from_dir(args.from_dir)
    else:
        data = load_data(args.data)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = markdown_table(data)
    (args.out_dir / "table.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {args.out_dir / 'table.md'}")
    plot(data, args.out_dir)


if __name__ == "__main__":
    main()
