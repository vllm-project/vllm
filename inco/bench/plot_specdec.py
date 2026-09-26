# SPDX-License-Identifier: Apache-2.0
"""Plot a speculative-decoding A/B written by ``modal_speculators.py::acceptance``.

    python -m bench.plot_specdec results/spec/results.json

Three single-axis panels, because the three measures have unrelated scales and
a twin axis would invite reading one curve against the other's gridlines:

1. the Pareto frontier both phases trace as concurrency rises,
2. the drafted/undrafted throughput ratio, which is the claim,
3. acceptance length, which is what the ratio is bought with.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.report import is_steep

# Validated as a categorical pair (CVD dE 21.6 protan, 30.4 normal vision).
# Hue is the model; drafted vs not is solid vs dashed, so identity never
# rests on colour alone.
MODEL_COLORS = ("#d95f02", "#1f77b4")
MODEL_MARKERS = ("o", "s")
# Ink, never a series colour: identity is carried by the marks beside the text.
INK = "#222222"
MUTED = "#666666"

DRAFT_PHASE = "draft"
PLAIN_PHASE = "no-draft"


class MalformedResults(ValueError):
    pass


@dataclass(frozen=True)
class Row:
    """One concurrency, both phases, as the comparison table shows it."""

    concurrency: int
    draft_gpu: float
    plain_gpu: float
    draft_user: float
    plain_user: float
    accept_len: float | None

    @property
    def speedup(self) -> float:
        return self.draft_gpu / self.plain_gpu


def merge(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate several ``acceptance`` runs into one set of phases.

    Later runs win on a shared concurrency. Runs measured on different
    containers are only stitchable when they share at least one point and
    agree on it -- see :func:`overlap_disagreement`, which the caller should
    check before plotting a merged curve.
    """
    if not runs:
        raise MalformedResults("no results to merge")

    merged: dict[str, Any] = {**runs[0], "phases": {}}
    for name in (DRAFT_PHASE, PLAIN_PHASE):
        points: dict[int, Any] = {}
        audit = None
        for run in runs:
            phase = (run.get("phases") or {}).get(name)
            if phase is None:
                continue
            audit = phase.get("audit", audit)
            points.update({p["concurrency"]: p for p in phase["points"]})
        if points:
            merged["phases"][name] = {
                "audit": audit or {},
                "points": [points[c] for c in sorted(points)],
            }
    return merged


def overlap_disagreement(
    runs: list[dict[str, Any]],
) -> dict[tuple[str, int], float]:
    """Relative throughput gap between runs, per (phase, concurrency).

    Keyed by phase as well as concurrency: the two phases are *meant* to differ
    at a given concurrency, so pooling them would report the speedup itself as
    a disagreement. Cross-container variation of a few percent is expected; a
    large gap means the runs are not describing the same system and their
    points must not share a curve.
    """
    seen: dict[tuple[str, int], list[float]] = {}
    for run in runs:
        for name in (DRAFT_PHASE, PLAIN_PHASE):
            phase = (run.get("phases") or {}).get(name)
            for point in (phase or {}).get("points", []):
                key = (name, point["concurrency"])
                seen.setdefault(key, []).append(point["tok_s_gpu"])
    return {
        key: (max(v) - min(v)) / min(v) for key, v in seen.items() if len(v) > 1
    }


def comparison_rows(results: dict[str, Any]) -> list[Row]:
    """Join the two phases on concurrency.

    Raises:
        MalformedResults: if a phase is missing, or the phases were measured at
            different concurrencies -- which would silently produce a ratio
            between two unrelated points.
    """
    phases = results.get("phases") or {}
    missing = [name for name in (DRAFT_PHASE, PLAIN_PHASE) if name not in phases]
    if missing:
        raise MalformedResults(
            f"results have no {missing} phase; run without --skip-baseline"
        )

    drafted = {p["concurrency"]: p for p in phases[DRAFT_PHASE]["points"]}
    plain = {p["concurrency"]: p for p in phases[PLAIN_PHASE]["points"]}
    if drafted.keys() != plain.keys():
        raise MalformedResults(
            f"phases disagree on concurrency: {sorted(drafted)} vs {sorted(plain)}"
        )

    return [
        Row(
            concurrency=c,
            draft_gpu=drafted[c]["tok_s_gpu"],
            plain_gpu=plain[c]["tok_s_gpu"],
            draft_user=drafted[c]["tok_s_user"],
            plain_user=plain[c]["tok_s_user"],
            accept_len=drafted[c].get("accept_len"),
        )
        for c in sorted(drafted)
    ]


def _annotate_concurrency(ax, xs, ys, rows, color) -> None:
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(
            f"c={row.concurrency}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )


def _frontier_labels(curve, xs, ys, drafted: bool, every: bool, spans=None):
    """Which points of one frontier curve get a ``c=`` label, and where.

    Labelling every point only works on the curve's empty side, and which side
    that is changes along the curve. Where it is shallow the free side is
    vertical -- above a drafted point, below an undrafted one, which keeps the
    two apart where they run close. Where throughput saturates and the curve
    turns vertical, a label above or below lands on the next marker, so it
    goes beside the point instead: outward, away from the other phase.

    Otherwise only the ends are labelled -- on a figure carrying four curves a
    label on all ten points of each is noise. Drafted ends sit above the point
    and undrafted below, because the two undrafted curves converge in the
    bottom-left corner and would collide.
    """
    if every:
        placed = []
        for i, (point, x, y) in enumerate(zip(curve, xs, ys)):
            if is_steep(xs, ys, i, spans):
                beside = ((8, 0), "left") if drafted else ((-8, 0), "right")
                placed.append((point, x, y, *beside, "center"))
            else:
                offset = (0, 9) if drafted else (0, -16)
                placed.append((point, x, y, offset, "center", "baseline"))
        return placed

    offset = (6, 5) if drafted else (6, -11)
    # Undrafted c=1 is skipped: every model lands in the same corner there,
    # so those labels stack into an unreadable smudge.
    ends = [(curve[-1], xs[-1], ys[-1], offset, "left", "baseline")]
    if drafted:
        ends.insert(0, (curve[0], xs[0], ys[0], offset, "left", "baseline"))
    return ends


def draw_frontier(
    ax,
    rows: dict[str, list[Row]],
    label_every: bool,
    indices: dict[str, int] | None = None,
) -> None:
    """Draw the latency-throughput frontier of every run onto one axis.

    Args:
        ax: Axis to draw on.
        rows: Joined comparison rows per run label.
        label_every: Label every concurrency, not just the ends of a curve.
        indices: Style slot per label. Pass the label's position in the *full*
            set of runs when drawing a subset, so a model keeps the hue and
            marker it has in the combined figure.
    """
    indices = indices or {label: i for i, label in enumerate(rows)}
    phases = ((True, "-", "+ draft"), (False, "--", "no draft"))
    drawn = []
    for label, curve in rows.items():
        color, marker = _style(indices[label])
        for drafted, style, suffix in phases:
            xs = [r.draft_user if drafted else r.plain_user for r in curve]
            ys = [r.draft_gpu if drafted else r.plain_gpu for r in curve]
            ax.plot(
                xs, ys, style, marker=marker, markersize=5, linewidth=2,
                color=color, label=f"{label}  {suffix}",
                markerfacecolor=color if drafted else "none",
            )
            drawn.append((curve, xs, ys, drafted))

    # Labels are placed against the finished axis: which side of a point is
    # free depends on the limits, so they cannot be decided while drawing.
    if label_every:
        ax.margins(x=0.1, y=0.06)
    spans = tuple(
        hi - lo or 1.0 for lo, hi in (ax.get_xlim(), ax.get_ylim())
    )
    for curve, xs, ys, drafted in drawn:
        for point, x, y, offset, ha, va in _frontier_labels(
            curve, xs, ys, drafted, label_every, spans
        ):
            ax.annotate(
                f"c={point.concurrency}", (x, y), textcoords="offset points",
                xytext=offset, fontsize=8, color=MUTED, ha=ha, va=va,
            )

    ax.set_xlabel("tokens/s/user  (interactivity)")
    ax.set_ylabel("tokens/s/gpu  (throughput)")
    ax.set_title("Latency-throughput frontier")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")


def _header(results: dict[str, Any], wrap: bool = False) -> str:
    """The run conditions every figure of a sweep is captioned with.

    `wrap` breaks before the settings clause, which is the only way the caption
    fits a single-panel figure a third the width of the three-panel one.
    """
    sources = results.get("sources") or {}
    subtitle = ", ".join(f"{n}x {name.split('/')[-1]}" for name, n in sources.items())
    return (
        f"DFlash2 speculative decoding  |  {results.get('prompts')} prompts "
        f"({subtitle})  |{chr(10) if wrap else '  '}"
        f"greedy, max_tokens {results.get('max_tokens')}, "
        "async scheduling and prefix caching off in every phase"
    )


def _pyplot():
    """`matplotlib.pyplot`, styled, or None when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )
    return plt


def _style(index: int) -> tuple[str, str]:
    """Colour carries the model, so a run keeps its hue in every panel."""
    return (
        MODEL_COLORS[index % len(MODEL_COLORS)],
        MODEL_MARKERS[index % len(MODEL_MARKERS)],
    )


def plot(runs: dict[str, dict[str, Any]], path: str | Path) -> Path | None:
    """Write the three-panel figure for one or more runs.

    Each run contributes a drafted and an undrafted curve. Identity is encoded
    twice over: hue for the model, solid vs dashed for drafted vs not, so the
    figure survives being printed or read with a colour vision deficiency.
    """
    plt = _pyplot()
    if plt is None:
        return None

    rows = {label: comparison_rows(results) for label, results in runs.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (pareto, ratio, accept) = plt.subplots(1, 3, figsize=(17, 5.4), dpi=150)

    draw_frontier(pareto, rows, label_every=False)

    for index, (label, curve) in enumerate(rows.items()):
        color, marker = _style(index)
        concurrencies = [r.concurrency for r in curve]
        ratio.plot(
            concurrencies, [r.speedup for r in curve], marker=marker,
            markersize=5, linewidth=2, color=color, label=label,
        )
        lengths = [(r.concurrency, r.accept_len) for r in curve if r.accept_len]
        if lengths:
            accept.plot(
                [c for c, _ in lengths], [v for _, v in lengths], marker=marker,
                markersize=5, linewidth=2, color=color, label=label,
            )

    ticks = sorted({r.concurrency for curve in rows.values() for r in curve})
    ratio.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    ratio.annotate(
        "no gain from drafting", (ticks[0], 1.0), textcoords="offset points",
        xytext=(2, 5), fontsize=8, color=MUTED,
    )
    ratio.set_ylim(0, max(r.speedup for c in rows.values() for r in c) * 1.15)
    ratio.set_title("Speedup from the drafter")
    ratio.set_ylabel("throughput ratio  (drafted / no draft)")

    accept.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    accept.annotate(
        "1.0 = nothing accepted", (ticks[0], 1.0), textcoords="offset points",
        xytext=(2, 5), fontsize=8, color=MUTED,
    )
    lengths = [r.accept_len for c in rows.values() for r in c if r.accept_len]
    if lengths:
        accept.set_ylim(0, max(lengths) * 1.3)
    accept.set_title("Mean accepted length")
    accept.set_ylabel("tokens committed per step")

    for axis in (ratio, accept):
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        # Rotated because 48/64/80/96 sit almost on top of each other on a
        # log2 axis, which is where the sweep's interesting points are.
        axis.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right")
        axis.set_xlabel("concurrency")
        axis.grid(alpha=0.3)
        if len(rows) > 1:
            axis.legend(loc="lower right")

    fig.suptitle(_header(next(iter(runs.values()))), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def slug(label: str) -> str:
    """Filename-safe form of a series label (``REAP-50%`` -> ``reap-50``)."""
    kept = "".join(c if c.isalnum() else "-" for c in label.lower())
    return "-".join(part for part in kept.split("-") if part) or "run"


def plot_frontier(
    runs: dict[str, dict[str, Any]],
    path: str | Path,
    label_every: bool = True,
    indices: dict[str, int] | None = None,
) -> Path | None:
    """Write the latency-throughput frontier on its own, one run or several.

    The panel is unreadable at a third of the width once every point carries a
    ``c=`` label, which is the point of drawing it alone.
    """
    plt = _pyplot()
    if plt is None:
        return None

    rows = {label: comparison_rows(results) for label, results in runs.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    draw_frontier(ax, rows, label_every=label_every, indices=indices)
    fig.suptitle(_header(next(iter(runs.values())), wrap=True), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    return path


def frontiers(runs: dict[str, dict[str, Any]], out: Path) -> list[Path]:
    """Write the frontier panel alone, once per run and once with all of them.

    Named off the combined figure: ``specdec-v5.png`` yields
    ``specdec-v5-frontier.png`` and ``specdec-v5-frontier-<run>.png``. A single
    run needs no all-runs copy -- it would be the same figure twice.
    """
    indices = {label: i for i, label in enumerate(runs)}
    written = []
    for label, results in runs.items():
        path = out.with_name(f"{out.stem}-frontier-{slug(label)}{out.suffix}")
        written.append(plot_frontier({label: results}, path, indices=indices))
    if len(runs) > 1:
        path = out.with_name(f"{out.stem}-frontier{out.suffix}")
        written.append(plot_frontier(runs, path, label_every=False))
    return [path for path in written if path is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="+", help="results.json from ::acceptance"
    )
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated series names, one per results file",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    loaded = [json.loads(path.read_text()) for path in args.results]
    names = [n.strip() for n in args.labels.split(",") if n.strip()]
    if names and len(names) != len(loaded):
        print(f"{len(names)} labels for {len(loaded)} files", file=sys.stderr)
        return 2
    if not names:
        names = [path.stem for path in args.results]

    runs = dict(zip(names, loaded))
    out = args.out or args.results[0].parent / "specdec-comparison.png"
    written = plot(runs, out)
    if written is None:
        print("matplotlib is not installed", file=sys.stderr)
        return 1
    for path in frontiers(runs, out):
        print(f"wrote {path}")
    for label, results in runs.items():
        print(f"--- {label}")
        for row in comparison_rows(results):
            print(
                f"c={row.concurrency:<3} {row.draft_gpu:>8.1f} / {row.plain_gpu:>8.1f} "
                f"tok/s/gpu = {row.speedup:.2f}x   accept_len {row.accept_len}"
            )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
