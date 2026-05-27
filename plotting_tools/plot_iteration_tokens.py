from __future__ import annotations

import argparse
from pathlib import Path

from plotting_tools.plots import plot_iteration_token_timeline, plot_prefill_zoom_panel
from plotting_tools.trace_io import parse_iteration_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot prefill vs decode tokens per iteration from Slurm logs."
    )
    parser.add_argument(
        "--slurm-out",
        type=Path,
        required=True,
        help="Path to Slurm .out file containing EngineCore Iteration(...) logs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PDF path",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Optional cap on number of iterations plotted",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    parser.add_argument(
        "--zoom-out",
        type=Path,
        default=None,
        help=(
            "Optional output path for prefill-focused zoom panel. "
            "Defaults to <out_stem>_prefill_zoom.pdf"
        ),
    )
    parser.add_argument(
        "--zoom-context-iters",
        type=int,
        default=30,
        help="Context iterations before/after prefill-active region in zoom plot",
    )
    args = parser.parse_args()

    iterations = parse_iteration_log(args.slurm_out)
    title = args.title or f"Tokens per iteration ({args.slurm_out.name})"
    plot_iteration_token_timeline(
        iterations, args.out, title=title, max_iters=args.max_iters
    )
    zoom_out = args.zoom_out or args.out.with_name(f"{args.out.stem}_prefill_zoom.pdf")
    plot_prefill_zoom_panel(
        iterations,
        zoom_out,
        title=f"Prefill-focused zoom ({args.slurm_out.name})",
        context_iters=args.zoom_context_iters,
    )


if __name__ == "__main__":
    main()

