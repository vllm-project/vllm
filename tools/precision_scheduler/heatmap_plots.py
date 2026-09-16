# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PNG rendering of the BF16 / INT4 / speedup matrices (matplotlib, imported lazily)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def plot_matrix(
    values: list[list[float | None]],
    batch_sizes: list[int],
    seq_lens: list[int],
    title: str,
    colorbar_label: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    array = np.array(
        [[np.nan if value is None else value for value in row] for row in values],
        dtype=float,
    )
    masked = np.ma.masked_invalid(array)
    is_speedup = "Speedup" in title
    cmap = plt.get_cmap("RdYlGn" if is_speedup else "viridis").with_extremes(
        bad="#d9d9d9"
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(seq_lens) * 1.05), max(4.5, len(batch_sizes) * 0.7))
    )
    finite = array[np.isfinite(array)]
    norm = None
    if is_speedup and finite.size:
        # Keep the semantic boundary fixed: slowdowns (< 1.0x) are red, 1.0x is
        # neutral yellow, and every speedup (> 1.0x) is green.
        norm = TwoSlopeNorm(
            vmin=min(float(np.min(finite)), 0.95),
            vcenter=1.0,
            vmax=max(float(np.max(finite)), 1.05),
        )
    image = axis.imshow(masked, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    axis.set_xticks(range(len(seq_lens)), labels=[str(value) for value in seq_lens])
    axis.set_yticks(
        range(len(batch_sizes)), labels=[str(value) for value in batch_sizes]
    )
    axis.set_xlabel("Prompt sequence length")
    axis.set_ylabel("Batch size")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    for row_index in range(len(batch_sizes)):
        for column_index in range(len(seq_lens)):
            value = array[row_index, column_index]
            label = "N/A" if not np.isfinite(value) else f"{value:.3f}"
            if np.isfinite(value):
                rgba = cmap(norm(value) if norm is not None else image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.48 else "black"
            else:
                color = "black"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    figure.tight_layout()
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, output)


def write_pngs(output_dir: Path, payload: dict[str, Any]) -> None:
    batch_sizes, seq_lens = payload["batch_sizes"], payload["seq_lens"]
    plot_matrix(
        payload["bf16_tpot_ms"],
        batch_sizes,
        seq_lens,
        "BF16 decode TPOT",
        "ms/token",
        output_dir / "bf16_tpot_heatmap.png",
    )
    plot_matrix(
        payload["int4_tpot_ms"],
        batch_sizes,
        seq_lens,
        "INT4 decode TPOT",
        "ms/token",
        output_dir / "int4_tpot_heatmap.png",
    )
    plot_matrix(
        payload["speedup_bf16_over_int4"],
        batch_sizes,
        seq_lens,
        "BF16 / INT4 Speedup",
        "speedup (higher is better for INT4)",
        output_dir / "speedup_heatmap.png",
    )
    if any(v is not None for row in payload.get("nvfp4_tpot_ms", []) for v in row):
        plot_matrix(
            payload["nvfp4_tpot_ms"],
            batch_sizes,
            seq_lens,
            "NVFP4 decode TPOT",
            "ms/token",
            output_dir / "nvfp4_tpot_heatmap.png",
        )
        plot_matrix(
            payload["speedup_bf16_over_nvfp4"],
            batch_sizes,
            seq_lens,
            "BF16 / NVFP4 Speedup",
            "speedup (higher is better for NVFP4)",
            output_dir / "speedup_nvfp4_heatmap.png",
        )
