# SPDX-License-Identifier: Apache-2.0
"""Expert saliency heatmaps from a REAP calibration observer dump.

The observer records one saliency scalar per (layer, expert); `reap` is
`mean(||expert_output|| * router_weight)` over the tokens routed to that
expert, which is the metric `--prune_method reap` ranks on. Pruning is
per-layer top-k, so the absolute scale of a layer never matters -- only the
spread within it. Absolute saliency grows ~200x from layer 0 to layer 47, so
the heatmaps are drawn as a log ratio to each layer's median and the raw scale
is kept in its own panel. A handful of layers hold one expert worth 80x their
median, which is why normalizing by the max instead would flatten every layer
that contains one.

Usage:
    python scripts/plot_reap_saliency.py [observations.pt] [--ratio 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    ListedColormap,
    LogNorm,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_DUMP = RESULTS / "saliency" / "observations_1024_cosine-seed_42.pt"
OUT = RESULTS / "saliency" / "reap-saliency-heatmap.png"

# Sequential single-hue ramp off the REAP style's blue: light = least salient
# (pruned first), dark = most salient.
SALIENCY_CMAP = LinearSegmentedColormap.from_list(
    "reap_blues", ["#f4f7fb", "#c3d5e8", "#7ba4cb", "#4477aa", "#1f3f61"]
)
PRUNED, RETAINED = "#e8e8e8", "#4477aa"

# Colour limits as a multiple of the layer median; the top decile runs well
# past 4x, so the ramp saturates there rather than spending its range on
# outliers.
VMIN, VMAX = 0.25, 4.0
CBAR_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0)


def load_saliency(path: Path, metric: str = "reap") -> np.ndarray:
    """Stack the per-expert saliency of every layer into a (layers, experts) array.

    Args:
        path: Observer state written by `reap.layerwise_prune`.
        metric: Observer key to read, e.g. `reap`, `ean_sum`, `expert_frequency`.

    Returns:
        Saliency indexed by ascending layer, then expert index.

    Raises:
        KeyError: The dump has no such metric.
    """
    import torch

    data = torch.load(path, weights_only=False)
    layers = sorted(data)
    missing = [layer for layer in layers if metric not in data[layer]]
    if missing:
        raise KeyError(
            f"metric {metric!r} missing for layers {missing[:4]}; "
            f"available: {sorted(data[layers[0]])}"
        )
    return np.stack([data[layer][metric].float().numpy() for layer in layers])


def relative_to_layer_median(saliency: np.ndarray) -> np.ndarray:
    """Express each expert as a multiple of its own layer's median saliency.

    The median is robust to the single outsized expert some layers carry, so
    unlike max-normalization it keeps the bulk of every layer in the middle of
    the colour ramp and layers stay comparable to each other.
    """
    mid = np.median(saliency, axis=1, keepdims=True)
    return np.divide(saliency, mid, out=np.zeros_like(saliency), where=mid != 0)


def sort_within_layer(saliency: np.ndarray) -> np.ndarray:
    """Reorder each layer's experts most- to least-salient (rank on the x axis)."""
    return -np.sort(-saliency, axis=1)


def experts_pruned(num_experts: int, ratio: float) -> int:
    """How many experts per layer a compression `ratio` deletes, as REAP rounds it."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    return int(num_experts * ratio)


def retained_mask(saliency: np.ndarray, ratio: float) -> np.ndarray:
    """True where an expert survives per-layer top-k pruning at `ratio`.

    Mirrors `reap.prune.prune`: the `n * ratio` lowest-saliency experts in each
    layer are dropped independently of every other layer.
    """
    n_prune = experts_pruned(saliency.shape[1], ratio)
    if n_prune == 0:
        return np.ones_like(saliency, dtype=bool)
    cut = np.argsort(np.argsort(saliency, axis=1), axis=1)
    return cut >= n_prune


def _colorbar(fig, image, ax):
    bar = fig.colorbar(
        image,
        ax=ax,
        label="saliency / layer median",
        pad=0.02,
        extend="both",
        ticks=CBAR_TICKS,
    )
    bar.ax.set_yticklabels([f"{tick:g}x" for tick in CBAR_TICKS])
    bar.ax.minorticks_off()
    return bar


def _heatmap(ax, values, title, xlabel, cmap, norm=None):
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else 0.0,
        vmax=None if norm else 1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("decoder layer")
    return image


def main(dump: Path = DEFAULT_DUMP, ratio: float = 0.5, out: Path = OUT) -> Path:
    saliency = load_saliency(dump)
    n_layers, n_experts = saliency.shape
    normalized = relative_to_layer_median(saliency)
    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    retained = retained_mask(saliency, ratio)
    n_prune = experts_pruned(n_experts, ratio)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    (index_ax, mask_ax), (rank_ax, scale_ax) = axes

    image = _heatmap(
        index_ax,
        normalized,
        "REAP saliency by expert index, as a multiple of the layer median\n"
        "light = least salient, pruned first",
        "expert index",
        SALIENCY_CMAP,
        norm,
    )
    _colorbar(fig, image, index_ax)

    _heatmap(
        mask_ax,
        retained.astype(float),
        f"Which experts survive {ratio:.0%} pruning "
        f"({n_experts - n_prune} of {n_experts} kept per layer)",
        "expert index",
        ListedColormap([PRUNED, RETAINED]),
    )
    mask_ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=RETAINED, label="retained"),
            plt.Rectangle((0, 0), 1, 1, fc=PRUNED, ec="#999999", label="pruned"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    ranked = _heatmap(
        rank_ax,
        sort_within_layer(normalized),
        "Same rows sorted by saliency: how concentrated each layer is",
        "within-layer saliency rank",
        SALIENCY_CMAP,
        norm,
    )
    _colorbar(fig, ranked, rank_ax)
    rank_ax.axvline(n_experts - n_prune - 0.5, color="#ee6677", lw=2)
    rank_ax.text(
        n_experts - n_prune + 2,
        n_layers * 0.02,
        f"{ratio:.0%} cut",
        color="#ee6677",
        fontsize=9,
        va="bottom",
    )

    layers = np.arange(n_layers)
    scale_ax.plot(layers, saliency.max(axis=1), label="max expert")
    scale_ax.plot(layers, np.median(saliency, axis=1), label="median expert")
    scale_ax.plot(layers, saliency.min(axis=1), label="min expert")
    scale_ax.set_yscale("log")
    scale_ax.set_title(
        "Absolute saliency per layer (log) -- why the heatmaps are normalized",
        fontsize=11,
    )
    scale_ax.set_xlabel("decoder layer")
    scale_ax.set_ylabel("REAP saliency")
    scale_ax.grid(alpha=0.3)
    scale_ax.legend(fontsize=9)

    fig.suptitle(
        "Qwen3-30B-A3B-Instruct-2507 REAP calibration saliency\n"
        f"{n_layers} layers x {n_experts} experts, evol-codealpaca-v1, "
        "1024 samples, seed 42, renormalized router weights",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", nargs="?", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(main(args.dump, args.ratio, args.out))
