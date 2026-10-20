# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot saved timings; each dot is one independent completed run."""

import json
import os
import statistics
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent


def panel(axis, labels, samples, colors, title):
    medians = [statistics.median(row) for row in samples]
    axis.barh(labels, medians, color=colors, alpha=0.85, height=0.6)
    for position, (values, median) in enumerate(zip(samples, medians)):
        offsets = (
            [0]
            if len(values) == 1
            else [0.12 * (i - (len(values) - 1) / 2) for i in range(len(values))]
        )
        axis.scatter(
            values,
            [position + offset for offset in offsets],
            s=21,
            color="#202020",
            zorder=3,
        )
        axis.text(
            median + max(medians) * 0.025,
            position,
            f"{median:,.2f} 秒",
            va="center",
            fontsize=9,
        )
    axis.invert_yaxis()
    axis.set_xlim(0, max(max(row) for row in samples) * 1.28)
    axis.set_xlabel("批量完成时间（秒，越低越好）", fontsize=9)
    axis.set_title(title, loc="left", fontsize=11, pad=14)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)


def main():
    os.environ["MPLCONFIGDIR"] = str(ROOT / "raw/plot-cache")
    os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    data = json.loads((ROOT / "summary.json").read_text())
    pressure = data["primary_pressure"]
    forecast = data["acceptance_forecasts"]
    fit = data["capacity_fit"]
    rejected = json.loads(
        (
            ROOT / "raw/results/rollout-2026-10-19/fit-cooperative-v3/matrix.json"
        ).read_text()
    )
    rejected_times = [row["seconds"] for row in rejected["runs"]]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "programmable-kv-cache-2026-10-20",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    panel(
        axes[0],
        ["原生引擎", "容量与顺序调度", "应用协同调度"],
        [pressure[group]["seconds"] for group in ["native", "quota", "cooperative"]],
        ["#8494a7", "#5795bd", "#246b50"],
        "A. 缓存压力下的主要对照",
    )
    panel(
        axes[1],
        ["记录工具耗时", "预测 1 毫秒恢复", "预测 30 秒恢复"],
        [
            forecast[group]["seconds"]
            for group in [
                "pressure-oracle",
                "pressure-forecast-early",
                "pressure-forecast-late",
            ]
        ],
        ["#b8a583", "#b8a583", "#c47b37"],
        "B. 恢复时间预测的影响",
    )
    axes[1].axvline(
        pressure["cooperative"]["median_seconds"],
        color="#246b50",
        linestyle="--",
        linewidth=1,
    )
    panel(
        axes[2],
        ["原生引擎", "应用协同调度", "强制搬运策略"],
        [fit["native"]["seconds"], fit["cooperative"]["seconds"], rejected_times],
        ["#8494a7", "#246b50", "#b85050"],
        "C. 容量充足时的搬运代价",
    )
    figure.suptitle(
        "可编程 KV 缓存性能实验  |  双卡 A100 80GB  |  Qwen3.8-27B",
        fontsize=13,
    )
    figure.supxlabel(
        "横条：完成时间中位数；散点：独立运行；虚线：常规应用协同调度。\n"
        "A、B 使用相同压力负载，C 使用容量充足负载；各图横轴范围不同。",
        fontsize=9,
    )
    for axis in axes:
        axis.get_xticklabels()
        axis.get_yticklabels()
    for label in figure.findobj(match=Text):
        label.set_fontproperties(
            FontProperties(fname=font_path, size=label.get_fontsize())
        )
    figure.savefig(ROOT / "performance.svg", metadata={"Date": None})
    svg_path = ROOT / "performance.svg"
    # Decimal glyph IDs avoid spelling false positives in generated character codes.
    svg_path.write_text(
        re.sub(
            r'(id|xlink:href)="(#?)NotoSansCJKjp-Regular-([0-9a-f]+)"',
            lambda match: f'{match[1]}="{match[2]}glyph-{int(match[3], 16)}"',
            svg_path.read_text(),
        )
    )
    figure.savefig(
        ROOT / "performance.png", dpi=180, metadata={"Software": "Matplotlib"}
    )
    print("Saved performance.svg and performance.png")


if __name__ == "__main__":
    main()
