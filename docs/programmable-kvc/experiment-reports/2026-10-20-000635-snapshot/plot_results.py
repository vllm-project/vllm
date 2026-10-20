# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot saved timings; each dot is one independent completed run."""

import argparse
import json
import os
import statistics
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent


def panel(axis, labels, samples, colors, title, *, english=False):
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
            f"{median:,.2f} {'s' if english else '秒'}",
            va="center",
            fontsize=9,
        )
    axis.invert_yaxis()
    axis.set_xlim(0, max(max(row) for row in samples) * 1.28)
    axis.set_xlabel(
        "Batch completion time (seconds; lower is better)"
        if english
        else "批量完成时间（秒，越低越好）",
        fontsize=9,
    )
    axis.set_title(title, loc="left", fontsize=11, pad=14)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    english = parser.parse_args().language == "en"

    def label(chinese, translation):
        return translation if english else chinese

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
        [
            label("原生引擎", "Native engine"),
            label("容量与顺序调度", "Capacity and\nrequest ordering"),
            label("应用协同调度", "Scheduling with\napplication signals"),
        ],
        [pressure[group]["seconds"] for group in ["native", "quota", "cooperative"]],
        ["#8494a7", "#5795bd", "#246b50"],
        label("A. 缓存压力下的主要对照", "A. Main comparison under cache pressure"),
        english=english,
    )
    panel(
        axes[1],
        [
            label("记录工具耗时", "Recorded tool\nduration"),
            label("预测 1 毫秒恢复", "Resume after 1 ms"),
            label("预测 30 秒恢复", "Resume after 30 s"),
        ],
        [
            forecast[group]["seconds"]
            for group in [
                "pressure-oracle",
                "pressure-forecast-early",
                "pressure-forecast-late",
            ]
        ],
        ["#b8a583", "#b8a583", "#c47b37"],
        label("B. 恢复时间预测的影响", "B. Effect of resume-time predictions"),
        english=english,
    )
    axes[1].axvline(
        pressure["cooperative"]["median_seconds"],
        color="#246b50",
        linestyle="--",
        linewidth=1,
    )
    panel(
        axes[2],
        [
            label("原生引擎", "Native engine"),
            label("应用协同调度", "Scheduling with\napplication signals"),
            label("强制搬运策略", "Forced transfer\npolicy"),
        ],
        [fit["native"]["seconds"], fit["cooperative"]["seconds"], rejected_times],
        ["#8494a7", "#246b50", "#b85050"],
        label("C. 容量充足时的搬运代价", "C. Transfer cost with ample capacity"),
        english=english,
    )
    figure.suptitle(
        label(
            "可编程 KV 缓存性能实验  |  双卡 A100 80GB  |  Qwen3.8-27B",
            "Programmable KV cache  |  Two A100 80GB GPUs  |  Qwen3.8-27B",
        ),
        fontsize=13,
    )
    figure.supxlabel(
        label(
            "横条：完成时间中位数；散点：独立运行；虚线：常规应用协同调度。\n"
            "A、B 使用相同压力负载，C 使用容量充足负载；各图横轴范围不同。",
            "Bars: median; dots: individual runs; dashed line: standard scheduling "
            "with application signals.\n"
            "A and B use the same pressure workload; C uses the ample-capacity "
            "workload. Horizontal scales differ.",
        ),
        fontsize=9,
    )
    for axis in axes:
        axis.get_xticklabels()
        axis.get_yticklabels()
    if not english:
        for text in figure.findobj(match=Text):
            text.set_fontproperties(
                FontProperties(fname=font_path, size=text.get_fontsize())
            )
    filename = "performance-en" if english else "performance"
    svg_path = ROOT / f"{filename}.svg"
    figure.savefig(svg_path, metadata={"Date": None})
    # Decimal glyph IDs avoid spelling false positives in generated character codes.
    svg_path.write_text(
        re.sub(
            r'(id|xlink:href)="(#?)NotoSansCJKjp-Regular-([0-9a-f]+)"',
            lambda match: f'{match[1]}="{match[2]}glyph-{int(match[3], 16)}"',
            svg_path.read_text(),
        )
    )
    figure.savefig(
        ROOT / f"{filename}.png", dpi=180, metadata={"Software": "Matplotlib"}
    )
    print(f"Saved {filename}.svg and {filename}.png")


if __name__ == "__main__":
    main()
