# SPDX-License-Identifier: Apache-2.0

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from vllm.profiler.layerwise_profile import SummaryStatsEntry

# === Configuration ===
OUTPUT_DIR = "./outputs"
FIGURE_OUTPUT_DIR = "./heatmap_figures"
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)


# === Parse filename like profiling_bs16_pl2048.json ===
def parse_filename(fname):
    parts = fname.replace(".json", "").split("_")
    bs = int(parts[1][2:])
    pl = int(parts[2][2:])
    return bs, pl


# === Extract root-level CUDA time from summary_stats ===
def extract_cuda_time(json_file, phase):
    with open(json_file) as f:
        data = json.load(f)
    if phase not in data:
        return None
    entries = data[phase]["summary_stats"]
    if not entries:
        return None
    root_entry = SummaryStatsEntry(**entries[0]["entry"])
    return root_entry.cuda_time_us / 1000.0  # ms


# === Load all times into dict {(bs, pl): time} ===
def load_profile_times(directory, phase):
    time_map = {}
    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue
        bs, pl = parse_filename(fname)
        json_file = os.path.join(directory, fname)
        t = extract_cuda_time(json_file, phase)
        if t is not None:
            time_map[(bs, pl)] = t
    return time_map


# === Collect times ===
prefill_times = load_profile_times(OUTPUT_DIR, "prefill")
decode_times = load_profile_times(OUTPUT_DIR, "decode_1")

# === Batch sizes and prompt lengths ===
batch_sizes = sorted({k[0] for k in prefill_times})
prompt_lens = sorted({k[1] for k in prefill_times})


# === Create heatmap matrix ===
def build_heatmap(time_data):
    t_mat = np.full((len(batch_sizes), len(prompt_lens)), np.nan)
    for i, bs in enumerate(batch_sizes):
        for j, pl in enumerate(prompt_lens):
            key = (bs, pl)
            if key in time_data:
                t_mat[i, j] = time_data[key]
    return t_mat


prefill_mat = build_heatmap(prefill_times)
decode_mat = build_heatmap(decode_times)


# === Plot heatmaps ===
def plot_heatmap(data, title, filename, fmt=".1f", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(prompt_lens)))
    ax.set_xticklabels(prompt_lens, rotation=45)
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Batch Size")
    ax.set_title(title)

    for i in range(len(batch_sizes)):
        for j in range(len(prompt_lens)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, filename))
    plt.close()


# === Save ===
plot_heatmap(prefill_mat, "Prefill - Time (ms)", "prefill_time.png")
plot_heatmap(decode_mat, "Decode - Time (ms)", "decode_time.png")
