# SPDX-License-Identifier: Apache-2.0

import json

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    data = []
    for name in ["disagg_prefill", "chunked_prefill"]:
        for qps in [2, 4, 6, 8]:
            with open(f"results/{name}-qps-{qps}.json") as f:
                x = json.load(f)
                x["name"] = name
                x["qps"] = qps
                data.append(x)

    df = pd.DataFrame.from_dict(data)
    dis_df = df[df["name"] == "disagg_prefill"]
    chu_df = df[df["name"] == "chunked_prefill"]

    plt.style.use("bmh")
    plt.rcParams["font.size"] = 20

    for key in [
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
    ]:
        fig, ax = plt.subplots(figsize=(11, 7))
        plt.plot(
            dis_df["qps"], dis_df[key], label="disagg_prefill", marker="o", linewidth=4
        )
        plt.plot(
            chu_df["qps"], chu_df[key], label="chunked_prefill", marker="o", linewidth=4
        )
        ax.legend()

        ax.set_xlabel("QPS")
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0)
        fig.savefig(f"results/{key}.png")
        plt.close(fig)
