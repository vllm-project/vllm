import math
import pickle
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data: List[TMeasurement] = pickle.load(f)

    results = defaultdict(lambda: list())
    for v in data:
        result = re.search(r"MKN=\(\d+x(\d+x\d+)\)", v.task_spec.sub_label)
        if result is not None:
            KN = result.group(1)
        else:
            raise Exception("MKN not found")
        result = re.search(r"MKN=\((\d+)x\d+x\d+\)", v.task_spec.sub_label)
        if result is not None:
            M = result.group(1)
        else:
            raise Exception("MKN not found")

        kernel = v.task_spec.description
        results[KN].append({
            "kernel": kernel,
            "batch_size": M,
            "median": v.median
        })

    rows = int(math.ceil(len(results) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axs = axs.flatten()
    axs_idx = 0
    for shape, data in results.items():
        plt.sca(axs[axs_idx])
        df = pd.DataFrame(data)
        sns.lineplot(data=df,
                     x="batch_size",
                     y="median",
                     hue="kernel",
                     style="kernel",
                     markers=True,
                     dashes=False,
                     palette="Dark2")
        plt.title(f"Shape: {shape}")
        plt.ylabel("time (median, s)")
        axs_idx += 1
    plt.tight_layout()
    plt.savefig("graph_machete_bench.pdf")
