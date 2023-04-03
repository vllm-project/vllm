import csv
import json
import os
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pandas as pd

log_dir = 'log/'
log_files = os.listdir(log_dir)
all_results = []

for log_file in log_files:
    file_path = os.path.join(log_dir, log_file)
    lines = list(open(file_path).readlines())
    profile_arguments = json.loads(lines[0])
    results = defaultdict(list)
    for line in lines:
        if "prompt_latency_seconds" not in line:
            continue
        result = json.loads(line)
        for k, v in result.items():
            if k == "step":
                continue
            results[k].append(v)
    final_result = {
        "model": profile_arguments["model"],
        "batch_size": profile_arguments["batch_size"],
        "input_len": profile_arguments["input_len"],
        "output_len": profile_arguments["output_len"],
        "tensor_parallel_size": profile_arguments["tensor_parallel_size"],
    }

    for k, v in results.items():
        final_result[k + "_mean"] = np.mean(v)
        final_result[k + "_std"] = np.std(v)

    all_results.append(final_result)

df = pd.DataFrame.from_records(all_results)

print(df)

df.to_csv('parse_result.csv', index=False)