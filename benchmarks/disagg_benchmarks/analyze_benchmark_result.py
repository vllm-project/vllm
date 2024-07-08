
import argparse
import json
import yaml
import os
from pathlib import Path

def load(path):
    
    with open(str(path), 'r') as f:
        return json.loads(f.read())

def main(args):

    results = Path(args.results_folder)

    chunk = load(results / "chunked_prefill_tp4.json")
    prefill = load(results / "disagg_prefill_tp4.json")
    decode = load(results / "disagg_decode_tp4.json")

    ttft_ratio = chunk["mean_ttft_ms"] / prefill["mean_ttft_ms"]
    itl_ratio = chunk["mean_itl_ms"] / decode["mean_itl_ms"]
    prefill_decode_ratio = prefill["mean_ttft_ms"] / (decode["mean_itl_ms"] * args.output_len)
    
    with open(results / args.output_file, 'a') as f:
        f.write(yaml.dump([{
            'qps': args.qps,
            'output_len': args.output_len,
            'prefill_decode_ratio': prefill_decode_ratio,
            'ttft_ratio': ttft_ratio,
            'itl_ratio': itl_ratio,
            "chunk_ttft": chunk["mean_ttft_ms"],
            "chunk_itl": chunk["mean_itl_ms"],
            "disagg_ttft": prefill["mean_ttft_ms"],
            "disagg_itl": decode["mean_itl_ms"]
        }]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-folder", required=True, help="Path to the results folder")
    parser.add_argument("--output-len", type=int, required=True, help="Target output length")
    parser.add_argument("--qps", type=int, required=True, help="Target QPS")
    parser.add_argument("--output-file", type=str, default="chunk_vs_disagg.yaml")

    args = parser.parse_args()
    main(args)