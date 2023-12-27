from dataclasses import dataclass
import subprocess
from subprocess import CompletedProcess
from typing import List
from enum import Enum
import torch
import os
import time
import signal
import csv
import argparse
import json
from pathlib import Path
# from analyse import analyse_results

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROFILING_DATA_DIR = PROJECT_ROOT_DIR / "data"

class DumpFormat(Enum):
    JSON = 1
    CSV = 2

# TODO: for automation, load these from a json file
@dataclass
class BenchSetting:
    model: str
    tokenizer: str
    device: str
    backend: str
    dataset: str
    req_rate: float
    tp: int
    iteration_num: int
    gen_len: int
    prompt_len: int
    num_requests: int

    @staticmethod
    def get_head(format):
        assert format == DumpFormat.CSV
        return ["model", "device", "backend", "dataset", "req_rate", "tp"]

    def get_value_list(self, format):
        assert format == DumpFormat.CSV
        return [self.model, self.device, self.backend, self.dataset, self.req_rate, self.tp]

@dataclass
class BenchResult:
    setting: BenchSetting
    avg_per_token_latency: float
    avg_per_output_token_latency: float
    avg_latency: float
    total_time: float
    throughput: float

    @staticmethod
    def get_head(format):
        assert format == DumpFormat.CSV
        return (BenchSetting.get_head(format) +
                ["avg_per_token_latency", "avg_per_output_token_latency",
                 "avg_latency", "total_time", "throughput"])

    def get_value_list(self, format):
        assert format == DumpFormat.CSV
        return (self.setting.get_value_list(format) +
                [self.avg_per_token_latency, self.avg_per_output_token_latency,
                 self.avg_latency, self.total_time, self.throughput])

class Util:
    @staticmethod
    def run_cmd(cmd, blocking = True):
        def set_new_pgroup():
            os.setpgrp()
        print(cmd)
        if blocking:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            return subprocess.Popen(cmd, shell=True, preexec_fn=set_new_pgroup)

class BenchEngine:
    def __init__(self, backend, model, tokenizer, tp, vllm_dir) -> None:
        self.backend_process: subprocess.Popen = None
        # launch backend
        if backend == "vllm":
            cmd = ("python -m vllm.entrypoints.api_server"
                  f" --model {model} --tensor-parallel-size {tp} "
                  f" --tokenizer {tokenizer} --disable-log-requests")
            self.backend_process = Util.run_cmd(cmd, False)
            self.vllm_dir = vllm_dir
        elif backend == "tgi":
            cmd = ("docker run --gpus all --shm-size 1g -p 8000:80 "
                "-v $PWD/data:/data "
                "ghcr.io/huggingface/text-generation-inference:0.9 "
                "--model-id lmsys/vicuna-13b-v1.3 "
                f"--num-shard {tp}  "
                "--max-input-length 4096 "
                "--max-total-tokens 4098 "
                "--max-best-of 5 "
                "--max-concurrent-requests 5000 "
                "--max-batch-total-tokens 4098")
            self.backend_process = Util.run_cmd(cmd, False)
        else:
            raise NotImplementedError(f"{backend}")

    def bench(self, runs : List[BenchSetting]) -> List[BenchResult]:
        time.sleep(120)
        print("============Start Benchmarking==================")
        return [self.bench_single(run) for run in runs]

    def bench_single(self, run: BenchSetting) -> BenchResult:
        cmd = (f"python {self.vllm_dir}/benchmarks/cllam/bench_serving.py"
               f" --dataset {run.dataset} --backend {run.backend}"
               f" --tokenizer {run.tokenizer} --request-rate {run.req_rate}"
               f" --iteration_num {run.iteration_num} --gen_len {run.gen_len} --prompt_len {run.prompt_len}"
               f" --num-prompts {run.num_requests}")
        completed_process: CompletedProcess = Util.run_cmd(cmd, True)
        def process_output(completed_process: CompletedProcess):
            if completed_process.returncode != 0:
                print(f"[Error] {completed_process.stdout}")
                print(f"[Error] {completed_process.stderr}")
                return BenchResult(run, -1, -1, -1, -1, -1)
            lines = completed_process.stdout.split('\n')
            for line in lines:
                if 'Total time' in line:
                    total_time = float(line.split(" ")[-2])
                if 'Throughput' in line:
                    throughput = float(line.split(" ")[-2])
                if 'Average latency' in line:
                    avg_latency = float(line.split(" ")[-2])
                if 'Average latency per token' in line:
                    avg_per_token_latency = float(line.split(" ")[-2])
                if 'Average latency per output token' in line:
                    avg_per_output_token_latency = float(line.split(" ")[-2])
            return BenchResult(run, avg_per_token_latency, avg_per_output_token_latency, avg_latency, total_time, throughput)
        return process_output(completed_process)

    def dump_results(self, results: List[BenchResult], outfile: str, format: DumpFormat) -> None:
        with open(outfile, "w") as f:
            if format == DumpFormat.CSV:
                writer = csv.writer(f)
                writer.writerow(BenchResult.get_head(format))
            for result in results:
                writer.writerow(result.get_value_list(format))


    def __del__(self):
        # stop backend
        print("==============Finish Benchmarking==============")
        if self.backend_process.poll() is None:  # If poll() returns None, the process is still running
            print("Process is running, trying to kill...")
            os.killpg(self.backend_process.pid, signal.SIGINT)
            time.sleep(10) # wait a bit for cleaning resources
            self.backend_process.terminate()
            self.backend_process.wait()
            time.sleep(1)
            if self.backend_process.poll() is not None:
                print(f"Process {self.backend_process.pid} killed successfully.")
            else:
                print("Failed to kill process.")
        else:
            print("Process already terminated.")

def dump_latency_per_token(mean_latency_per_prompt_token, mean_latency_per_gen_token, request_rates, prompt_len, gen_len):
    data = []
    for i, req_rate in enumerate(request_rates):
        data.append({
            "request_rate": req_rate,
            "latency_per_prompt_token": mean_latency_per_prompt_token[i],
            "latency_per_gen_token": mean_latency_per_gen_token[i]
        })

    with open(f"data/latency_per_token-{prompt_len}-{gen_len}.json", "w") as f:
        json.dump(data, f, indent=4)

def main(vllm_dir, model, tokenizer, backend, dataset, outfile, prompt_len, gen_len, num_requests, repeat_num, request_rate_params):
    device = torch.cuda.get_device_name(0)
    request_rate_start, request_rate_end, step = request_rate_params

    for tp in [1]:
        runs = []
        # warmup
        runs.append(BenchSetting(model, tokenizer, device, backend, dataset, 50.0, tp, -1, gen_len, prompt_len, num_requests))
        request_rates = []
        for iteration_num in range(repeat_num):
            # All * 10 to generate non-integer request rates
            for req_rate in range(request_rate_start * 10, (request_rate_end + step) * 10, step * 10):
                req_rate = req_rate / 10.0
                request_rates.append(req_rate)
                runs.append(BenchSetting(model, tokenizer, device, backend, dataset, req_rate, tp, iteration_num, gen_len, prompt_len, num_requests))
        engine = BenchEngine(backend, model, tokenizer, tp, vllm_dir)
        results = engine.bench(runs)
        engine.dump_results(results, "data/" + outfile + f"_tp{tp}", DumpFormat.CSV)

        # # analyse and dump latency per prompt and generated token
        # mean_latency_per_prompt_token, mean_latency_per_gen_token = analyse_results(PROFILING_DATA_DIR, request_rates, repeat_num, num_requests, prompt_len, gen_len)
        # dump_latency_per_token(mean_latency_per_prompt_token, mean_latency_per_gen_token, request_rates, prompt_len, gen_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/model/vicuna-7b-v1.3/")
    parser.add_argument("--tokenizer", type=str, default="/data/model/vicuna-7b-v1.3/")
    parser.add_argument("--backend", type=str, choices=["vllm"], default="vllm")
    parser.add_argument("--dataset", type=str,
        default="/data/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--vllm_dir", type=str, default="/home/lily/vllm_bench_jerry/experimentation-for-cllam")
    parser.add_argument("--outfile", type=str, default="bench_results")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--num_requests", type=int, default=500)
    parser.add_argument("--repeat_num", type=int, default=3)
    parser.add_argument("--request_rate_params", type=tuple, help="(start_request_rate, end_request_rate, step_size). End_request_size is INCLUDED.", default=(2, 14, 2))
    args = parser.parse_args()
    main(args.vllm_dir, args.model, args.tokenizer, args.backend, args.dataset, args.outfile, args.prompt_len, args.gen_len, args.num_requests, args.repeat_num, args.request_rate_params)