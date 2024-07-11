import os
import time
import json
import subprocess
from datetime import datetime

SMALL_MODELS = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-2.7b"]
LARGE_MODELS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", 
                "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"]

LARGE_MODELS = []
MODELS = SMALL_MODELS + LARGE_MODELS

NUM_PROMPTS = 100
NUM_GPUS = [1]
BATCH_SIZES = [10]
MAX_OUTPUT_LEN = 100
OUTPUT_PATH = "/home/mcw/common/results/test_2"
DATASET_PATH = "/home/mcw/common/learning/testing_data/prompts.json"
DATASET_PATH = "/home/mcw/thrisha/data/ShareGPT_V3_unfiltered_cleaned_split.json"

SCRIPT_PATH = "/home/mcw/vllm/benchmarks/benchmark_throughput.py"

TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
OUTPUT_PATH = os.path.join(OUTPUT_PATH, TIMESTAMP)
os.makedirs(OUTPUT_PATH, exist_ok=True)

def run_subprocess_realtime(cmd: list) -> int:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return process.returncode

def main():
    for model in MODELS:
        for tp in NUM_GPUS:
            if tp == 1 and model in LARGE_MODELS:
                print("Model cannot fit into a single GPU..")
                continue
            
            for batch_size in BATCH_SIZES:
                max_output_tokens = MAX_OUTPUT_LEN
                model_string = model.replace('/', '_').replace('.', '_').replace('-', '_')
                experiment_name = f"{model_string}_gpu{tp}_bs{batch_size}_o{max_output_tokens}"
                output_json = os.path.join(OUTPUT_PATH, f"{experiment_name}.json")
                
                try:
                    command = ["python3", SCRIPT_PATH, \
                              f"--dataset", DATASET_PATH, \
                              f"--num-prompts", f"{NUM_PROMPTS}",\
                              f"--model", model,
                              f"--max-num-seqs", f"{batch_size}", \
                              f"--tensor-parallel-size", f"{tp}",\
                              f"--output-json", output_json,
                              f"--output-len", f"{max_output_tokens}"]
                    
                    print(f"Executing command {' '.join(command)}")
                    run_subprocess_realtime(command)
                
                    if os.path.exists(output_json):
                        with open(output_json, 'r') as f:
                            results = json.load(f)
                        
                        results['model_name'] = experiment_name
                        results['num_gpus'] = tp
                        results['max_output_len'] = max_output_tokens
                        
                        with open(output_json, 'w') as f:
                            json.dump(results, f, indent=4)

                except Exception as e:
                    print(f"FAILED [[{experiment_name}]] : {e}")
                
                print(f"[{experiment_name}] Done.")
                
    print("Done")
                        
if __name__ == "__main__":
    main()