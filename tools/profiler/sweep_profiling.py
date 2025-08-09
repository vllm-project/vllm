# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import signal
import subprocess
import time
from datetime import timedelta

# Create the argument parser
parser = argparse.ArgumentParser(
    description="Run vLLM profiling with different batch sizes")
parser.add_argument(
    '--model',
    type=str,
    default="meta-llama/Llama-3.2-3B",
    help='Model name to use for profiling',
)
parser.add_argument(
    '--max-tokens',
    type=int,
    default=int(1e5),
    help='Maximum number of batched tokens for GPU (default: 100000)',
)
parser.add_argument(
    '--tensor-parallel-size',
    type=int,
    default=1,
    help=('Tensor parallelism degree '
          '(number of GPUs to split model across)'),
)

args = parser.parse_args()
model_name = args.model
MAX_NUM_BATCHED_TOKENS_GPU = args.max_tokens
TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
batch_sizes = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
start_time = time.time()
print("Note that --prompt-len and --json are DUMMY flags in this file, "
      "the inputs are handled in profiling.py.")
print(f"WARNING: MAX_NUM_BATCHED_TOKENS_GPU is "
      f"set to {MAX_NUM_BATCHED_TOKENS_GPU}. "
      f"This may not fit on a smaller GPU!")
print(f"INFO: The model being used is {model_name}")
print(f"INFO: Using tensor parallel size of {TENSOR_PARALLEL_SIZE}")
# Process each batch size
for batch_size in batch_sizes:
    print(f"\n=== Starting profiling for batch_size = {batch_size} ===")
    max_num_batched_tokens = MAX_NUM_BATCHED_TOKENS_GPU

    command = [
        "VLLM_USE_V1=0",
        "python",
        "profiling.py",
        "--model",
        model_name,
        "--batch-size",
        str(batch_size),
        "--prompt-len",
        str(0),  # dummy value, handled in profiling.py
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--json",
        "DUMMY_ARG.json",  # dummy, handled inside profiling.py
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "run_num_steps",
        "-n",
        "2"
    ]

    cmd_string = " ".join(command)
    print(f"\nRunning: {cmd_string}")

    try:
        # Start the process with shell=True
        # since we're using environment variables
        process = subprocess.Popen(cmd_string, shell=True)
        return_code = process.wait()
        if return_code == 0:
            print(f"Successfully completed: batch_size={batch_size}")
        else:
            print(f"Command failed with exit code {return_code}")
            print("Encountered an error. Moving to next batch size.")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error traceback:\n{error_details}")
        with open('profiling_errors.log', 'a') as log_file:
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                           f"Error with batch_size={batch_size}: {str(e)}\n")
            log_file.write(f"Traceback:\n{error_details}\n\n")
        print("Encountered an exception. Moving to next batch size.")
    finally:
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.5)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError as e:
                print(f"Warning when terminating process: {e}")

end_time = time.time()
total_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(total_time)))
print("\n=== Profiling Complete ===")
print(f"Total execution time: {formatted_time}")
