import os
import subprocess
import json
import time
import shutil

"""
Utility functions to manage the vLLM server for online profiling.
Main functions are setup_server(), start_server(), and kill_server().
"""

def wait_for_server(port: int) -> bool:
    timeout = 1200 # 20 mins
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            subprocess.run(["curl", "-X", "POST", f"localhost:{port}/v1/completions"], check=True)
            return True
        except subprocess.CalledProcessError:
            time.sleep(10) # wait for 10 seconds before retrying
    return False


def kill_gpu_processes(port: int):
    subprocess.run(["ps", "-aux"])
    subprocess.run([f"lsof -t -i:{port} | xargs -r kill -9"], shell=True)

    # Use ps to list all Python processes and grep to exclude the specific one
    command = ["ps", "aux"]
    ps_output = subprocess.check_output(command, text=True)

    # Do not kill this process
    filename = os.path.basename(__file__)
    pids_to_kill = []
    for line in ps_output.split("\n"):
        if "python3" in line and filename not in line:
            pid = line.split()[1]
            pids_to_kill.append(pid)
    
    # Kill other processes
    for pid in pids_to_kill:
        subprocess.run(["kill", "-9", pid])

    # Wait until all GPUs have memory usage < 1000 MB
    if shutil.which("nvidia-smi"):
        while True:
            # Get GPU memory usage for all GPUs
            memory_usage = subprocess.check_output(["nvidia-smi", 
                                                    "--query-gpu=memory.used", 
                                                    "--format=csv,noheader,nounits"], 
                                                    text=True)
            # Split the output into individual GPU memory usage values
            gpu_memory_usage = [int(x) for x in memory_usage.strip().split("\n")]
            # Check if any GPU has memory usage >= 1000 MB
            if all(usage < 1000 for usage in gpu_memory_usage):
                break
            time.sleep(1)
    elif shutil.which("amd-smi"):
        while True:
            memory_usage = subprocess.check_output(["amd-smi", 
                                                    "metric", 
                                                    "-g", 
                                                    "0"], 
                                                    text=True)
            used_vram = int(memory_usage.split("USED_VRAM")[1].split()[0])
            if used_vram < 1000:
                break
            time.sleep(1)

    subprocess.run(["rm", "-rf", "~/.config/vllm"])


def setup_server():
    # install dependencies
    dependencies = ["lsof", "curl", "pgrep"]
    for dep in dependencies:
        if not shutil.which(dep):
            subprocess.run(["apt-get", "update"])
            subprocess.run(["apt-get", "install", "-y", dep])


def start_server(port: int,
                 target_model_dir: str, 
                 spec_config: dict | None, 
                 tp: int, 
                 max_vllm_bs: int,
                 dry_run: bool = False) -> subprocess.Popen | None:
    
    # NOTE: no Prompt Caching, but enabled chunked prefill
    server_command = f"""VLLM_USE_V1=1 vllm serve {target_model_dir} \
                    --disable-log-requests --port {port} \
                    --gpu_memory_utilization 0.95 \
                    --max_num_seqs {max_vllm_bs} \
                    --tensor_parallel_size {tp} \
                    --enable-chunked-prefill \
                    --no-enable-prefix-caching """

    if spec_config:
        speculative_config_json_serialized = json.dumps(spec_config).replace('"', '\\"')
        server_command += f'--speculative_config "{speculative_config_json_serialized}" '
    
    print(f"Server command: {server_command}")

    # start vllm server
    if not dry_run:
        server_process = subprocess.Popen(server_command, shell=True)

        if wait_for_server(port):
            print("vllm server is up and running.")
        else:
            print("vllm failed to start within the timeout period.")
            server_process.kill()

        return server_process
    else:
        return None


def kill_server(port: int, server_process: subprocess.Popen | None):
    if server_process:
        server_process.kill()
    kill_gpu_processes(port)