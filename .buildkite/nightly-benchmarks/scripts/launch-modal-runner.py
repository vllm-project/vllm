import modal
import subprocess
import os
import sys

# Env vars used for constructing modal image
BUILDKITE_COMMIT = os.getenv("BUILDKITE_COMMIT")
GPU = os.getenv("GPU")
CMD = os.getenv("CMD")

# Local env vars from buildkite job that we'll want to drill through to GPU container
PASSTHROUGH_ENV_VARS = {
    "BUILDKITE_COMMIT": BUILDKITE_COMMIT,
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "VLLM_USAGE_SOURCE": os.getenv("VLLM_USAGE_SOURCE"),
}

# Modal app and associated volumes
app = modal.App("buildkite-runner")
hf_cache = modal.Volume.from_name("vllm-benchmark-hf-cache", create_if_missing=True)

# Image to use for runner
BASE_IMG = f"public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:{BUILDKITE_COMMIT}"
image = (
    modal.Image.from_registry(BASE_IMG, add_python="3.12")
    .workdir("/vllm-workspace")
    .env(PASSTHROUGH_ENV_VARS)
)

# Remote function to run in the container
HOURS = 60 * 60
@app.function(
    image=image,
    gpu=GPU, # GPU can be "A100", "A10G", "H100", "T4"
    timeout=4 * HOURS,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def runner(env: dict, cmd: str = ""):
    # Set passthrough environment variables in remote container
    for k, v in env.items():
        os.environ[k] = v

    # TODO: the vllm-ci-postmerge-repo does not contain the VLLM package itself
    # we need to figure out how to install from source.
    # I tried doing this in the build step but it appears to fail without an underlying GPU
    # So I am doing here on the runner GPU. 
    # The VLLM install still fails, logs: https://gist.github.com/erik-dunteman/f75f0733ac6a78de73d25220a4a3f58a
    print("Installing VLLM from source, commit:", BUILDKITE_COMMIT)
    # Install vllm from source
    subprocess.run(["pip", "install", f"git+https://github.com/vllm-project/vllm.git@{BUILDKITE_COMMIT}"])

    # TODO: remove below env debugging code
    print("\n=== Python Environment Info ===")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path}")
    print("\n=== Installed Packages ===")
    subprocess.run(["pip", "list"])
    print("\n=== VLLM Package Info ===")
    subprocess.run(["pip", "show", "vllm"])
    print("\n=== Working Directory Info ===")
    print(f"Current Directory: {os.getcwd()}")
    subprocess.run(["ls", "-la"])

    # TODO: remove these cmd overrides
    # cmd = "python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct"
    cmd = "nvidia-smi"

    print("\n=== Executing Command ===")
    print(f"Command: {cmd}")
    
    # Execute the command
    subprocess.run(cmd.split(" "))

@app.local_entrypoint()
def main():
    print("Modal client started:")
    required_env_vars = ["BUILDKITE_COMMIT", "GPU", "CMD"]
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            print(f"Missing required environment variable: {env_var}")
            sys.exit(1)
    print(f"\t- Commit:  {BUILDKITE_COMMIT}")
    print(f"\t- GPU:     {GPU}")
    print(f"\t- Command: {CMD}")
    runner.remote(env=PASSTHROUGH_ENV_VARS, cmd=CMD)