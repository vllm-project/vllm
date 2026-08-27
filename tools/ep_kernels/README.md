# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the Python libraries ([DeepEP](https://github.com/deepseek-ai/DeepEP)), including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

Step 2 is necessary for multi-node deployment.

All scripts accept a positional argument as workspace path for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

## NCCL version requirement (CUDA 13+)

DeepEPv2 uses the NCCL GIN (GPU-Initiated Networking) backend, which requires
NCCL >= 2.30.4 at both compile time and runtime. PyTorch 2.11 pins
`nvidia-nccl-cu13==2.28.9` as a transitive dependency, so you need to
override it.

**With uv** (recommended):

```bash
# Create an override file
echo "nvidia-nccl-cu13>=2.30.4" > /tmp/nccl-override.txt
export UV_OVERRIDE=/tmp/nccl-override.txt

# All subsequent uv pip install commands will respect the override
uv pip install vllm
```

**With pip**:

```bash
pip install vllm
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

The override / reinstall must happen before building DeepEP (for GIN device
headers) and must remain in place at runtime. You can verify with:

```bash
python -c "from vllm.utils.import_utils import has_deep_ep_v2; print(has_deep_ep_v2())"
```

## Usage

```bash
# for hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# for blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
```

Additional step for multi-node deployment:

```bash
sudo bash configure_system_drivers.sh # update-initramfs can take several minutes
sudo reboot # Reboot is required to load the new driver
```
