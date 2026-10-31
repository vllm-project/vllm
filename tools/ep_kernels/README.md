# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the Python libraries ([DeepEP](https://github.com/deepseek-ai/DeepEP) and [MoonEP](https://github.com/MoonshotAI/MoonEP)), including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

Step 2 is necessary for multi-node deployment.

MoonEP has no releases or PyPI wheels yet, so it is built from source at a
pinned commit (`MOONEP_COMMIT_HASH` / `--moonep-ref`), the same way DeepEP is
handled. It has no NVSHMEM dependency; at runtime it requires NVSwitch
multicast capable GPUs (single-node NVLink symmetric memory).

All scripts accept a positional argument as workspace path for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

## NCCL version requirement for DeepEPv2

DeepEPv2 (`--all2all-backend deepep_v2`) uses the NCCL GIN (GPU-Initiated
Networking) backend, which requires NCCL >= 2.30.4 both when DeepEP is built
and at runtime. PyTorch pins an older release as a dependency (PyTorch 2.13
pins `nvidia-nccl-cu13==2.29.7`), so a plain `pip install` or `uv pip install`
of vLLM is not enough. The vLLM Docker images already override it.

Upgrade NCCL before running `install_python_libraries.sh`. Use
`nvidia-nccl-cu12` on CUDA 12.

**With uv** (recommended):

```bash
# Create an override file
echo "nvidia-nccl-cu13>=2.30.4" > /tmp/nccl-override.txt
export UV_OVERRIDE=/tmp/nccl-override.txt

# All subsequent uv pip install commands will respect the override
uv pip install vllm
```

Keep `UV_OVERRIDE` set for later installs into the same environment. Without
it, any `uv pip install` that resolves PyTorch again restores the pinned NCCL.

**With pip**, or to fix an existing environment:

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

`install_python_libraries.sh` warns when the installed NCCL is too old. If
DeepEP was built against an older NCCL, upgrade NCCL and run the script again.
When NCCL is too old at runtime, vLLM logs the reason at startup and
`deepep_v2` fails with the same message. You can check with:

```bash
python -c "from vllm.utils.import_utils import deep_ep_v2_unavailable_reason; print(deep_ep_v2_unavailable_reason() or 'deepep_v2 is available')"
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
