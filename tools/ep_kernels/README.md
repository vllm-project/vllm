Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:
1. Build and install the Python libraries (both [pplx-kernels](https://github.com/ppl-ai/pplx-kernels) and [DeepEP](https://github.com/deepseek-ai/DeepEP)), including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

2 is necessary for multi-node deployment.

All scripts accept a positional argument as workspace path for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

# Usage

## Single-node

```bash
bash install_python_libraries.sh
```

## Multi-node

```bash
bash install_python_libraries.sh
sudo bash configure_system_drivers.sh
sudo reboot # Reboot is required to load the new driver
```
