# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the Python libraries ([DeepEP](https://github.com/deepseek-ai/DeepEP)), including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

Step 2 is necessary for multi-node deployment.

All scripts accept a positional argument as workspace path for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

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

## HybridEP for GB200 NVL72

Standard DeepEP crashes on GB200 (aarch64 incompatible). The `hybrid-ep` branch
of DeepEP provides GB200-compatible high-throughput kernels optimized for
NVLink-only topologies like NVL72.

**Install:**

```bash
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh --deepep-branch hybrid-ep
```

**Run:**

```bash
vllm serve <model> \
  --enable-expert-parallel \
  --all2all-backend hybrid_ep \
  --data-parallel-size <N>
```

**Limitations:**

- High throughput only (prefill). Not suitable for latency-sensitive decode.
- For decode on GB200, use `--all2all-backend allgather_reducescatter`
  (optionally with `VLLM_USE_NCCL_SYMM_MEM=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1`)
  or `--all2all-backend flashinfer_nvlink_one_sided`.

**Validate:** Check the server logs for `Using HybridEPAll2AllManager` to confirm
the backend is active. Compare throughput against `allgather_reducescatter`
at concurrency >= 256.
