# Using Slurm

This guide shows how to run a vLLM server across several nodes of a Slurm cluster, using an Apptainer (or SingularityCE) container and vLLM's multiprocessing backend. No Ray cluster is needed: Slurm starts one task per node, and the tasks join through `--nnodes`, `--node-rank` and `--master-addr`.

The example scripts are in [examples/deployment/slurm](../../examples/deployment/slurm):

- `serve_multinode.sbatch` reserves the nodes, starts `node.sh` on each one, and waits until the API server answers.
- `node.sh` sets up the environment inside the container and runs `vllm serve` with the node's rank.

## Prepare

1. Pull a vLLM image to a shared filesystem:

    ```bash
    apptainer pull vllm.sif docker://vllm/vllm-openai:latest          # NVIDIA
    apptainer pull vllm.sif docker://vllm/vllm-openai-rocm:latest     # AMD
    ```

2. Download the model into the cache that the job will use. Compute nodes on many clusters have no internet access, so do this on a login node:

    ```bash
    HF_HOME=$WORKDIR/hf_cache hf download <model>
    ```

## Launch

```bash
export SIF=$PWD/vllm.sif MODEL=<model> WORKDIR=$PWD
sbatch --account=<account> --partition=<gpu-partition> --nodes=2 serve_multinode.sbatch
```

By default, tensor parallelism spans the GPUs of one node and pipeline parallelism spans the nodes (`TP=GPUS_PER_NODE`, `PP=nodes`). To use another layout, set `TP` and `PP` so that their product equals the total number of GPUs, for example `TP=16 PP=2` on four 8-GPU nodes.

The API server runs on the first node of the allocation, and the job log prints its address. Other settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPUS_PER_NODE` | `8` | GPUs per node; must match the `--gpus-per-node` request |
| `CONTAINER_CMD` | `apptainer` | Use `singularity` on SingularityCE sites |
| `GPU_FLAG` | empty | `--nv` on NVIDIA; usually empty on AMD (see below) |
| `VLLM_ARGS` | empty | Extra `vllm serve` arguments, e.g. `--max-model-len 8192` |
| `API_PORT`, `MASTER_PORT` | `8000`, `29511` | API and torch.distributed ports on the first node |

## Cluster pitfalls

These are handled by `node.sh`. Each one stops a multi-node launch at startup.

- **Long `TMPDIR`.** vLLM creates ZeroMQ IPC sockets in `TMPDIR`. Unix socket paths are limited to about 107 bytes, and per-job scratch directories are often longer. `node.sh` uses a short directory under `/tmp`.
- **User site packages.** The container sees your home directory. Packages installed with `pip install --user` for the same Python version can shadow the image's packages and fail at import. `node.sh` sets `PYTHONNOUSERSITE=1`.
- **`ROCR_VISIBLE_DEVICES` on AMD.** Slurm can expose AMD GPUs through `ROCR_VISIBLE_DEVICES`. If Ray is installed, importing it raises an error when this variable is set without `HIP_VISIBLE_DEVICES`, and vLLM imports Ray during startup even with the multiprocessing backend. `node.sh` moves the value to `HIP_VISIBLE_DEVICES`.
- **`--rocm` and host libraries.** On AMD, the container runtime's `--rocm` flag binds host ROCm libraries into the image. When the host libraries need a newer glibc than the image provides, the import fails with `GLIBC_x.yy not found`. GPU devices are usually visible without the flag.
- **Caches in `$HOME`.** Triton writes compiled kernels to `~/.triton` unless `TRITON_CACHE_DIR` is set, regardless of `VLLM_CACHE_ROOT`. Home directories on clusters are often small, and a home directory that links elsewhere breaks inside the container if the link target is not bound. `node.sh` puts the vLLM cache under `$WORKDIR/cache` and the Triton cache in node-local `/tmp`.
- **Triton cache on a shared filesystem.** With a cold cache, workers on several nodes compile the same kernels at the same time. On a shared filesystem a worker can read a cache file that another node has not finished writing, and startup fails with `JSONDecodeError` in `triton/compiler/compiler.py`. A node-local `TRITON_CACHE_DIR` avoids this; the cost is recompiling kernels in each job.
- **`--trust-remote-code` models.** Transformers copies the model's custom code into `$HF_HOME/modules` the first time it is loaded. When every worker on every node does this at once with an empty module cache, some workers import the module before it exists (`ModuleNotFoundError: No module named 'transformers_modules...'`). Load the config once on a login node first, for example `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('<model>', trust_remote_code=True)"` with the same `HF_HOME`.
- **Passing variables.** `--env` splits values on commas, so lists such as `NCCL_SOCKET_IFNAME=hsn0,hsn1` break. `node.sh` passes variables through the `APPTAINERENV_` and `SINGULARITYENV_` prefixes instead.
- **Network interfaces.** If NCCL, RCCL or Gloo choose the wrong interface, set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the high-speed interconnect (for example `hsn0,...` on Slingshot or `ib0` on InfiniBand). Ask your site which interfaces and NCCL network plugin to use.

## Startup time

The first start compiles Triton kernels and `torch.compile` graphs, which can take a long time on large clusters. In the tested setup below, a small model took 30 to 46 minutes to serve on 2 or 4 nodes, most of it compilation. Request enough time in `--time`, or add `--enforce-eager` to `VLLM_ARGS` to skip `torch.compile` and graph capture, at some cost in decode speed.

## Tested configuration

The example scripts were tested on LUMI (AMD MI250X, 8 GPUs per node, Slingshot interconnect, SingularityCE 4.1) with the `vllm/vllm-openai-rocm:v0.31.0` image and `Qwen/Qwen3-0.6B`, on 2 nodes (`TP=8 PP=2`) and 4 nodes (`TP=16 PP=2`).
