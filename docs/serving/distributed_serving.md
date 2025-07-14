# Distributed inference and serving

## Distributed inference strategies for a single model replica

To choose a distributed inference strategy for a single model replica, use the following guidelines:

- **Single GPU (no distributed inference):** if the model fits on a single GPU, distributed inference is probably unnecessary. Run inference on that GPU.
- **Single-node multi-GPU (tensor parallel inference):** if the model is too large for a single GPU but fits on a single node with multiple GPUs, use tensor parallelism. Set `tensor_parallel_size` to the number of GPUs (for example, 4 for a 4 GPU node).
- **Multi-node multi-GPU (tensor parallel plus pipeline parallel inference):** if the model is too large for a single node, combine tensor parallelism with pipeline parallelism. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes (for example, 8 GPUs per node and 2 nodes: `tensor_parallel_size=8`, `pipeline_parallel_size=2`).

Increase the number of GPUs and nodes until there is enough GPU memory for the model. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.

After you provision sufficient resources to fit the model, run `vllm`. Look for a log message similar to `# GPU blocks: 790`. Multiply that number by `16` (the block size) to estimate the maximum number of tokens the configuration can serve. If this estimate is less than your throughput requirements, increase the number of GPUs in your cluster.

!!! note
    Edge case: if the model fits within a single node, but the GPU count does not evenly divide the model size, enable pipeline parallelism, which splits the model along layers and supports uneven splits. In this scenario, set `tensor_parallel_size=1` and `pipeline_parallel_size` to the number of GPUs.

### Distributed serving of MoE (Mixture of Experts) models

It is often advantageous to exploit the inherent parallelism of experts by using a separate parallelism strategy for the expert layers. vLLM supports large-scale deployment combining Data Parallel attention with Expert or Tensor Parallel MoE layers. See the page on [Data Parallel Deployment](data_parallel_deployment.md) for more information.

## Single node deployment

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. The current implementation includes [Megatron-LM's tensor parallel algorithm](https://arxiv.org/pdf/1909.08053.pdf).

vLLM's default distributed runtimes are [Ray](https://github.com/ray-project/ray) for multi-node inference and Python's native `multiprocessing` for single-node inference. You can override the defaults by setting `distributed_executor_backend` in the `LLM` class or `--distributed-executor-backend` in the API server. Use `mp` for `multiprocessing` or `ray` for Ray.

Set `tensor_parallel_size` in the `LLM` class to the desired GPU count for multi-GPU inference. For example, to run inference on 4 GPUs:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

For multi-GPU serving, include `--tensor-parallel-size` when starting the server. For example, to run the API server on 4 GPUs:

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

Enable pipeline parallelism by adding `--pipeline-parallel-size`. For example, to run the API server on 8 GPUs with pipeline parallelism and tensor parallelism:

```bash
# Eight GPUs total
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## Multi-node deployment

If a single node lacks sufficient GPUs to hold the model, deploy vLLM across multiple nodes. Multi-node deployments require Ray as the runtime engine. Ensure that every node provides an identical execution environment, including the model path and Python packages. Using container images is recommended because they provide a convenient way to keep environments consistent and to hide host heterogeneity.

### Ray cluster setup with containers

The helper script `<gh-file:examples/online_serving/run_cluster.sh>` starts containers across nodes and initializes Ray. By default, the script runs Docker without administrative privileges, which prevents access to the GPU performance counters when profiling or tracing. Add the `--cap-add=CAP_SYS_ADMIN` flag to the Docker command to enable those capabilities.

Choose one node as the head node and run:

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <head_node_ip> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<head_node_ip>
```

On each worker node, run:

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <head_node_ip> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<head_node_ip>
```

Note that `VLLM_HOST_IP` is unique for each worker. Keep the shells running these commands open; closing any shell terminates the cluster. Ensure that all nodes can communicate with each other through their IP addresses.

!!! warning
    For security, set `VLLM_HOST_IP` to an address on a private network segment. Traffic sent over this network is unencrypted, and the endpoints exchange data in a format that can be exploited to execute arbitrary code if an adversary gains network access. Ensure that untrusted parties cannot reach the network.

From any node, enter a container and run `ray status` and `ray list nodes` to verify that Ray finds the expected number of nodes and GPUs.

!!! warning
    Alternatively, set up the Ray cluster using KubeRay. See the [KubeRay vLLM documentation](https://docs.ray.io/en/latest/cluster/kubernetes/examples/vllm-rayservice.html) for details.

### Running vLLM on a Ray cluster

!!! warning
     If Ray is running inside containers, run the commands in the remainder of this guide *inside the containers*, not the host. To open a shell inside a container, connect to a node and use `docker exec -it <container_name> /bin/bash`.

Once a Ray cluster is running, use vLLM as you would in a single-node setting. All resources across the Ray cluster are visible to vLLM, so a single `vllm` command on a single node is sufficient.

The common practice is to set the tensor parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes. For example, if you have 16 GPUs across 2 nodes (8 GPUs per node), set the tensor parallel size to 8 and the pipeline parallel size to 2:

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 8 \
     --pipeline-parallel-size 2
```

Alternatively, you can simply set `tensor_parallel_size` to the total number of GPUs in the cluster:

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16
```

## Troubleshooting distributed deployments

To make tensor parallelism performant, ensure that communication between nodes is efficient, for example, by using high-speed network cards such as InfiniBand. To set up the cluster to use InfiniBand, append additional arguments like `--privileged -e NCCL_IB_HCA=mlx5` to the `run_cluster.sh` script. Contact your system administrator for more information about the required flags. One way to confirm if InfiniBand is working is to run `vllm` with the `NCCL_DEBUG=TRACE` environment variable set, for example `NCCL_DEBUG=TRACE vllm serve ...`, and check the logs for the NCCL version and the network used. If you find `[send] via NET/Socket` in the logs, NCCL uses a raw TCP socket, which is not efficient for cross-node tensor parallelism. If you find `[send] via NET/IB/GDRDMA` in the logs, NCCL uses InfiniBand with GPUDirect RDMA, which is efficient.

## Enabling GPUDirect RDMA

To enable GPUDirect RDMA with vLLM, configure the following settings:

- `IPC_LOCK` security context: add the `IPC_LOCK` capability to the container's security context to lock memory pages and prevent swapping to disk.
- Shared memory with `/dev/shm`: mount `/dev/shm` in the pod spec to provide shared memory for interprocess communication (IPC).

If you use Docker, set up the container as follows:

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

If you use Kubernetes, set up the pod spec as follows:

```yaml
...
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
...
```

Efficient tensor parallelism requires fast inter-node communication, preferably through high-speed network adapters such as InfiniBand. To enable InfiniBand, append flags such as `--privileged -e NCCL_IB_HCA=mlx5` to `run_cluster.sh`. Consult your system administrator for cluster-specific settings.

Confirm InfiniBand operation by enabling detailed NCCL logs:

```bash
NCCL_DEBUG=TRACE vllm serve ...
```

Search the logs for the transport method. Entries containing `[send] via NET/Socket` indicate raw TCP sockets, which perform poorly for cross-node tensor parallelism. Entries containing `[send] via NET/IB/GDRDMA` indicate InfiniBand with GPUDirect RDMA, which provides high performance.

!!! warning
    After you start the Ray cluster, verify GPU-to-GPU communication across nodes. Proper configuration can be non-trivial. Refer to the [sanity check script][troubleshooting-incorrect-hardware-driver] for details. If you need additional environment variables for communication configuration, append them to `run_cluster.sh`, for example `-e NCCL_SOCKET_IFNAME=eth0`. Setting environment variables during cluster creation is recommended because the variables propagate to all nodes. In contrast, setting environment variables in the shell affects only the local node. See <gh-issue:6803> for more information.

!!! warning
    If you use Hugging Face models, downloading the model before starting vLLM is recommended. Download the model on every node to the same path, or store the model on a distributed file system accessible by all nodes. Then pass the path to the model in place of the repository ID. Otherwise, supply a Hugging Face token by appending `-e HF_TOKEN=<token>` to `run_cluster.sh`.

!!! warning
    The error message `Error: No available node types can fulfill resource request` can appear even when the cluster has enough GPUs. The issue often occurs when nodes have multiple IP addresses and vLLM cannot select the correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in `run_cluster.sh` (with a different value on each node). Use `ray status` and `ray list nodes` to verify the chosen IP address. See <gh-issue:7815> for more information.
