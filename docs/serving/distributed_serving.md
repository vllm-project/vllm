---
title: Distributed Inference and Serving
---
[](){ #distributed-serving }

## Select a distributed inference strategy for a single model replica

Before exploring distributed inference and serving for a single model, first determine whether distributed inference is required, and if so, which strategy to adopt. The common practice is:

- **Single GPU (no distributed inference)**: When the model fits on a single GPU, distributed inference is probably unnecessary. Run inference on that GPU.
- **Single-Node Multi-GPU (tensor parallel inference)**: When the model is too large for a single GPU but fits on a single node with multiple GPUs, use tensor parallelism. Set `tensor_parallel_size` to the number of GPUs (for example, 4 for a 4-GPU node).
- **Multi-Node Multi-GPU (tensor parallel plus pipeline parallel inference)**: When the model is too large for a single node, combine tensor parallelism with pipeline parallelism. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes (for example, 8 GPUs per node and 2 nodes: `tensor_parallel_size=8`, `pipeline_parallel_size=2`).

Increase the number of GPUs and nodes until the configuration provides enough GPU memory for the model. Set `tensor_parallel_size` to the GPU count per node and `pipeline_parallel_size` to the node count.

After provisioning sufficient resources, run vLLM. Find a log message that looks like `# GPU blocks: 790`. Multiply that number by `16` (the block size) to estimate the maximum number of tokens the configuration can serve. If this estimate is inadequate, increase the number of GPUs or nodes until the block count satisfies the throughput target.

!!! note
    Edge case: When the model fits within a single node but the GPU count does not evenly divide the model size, enable pipeline parallelism, which splits the model along layers and supports uneven splits. In this scenario set `tensor_parallel_size=1` and `pipeline_parallel_size` to the number of GPUs.

## Run vLLM on a single node

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. The current implementation includes [Megatron-LM's tensor parallel algorithm](https://arxiv.org/pdf/1909.08053.pdf).

vLLM's default distributed runtimes are [Ray](https://github.com/ray-project/ray) for multi-node inference and Python's native `multiprocessing` for single-node inference. Override the default by setting `distributed_executor_backend` in the `LLM` class or `--distributed-executor-backend` in the API server using `mp` for multiprocessing or `ray` for Ray.

Set `tensor_parallel_size` in the `LLM` class to the desired GPU count for multi-GPU inference:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

For multi-GPU serving, include `--tensor-parallel-size` when starting the server:

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

Enable pipeline parallelism by adding `--pipeline-parallel-size`:

```bash
# Eight GPUs total
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## Run vLLM on multiple nodes

When a single node lacks sufficient GPUs to hold the model, deploy vLLM across multiple nodes. This requires Ray as the runtime engine. Ensure that every node provides an identical execution environment, including the model path and Python packages. Container images are the recommended pattern here, as they provide a convenient way to keep environments consistent and to hide host heterogeneity.

### Starting a Ray cluster using containers

First, start containers across a Ray cluster. The helper script `<gh-file:examples/online_serving/run_cluster.sh>` initializes Ray across the nodes using containers. By default the script runs Docker without administrative privileges, which prevents access to GPU performance counters when profiling or tracing. Add the `--cap-add=CAP_SYS_ADMIN` flag to the Docker command to enable those capabilities.

Choose one node as the head node and run:

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <head_node_ip> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<this_node_ip>
```

On each worker node, run:

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <head_node_ip> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<this_node_ip>
```

Keep the shells running these commands open; closing a shell terminates the cluster. Ensure that all nodes can communicate to each through through their IP addresses. Set `VLLM_HOST_IP` on each worker to its unique IP address.

!!! warning
    For security, set `VLLM_HOST_IP` to an address on a private network segment. Traffic sent over this network is unencrypted, and the endpoints exchange data in a format that can be exploited to execute arbitrary code if an adversary gains network access. Ensure that untrusted parties cannot reach the network.

!!! warning
    Download the model on every node (to the same path) or store the model on a distributed file system accessible by all nodes.

    When a Hugging Face repository ID is used, supply a Hugging Face token by appending `-e HF_TOKEN=<token>` to `run_cluster.sh`. Downloading the model before starting vLLM is recommended; pass the path to the model instead of the repository ID.

From any node, enter a container and run `ray status` and `ray list nodes` to verify that the Ray cluster sees the expected number of nodes and GPUs.

!!! warning
    Alternatively, set up the Ray cluster using KubeRay. See the [KubeRay vLLM documentation](https://docs.ray.io/en/latest/cluster/kubernetes/examples/vllm-rayservice.html) for details.

### Executing vLLM commands on a running cluster

!!! warning
     If Ray is running inside containers, run the commands in the remainder of this guide *inside the containers*. To open a shell inside a container, connect to a node and use `docker exec -it node /bin/bash`.

Once a Ray cluster is running, you can use vLLM as you would in single-node setting. All the resources across the Ray cluster are visible to vLLM, so a single invocation on a single node is sufficient:

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 8 \
     --pipeline-parallel-size 2
```

The example above assumes 16 GPUs across two nodes (8 GPUs per node). Omitting pipeline parallelism is valid; set `tensor_parallel_size` to the total GPU count in the cluster and vLLM spawns the models across the cluster:

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16
```

### Troubleshooting

Efficient tensor parallelism requires fast inter-node communication, preferably through high-speed network adapters such as InfiniBand. To enable InfiniBand, append flags such as `--privileged -e NCCL_IB_HCA=mlx5` to `run_cluster.sh`. Consult the system administrator for cluster-specific settings.

Confirm InfiniBand operation by enabling detailed NCCL logs:

```bash
NCCL_DEBUG=TRACE vllm serve ...
```

Search the logs for the transport method. Entries containing `[send] via NET/Socket` indicate raw TCP sockets, which perform poorly for cross-node tensor parallelism. Entries containing `[send] via NET/IB/GDRDMA` indicate InfiniBand with GPU-Direct RDMA, which provides high performance.

!!! warning
    After starting the Ray cluster, verify GPU-to-GPU communication across nodes. Configuring it up properly can be non-trivial. Refer to the [sanity check script][troubleshooting-incorrect-hardware-driver] for details. If additional environment variables are required for communication configuration, append them to `run_cluster.sh`, for example `-e NCCL_SOCKET_IFNAME=eth0`. Setting environment variables during cluster creation is recommended because the variables propogate to all nodes. In contrast, setting environment variables in the shell affects only the local node. See <gh-issue:6803> for more information.

!!! warning
    The error message `Error: No available node types can fulfill resource request` can appear even when the cluster has enough GPUs. The issue often occurs when nodes have multiple IP addresses and vLLM cannot select the correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in `run_cluster.sh` (with a different value on each node). Use `ray status` and `ray list nodes` to verify the chosen IP address. See <gh-issue:7815> for more information.
