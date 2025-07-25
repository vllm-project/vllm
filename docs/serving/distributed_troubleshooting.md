# Troubleshooting distributed deployments

## Ray observability

Debugging a distributed system can be challenging due to the large scale and complexity. Ray provides a suite of tools to help monitor, debug, and optimize Ray applications and clusters:

- Ray Dashboard – real-time cluster and application metrics, centralized logs, and granular observability into individual tasks, actors, nodes, and more.
- Ray Tracing (OpenTelemetry) – distributed execution traces for performance bottleneck analysis
- Ray Distributed Debugger – step-through execution of remote tasks with breakpoints, and port-mortem debugging of unhandled exceptions.
- Ray State CLI & State API – programmatic access to jobs, actors, tasks, objects, and node state
- Ray Logs CLI – programmatic access to Ray logs at the task, actor, cluster, node, job, or worker levels.
- Ray Metrics (Prometheus & Grafana) – scrapeable metrics that can be integrated with existing monitoring systems.
- Distributed Profiling – diagnose performance bottlenecks with integrated tools for analyzing CPU, memory, and GPU usage across a cluster.

For more information about Ray observability, visit the [official Ray observability docs](https://docs.ray.io/en/latest/ray-observability/index.html). For more information about debugging Ray applications, visit the [Ray Debugging Guide](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html).

## KubeRay

https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html#serve-multi-node-gpu-troubleshooting

## Optimizing network communication for tensor parallelism

To make tensor parallelism performant, ensure that communication between nodes is efficient, for example, by using high-speed network cards such as InfiniBand. To set up the cluster to use InfiniBand, append additional arguments like `--privileged -e NCCL_IB_HCA=mlx5` to the `run_cluster.sh` script. Contact your system administrator for more information about the required flags. One way to confirm if InfiniBand is working is to run `vllm` with the `NCCL_DEBUG=TRACE` environment variable set, for example `NCCL_DEBUG=TRACE vllm serve ...`, and check the logs for the NCCL version and the network used. If you find `[send] via NET/Socket` in the logs, NCCL uses a raw TCP socket, which is not efficient for cross-node tensor parallelism. If you find `[send] via NET/IB/GDRDMA` in the logs, NCCL uses InfiniBand with GPUDirect RDMA, which is efficient.

## Enabling GPUDirect RDMA

To enable GPUDirect RDMA with vLLM, configure the following settings:

- `IPC_LOCK` security context: add the `IPC_LOCK` capability to the container's security context to lock memory pages and prevent swapping to disk.
- Shared memory with `/dev/shm`: mount `/dev/shm` in the pod spec to provide shared memory for interprocess communication (IPC).
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

Efficient tensor parallelism requires fast inter-node communication, preferably through high-speed network adapters such as InfiniBand. To enable InfiniBand, append flags such as `--privileged -e NCCL_IB_HCA=mlx5` to `run_cluster.sh`. For cluster-specific settings, consult your system administrator.

To confirm InfiniBand operation, enable detailed NCCL logs:

```bash
NCCL_DEBUG=TRACE vllm serve ...
```

Search the logs for the transport method. Entries containing `[send] via NET/Socket` indicate raw TCP sockets, which perform poorly for cross-node tensor parallelism. Entries containing `[send] via NET/IB/GDRDMA` indicate InfiniBand with GPUDirect RDMA, which provides high performance.

!!! tip "Verify inter-node GPU communication"
    After you start the Ray cluster, verify GPU-to-GPU communication across nodes. Proper configuration can be non-trivial. For more information, see [troubleshooting script][troubleshooting-incorrect-hardware-driver]. If you need additional environment variables for communication configuration, append them to `run_cluster.sh`, for example `-e NCCL_SOCKET_IFNAME=eth0`. Setting environment variables during cluster creation is recommended because the variables propagate to all nodes. In contrast, setting environment variables in the shell affects only the local node. For more information, see <gh-issue:6803>.


!!! tip
    The error message `Error: No available node types can fulfill resource request` can appear even when the cluster has enough GPUs. The issue often occurs when nodes have multiple IP addresses and vLLM can't select the correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in `run_cluster.sh` (with a different value on each node). Use `ray status` and `ray list nodes` to verify the chosen IP address. For more information, see <gh-issue:7815>.
