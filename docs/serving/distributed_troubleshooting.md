# Troubleshooting distributed deployments

For general troubleshooting, see [Troubleshooting](../usage/troubleshooting.md).

## Verify inter-node GPU communication

After you start the Ray cluster, verify GPU-to-GPU communication across nodes. Proper configuration can be non-trivial. For more information, see [troubleshooting script](../usage/troubleshooting.md#incorrect-hardwaredriver). If you need additional environment variables for communication configuration, append them to [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh), for example `-e NCCL_SOCKET_IFNAME=eth0`. Setting environment variables during cluster creation is recommended because the variables propagate to all nodes. In contrast, setting environment variables in the shell affects only the local node. For more information, see <https://github.com/vllm-project/vllm/issues/6803>.

## No available node types can fulfill resource request

The error message `Error: No available node types can fulfill resource request` can appear even when the cluster has enough GPUs. The issue often occurs when nodes have multiple IP addresses and vLLM can't select the correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) (with a different value on each node). Use `ray status` and `ray list nodes` to verify the chosen IP address. For more information, see <https://github.com/vllm-project/vllm/issues/7815>.

## Ray observability

Debugging a distributed system can be challenging due to the large scale and complexity. Ray provides a suite of tools to help monitor, debug, and optimize Ray applications and clusters. For more information about Ray observability, visit the [official Ray observability docs](https://docs.ray.io/en/latest/ray-observability/index.html). For more information about debugging Ray applications, visit the [Ray Debugging Guide](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html). For information about troubleshooting Kubernetes clusters, see the
[official KubeRay troubleshooting guide](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html).

## High latency or decode collapse in Docker with Pipeline Parallelism (PP)

When running pipeline-parallel inference (`pipeline_parallel_size > 1`) inside a Docker container, you may observe severe decode throughput degradation on long contexts while GPU utilization remains low (e.g. 10–20%).

- **Symptom:** Profiling reveals high latency on `user_annotation gloo:send` (up to ~500ms per send during decode steps).
- **Cause:** PyTorch's `gloo` CPU control plane resolves the container hostname to the Docker bridge IP (e.g. `172.17.0.2`) instead of the loopback interface (`lo`). Control messages (which grow with context length due to KV block table metadata) traverse the Docker bridge network stack instead of communicating over loopback.
- **Solution:** Set `GLOO_SOCKET_IFNAME=lo` inside the container or pass `-e GLOO_SOCKET_IFNAME=lo` to `docker run`. For multi-node deployments across separate hosts, ensure `GLOO_SOCKET_IFNAME` is set to the physical network interface (e.g., `eth0`) rather than `lo`.
