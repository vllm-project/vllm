# Troubleshooting distributed deployments

For general troubleshooting, see [Troubleshooting](../usage/troubleshooting.md).

## Verify inter-node GPU communication

After you start the Ray cluster, verify GPU-to-GPU communication across nodes. Proper configuration can be non-trivial. For more information, see [troubleshooting script][troubleshooting-incorrect-hardware-driver]. If you need additional environment variables for communication configuration, append them to <gh-file:examples/online_serving/run_cluster.sh>, for example `-e NCCL_SOCKET_IFNAME=eth0`. Setting environment variables during cluster creation is recommended because the variables propagate to all nodes. In contrast, setting environment variables in the shell affects only the local node. For more information, see <gh-issue:6803>.

## No available node types can fulfill resource request

The error message `Error: No available node types can fulfill resource request` can appear even when the cluster has enough GPUs. The issue often occurs when nodes have multiple IP addresses and vLLM can't select the correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP` in <gh-file:examples/online_serving/run_cluster.sh> (with a different value on each node). Use `ray status` and `ray list nodes` to verify the chosen IP address. For more information, see <gh-issue:7815>.

## Ray observability

Debugging a distributed system can be challenging due to the large scale and complexity. Ray provides a suite of tools to help monitor, debug, and optimize Ray applications and clusters. For more information about Ray observability, visit the [official Ray observability docs](https://docs.ray.io/en/latest/ray-observability/index.html). For more information about debugging Ray applications, visit the [Ray Debugging Guide](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html). For information about troubleshooting Kubernetes clusters, see the
[official KubeRay troubleshooting guide](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html).
