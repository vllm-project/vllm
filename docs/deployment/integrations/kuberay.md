---
title: KubeRay
---
[](){ #deployment-kuberay }

[KubeRay](https://github.com/ray-project/kuberay) provides a Kubernetes-native way to run vLLM workloads on Ray clusters.
A Ray cluster can be declared in YAML, and the operator then handles pod scheduling, networking configuration, restarts, and rolling upgrades—all while preserving the familiar Kubernetes experience.

---

## Why KubeRay instead of manual scripts?

| Feature | Manual scripts | KubeRay |
|---------|-----------------------------------------------------------|---------|
| Cluster bootstrap | Manually SSH into every node and run a script | One command to create or update the whole cluster: `kubectl apply -f cluster.yaml` |
| Fault-tolerance | Nodes must be restarted by hand | Pods are automatically rescheduled; head-node fail-over supported |
| Autoscaling | Unsupported | Native horizontal **and** vertical autoscaling via Ray Autoscaler & Kubernetes HPA |
| Upgrades | Tear down & re-create manually | Rolling updates handled by the operator |
| Monitoring | ad-hoc | Distributed observability with Ray Dashboard |
| Declarative config | Bash flags & environment variables | Git-ops-friendly YAML CRDs (RayCluster/RayService) |

Using KubeRay reduces the operational burden and simplifies integration of Ray + vLLM with existing Kubernetes workflows (CI/CD, secrets, storage classes, etc.).

---

## Quick start

1. Install the KubeRay operator (via Helm or `kubectl apply`).
2. Create a `RayService` that runs vLLM.

```bash
# FIXME create this yaml before merging PR
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/refs/heads/master/ray-operator/config/samples/vllm/ray-service.vllm.yaml
```

The YAML above spins up a Ray cluster and a Ray Serve application that serves the
`meta-llama/Meta-Llama-3-8B-Instruct` model using vLLM. Wait until the
`RayService` reports **RUNNING**, then port-forward and query the model:

```bash
kubectl port-forward svc/llama-3-8b-serve-svc 8000 &

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Provide a brief sentence describing the Ray open-source project."}
        ],
        "temperature": 0.7
      }'
```

---

## Learn more

* ["Serve a Large Language Model with vLLM on Kubernetes"](https://docs.ray.io/en/latest/cluster/kubernetes/examples/vllm-rayservice.html):
  End-to-end walkthrough for deploying Llama-3 8B with `RayService`.
* [KubeRay documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)