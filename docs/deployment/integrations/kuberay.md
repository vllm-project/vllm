# KubeRay

[KubeRay](https://github.com/ray-project/kuberay) provides a Kubernetes-native way to run vLLM workloads on Ray clusters.
A Ray cluster can be declared in YAML, and the operator then handles pod scheduling, networking configuration, restarts, and blue-green deployments — all while preserving the familiar Kubernetes experience.

## Why KubeRay instead of manual scripts?

| Feature | Manual scripts | KubeRay |
|---------|-----------------------------------------------------------|---------|
| Cluster bootstrap | Manually SSH into every node and run a script | One command to create or update the whole cluster: `kubectl apply -f cluster.yaml` |
| Autoscaling | Manual | Automatically patches CRDs for adjusting cluster size |
| Upgrades | Tear down & re-create manually | Blue/green deployment updates supported |
| Declarative config | Bash flags & environment variables | Git-ops-friendly YAML CRDs (RayCluster/RayService) |

Using KubeRay reduces the operational burden and simplifies integration of Ray + vLLM with existing Kubernetes workflows (CI/CD, secrets, storage classes, etc.).

## Learn more

* ["Serve a Large Language Model using Ray Serve LLM on Kubernetes"](https://docs.ray.io/en/master/cluster/kubernetes/examples/rayserve-llm-example.html) - An end-to-end example of how to serve a model using vLLM, KubeRay, and Ray Serve.
* [KubeRay documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
