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

### Example: Serve an LLM using vLLM on KubeRay

This guide shows how to serve a large language model on Kubernetes with vLLM and KubeRay. The example deploys the `meta-llama/Meta-Llama-3-8B-Instruct` model from Hugging Face and runs on Google Kubernetes Engine (GKE) or a local Kubernetes cluster.


#### Prerequisites

* A Hugging Face account and an access token with read permission for gated
  repositories.
* Access to the `meta-llama/Meta-Llama-3-8B-Instruct` model on Hugging Face.
* Optional: the Google Cloud SDK (`gcloud`) and `kubectl` [authenticated against the target
  project](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#store_info).

#### Create a GPU-enabled GKE cluster

```bash
gcloud container clusters create kuberay-gpu-cluster \
  --machine-type=g2-standard-24 \
  --location=us-east4-c \
  --num-nodes=2 \
  --accelerator=type=nvidia-l4,count=2,gpu-driver-version=latest
```

Each model replica consumes two NVIDIA L4 GPUs through tensor parallelism.

Alternatively, set up a two-GPU on-premises Kubernetes cluster before continuing.

#### Install the KubeRay operator

Install the latest stable KubeRay operator by following the 
[official installation guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html).

Note that the KubeRay operator must be installed on the CPU node in order for
the CRD to taint the GPU node pool correctly.

#### Store the Hugging Face token as a Kubernetes secret

```bash
export HF_TOKEN="<hugging-face-access-token>"

kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=${HF_TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -
```

#### Deploy a `RayService` that serves the model

[This YAML](https://github.com/ray-project/kuberay/blob/564524cd0690cc4e389fd9e484959626fa15b0ce/ray-operator/config/samples/vllm/ray-service.vllm.yaml) defines a minimal Ray cluster and Ray Serve application that serves the LLama3 8B model using vLLM.
Create the cluster with the YAML manifest:


```bash
kubectl apply -f https://github.com/ray-project/kuberay/blob/564524cd0690cc4e389fd9e484959626fa15b0ce/ray-operator/config/samples/vllm/ray-service.vllm.yaml
```

Wait until the `RayService` reports **RUNNING** and **HEALTHY** before continuing:

```bash
kubectl get rayservice llama-3-8b -o yaml
```


#### Expose the HTTP endpoint and send a prompt

Establish a port-fowarding session for the Serve app:

```bash
kubectl port-forward svc/llama-3-8b-serve-svc 8000
```

KubeRay creates the Kubernetes Service only after all Pods and Ray Serve applications start. The process can take a few minutes.

Now send a prompt to the model:

```bash
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

The response contains an assistant message similar to the following:

```json
{"id":"cmpl-ce6585cd69ed47638b36ddc87930fded","object":"chat.completion","created":1723161873,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The Ray open-source project is a high-performance distributed computing framework that allows users to scale Python applications and machine learning models to thousands of nodes, supporting distributed data processing, distributed machine learning, and distributed analytics."},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":32,"total_tokens":74,"completion_tokens":42}}
```

---

## Learn more

* [KubeRay documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)