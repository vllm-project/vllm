# Kthena

[**Kthena**](https://github.com/volcano-sh/kthena) is a Kubernetes-native LLM inference platform that transforms how organizations deploy and manage Large Language Models in production. Built with declarative model lifecycle management and intelligent request routing, it provides high performance and enterprise-grade scalability for LLM inference workloads.

This guide shows how to deploy a production-grade, **multi-node vLLM** service on Kubernetes.

We’ll:

- Install the required components (Kthena + Volcano).
- Deploy a multi-node vLLM model via Kthena’s `ModelServing` CR.
- Validate the deployment.

---

## 1. Prerequisites

You need:

- A Kubernetes cluster with **GPU nodes**.
- `kubectl` access with cluster-admin or equivalent permissions.
- **Volcano** installed for gang scheduling.
- **Kthena** installed with the `ModelServing` CRD available.
- A valid **Hugging Face token** if loading models from Hugging Face Hub.

### 1.1 Install Volcano

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

This provides the gang-scheduling and network topology features used by Kthena.

### 1.2 Install Kthena

```bash
helm install kthena oci://ghcr.io/volcano-sh/charts/kthena --version v0.1.0 --namespace kthena-system --create-namespace
```

- The `kthena-system` namespace is created.
- Kthena controllers and CRDs, including `ModelServing`, are installed and healthy.

Validate:

```bash
kubectl get crd | grep modelserving
```

You should see:

```text
modelservings.workload.serving.volcano.sh   ...
```

---

## 2. The Multi-Node vLLM `ModelServing` Example

Kthena provides an example manifest to deploy a **multi-node vLLM cluster running Llama**. Conceptually this is equivalent to the vLLM production stack Helm deployment, but expressed with `ModelServing`.

A simplified version of the example (`llama-multinode`) looks like:

- `spec.replicas: 1` – one `ServingGroup` (one logical model deployment).
- `roles`:
    - `entryTemplate` – defines **leader** pods that run:
        - vLLM’s **multi-node cluster bootstrap script** (Ray cluster).
        - vLLM **OpenAI-compatible API server**.
    - `workerTemplate` – defines **worker** pods that join the leader’s Ray cluster.

Key points from the example YAML:

- **Image**: `vllm/vllm-openai:latest` (matches upstream vLLM images).
- **Command** (leader):

  ```yaml
  command:
    - sh
    - -c
    - >
      bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=2;
      python3 -m vllm.entrypoints.openai.api_server
        --port 8080
        --model meta-llama/Llama-3.1-405B-Instruct
        --tensor-parallel-size 8
        --pipeline-parallel-size 2
  ```

- **Command** (worker):

  ```yaml
  command:
    - sh
    - -c
    - >
      bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker --ray_address=$(ENTRY_ADDRESS)
  ```

---

## 3. Deploying Multi-Node llama vLLM via Kthena

### 3.1 Prepare the Manifest

**Recommended**: use a Secret instead of a raw env var:

```bash
kubectl create secret generic hf-token \
  -n default \
  --from-literal=HUGGING_FACE_HUB_TOKEN='<your-token>'
```

### 3.2 Apply the `ModelServing`

```bash
cat  <<EOF | kubectl apply -f -
apiVersion: workload.serving.volcano.sh/v1alpha1
kind: ModelServing
metadata:
  name: llama-multinode
  namespace: default
spec:
  schedulerName: volcano
  replicas: 1  # group replicas
  template:
    restartGracePeriodSeconds: 60
    gangPolicy:
      minRoleReplicas:
        405b: 1
    roles:
      - name: 405b
        replicas: 2
        entryTemplate:
          spec:
            containers:
              - name: leader
                image: vllm/vllm-openai:latest
                env:
                  - name: HUGGING_FACE_HUB_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: hf-token
                        key: HUGGING_FACE_HUB_TOKEN
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=2; 
                    python3 -m vllm.entrypoints.openai.api_server --port 8080 --model meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                ports:
                  - containerPort: 8080
                readinessProbe:
                  tcpSocket:
                    port: 8080
                  initialDelaySeconds: 15
                  periodSeconds: 10
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
        workerReplicas: 1
        workerTemplate:
          spec:
            containers:
              - name: worker
                image: vllm/vllm-openai:latest
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker --ray_address=$(ENTRY_ADDRESS)"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                env:
                  - name: HUGGING_FACE_HUB_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: hf-token
                        key: HUGGING_FACE_HUB_TOKEN
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm   
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
EOF
```

Kthena will:

- Create a `ModelServing` object.
- Derive a `PodGroup` for Volcano gang scheduling.
- Create the leader and worker pods for each `ServingGroup` and `Role`.

---

## 4. Verifying the Deployment

### 4.1 Check ModelServing Status

Use the snippet from the Kthena docs:

```bash
kubectl get modelserving -oyaml | grep status -A 10
```

You should see something like:

```yaml
status:
  availableReplicas: 1
  conditions:
    - type: Available
      status: "True"
      reason: AllGroupsReady
      message: All Serving groups are ready
    - type: Progressing
      status: "False"
      ...
  replicas: 1
  updatedReplicas: 1
```

### 4.2 Check Pods

List pods for your deployment:

```bash
kubectl get pod -owide -l modelserving.volcano.sh/name=llama-multinode
```

Example output (from docs):

```text
NAMESPACE   NAME                          READY   STATUS    RESTARTS   AGE   IP            NODE           ...
default     llama-multinode-0-405b-0-0    1/1     Running   0          15m   10.244.0.56   192.168.5.12   ...
default     llama-multinode-0-405b-0-1    1/1     Running   0          15m   10.244.0.58   192.168.5.43   ...
default     llama-multinode-0-405b-1-0    1/1     Running   0          15m   10.244.0.57   192.168.5.58   ...
default     llama-multinode-0-405b-1-1    1/1     Running   0          15m   10.244.0.53   192.168.5.36   ...
```

Pod name pattern:

- `llama-multinode-<group-idx>-<role-name>-<replica-idx>-<ordinal>`.

The first number indicates `ServingGroup`. The second (`405b`) is the `Role`. The remaining indices identify the pod within the role.

---

## 6. Accessing the vLLM OpenAI-Compatible API

Expose the entry via a Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-multinode-openai
  namespace: default
spec:
  selector:
    modelserving.volcano.sh/name: llama-multinode
    modelserving.volcano.sh/entry: "true"
    # optionally further narrow to leader role if you label it
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

Port-forward from your local machine:

```bash
kubectl port-forward svc/llama-multinode-openai 30080:80 -n default
```

Then:

- List models:

  ```bash
  curl -s http://localhost:30080/v1/models
  ```

- Send a completion request (mirroring vLLM production stack docs):

  ```bash
  curl -X POST http://localhost:30080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-405B-Instruct",
      "prompt": "Once upon a time,",
      "max_tokens": 10
    }'
  ```

You should see an OpenAI-style response from vLLM.

---

## 7. Clean Up

To remove the deployment and its resources:

```bash
kubectl delete modelserving llama-multinode -n default
```

If you’re done with the entire stack:

```bash
helm uninstall kthena -n kthena-system   # or your Kthena release name
helm uninstall volcano -n volcano-system
```
