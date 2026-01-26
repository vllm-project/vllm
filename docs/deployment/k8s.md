# Using Kubernetes

Deploying vLLM on Kubernetes is a scalable and efficient way to serve machine learning models. This guide walks you through deploying vLLM using native Kubernetes.

- [Deployment with CPUs](#deployment-with-cpus)
- [Deployment with GPUs](#deployment-with-gpus)
- [Drain Shutdown](#drain-shutdown)
- [Troubleshooting](#troubleshooting)
    - [Startup Probe or Readiness Probe Failure, container log contains "KeyboardInterrupt: terminated"](#startup-probe-or-readiness-probe-failure-container-log-contains-keyboardinterrupt-terminated)
- [Conclusion](#conclusion)

Alternatively, you can deploy vLLM to Kubernetes using any of the following:

- [Helm](frameworks/helm.md)
- [InftyAI/llmaz](integrations/llmaz.md)
- [llm-d](integrations/llm-d.md)
- [KAITO](integrations/kaito.md)
- [KServe](integrations/kserve.md)
- [Kthena](integrations/kthena.md)
- [KubeRay](integrations/kuberay.md)
- [kubernetes-sigs/lws](frameworks/lws.md)
- [meta-llama/llama-stack](integrations/llamastack.md)
- [substratusai/kubeai](integrations/kubeai.md)
- [vllm-project/aibrix](https://github.com/vllm-project/aibrix)
- [vllm-project/production-stack](integrations/production-stack.md)

## Deployment with CPUs

!!! note
    The use of CPUs here is for demonstration and testing purposes only and its performance will not be on par with GPUs.

First, create a Kubernetes PVC and Secret for downloading and storing Hugging Face model:

??? console "Config"

    ```bash
    cat <<EOF |kubectl apply -f -
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: vllm-models
    spec:
      accessModes:
        - ReadWriteOnce
      volumeMode: Filesystem
      resources:
        requests:
          storage: 50Gi
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: hf-token-secret
    type: Opaque
    stringData:
      token: "REPLACE_WITH_TOKEN"
    EOF
    ```

Here, the `token` field stores your **Hugging Face access token**. For details on how to generate a token,
see the [Hugging Face documentation](https://huggingface.co/docs/hub/en/security-tokens).

Next, start the vLLM server as a Kubernetes Deployment and Service.

Note that you will want to configure your vLLM image based on your processor arch:

??? console "Config"

    ```bash
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest       # use this for x86_64
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:latest # use this for arm64
    cat <<EOF |kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-server
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: vllm
      template:
        metadata:
          labels:
            app.kubernetes.io/name: vllm
        spec:
          containers:
          - name: vllm
            image: $VLLM_IMAGE
            command: ["/bin/sh", "-c"]
            args: [
              "vllm serve meta-llama/Llama-3.2-1B-Instruct"
            ]
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            ports:
              - containerPort: 8000
            volumeMounts:
              - name: llama-storage
                mountPath: /root/.cache/huggingface
          volumes:
          - name: llama-storage
            persistentVolumeClaim:
              claimName: vllm-models
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-server
    spec:
      selector:
        app.kubernetes.io/name: vllm
      ports:
      - protocol: TCP
        port: 8000
        targetPort: 8000
      type: ClusterIP
    EOF
    ```

We can verify that the vLLM server has started successfully via the logs (this might take a couple of minutes to download the model):

```bash
kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Deployment with GPUs

**Pre-requisite**: Ensure that you have a running [Kubernetes cluster with GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).

1. Create a PVC, Secret and Deployment for vLLM

      PVC is used to store the model cache and it is optional, you can use hostPath or other storage options

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ```

      </details>

      Secret is optional and only required for accessing gated models, you can skip this step if you are not using gated models

      ```yaml
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```
  
      Next to create the deployment file for vLLM to run the model server. The following example deploys the `Mistral-7B-Instruct-v0.3` model.

      Here are two examples for using NVIDIA GPU and AMD GPU.

      NVIDIA GPU:

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM needs to access the host's shared memory for tensor parallel inference.
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            containers:
            - name: mistral-7b
              image: vllm/vllm-openai:latest
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  nvidia.com/gpu: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: cache-volume
              - name: shm
                mountPath: /dev/shm
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 5
      ```

      </details>

      AMD GPU:

      You can refer to the `deployment.yaml` below if using AMD ROCm GPU like MI300X.

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            # PVC
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM needs to access the host's shared memory for tensor parallel inference.
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "8Gi"
            hostNetwork: true
            hostIPC: true
            containers:
            - name: mistral-7b
              image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
              securityContext:
                seccompProfile:
                  type: Unconfined
                runAsGroup: 44
                capabilities:
                  add:
                  - SYS_PTRACE
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  amd.com/gpu: "1"
                requests:
                  cpu: "6"
                  memory: 6G
                  amd.com/gpu: "1"
              volumeMounts:
              - name: cache-volume
                mountPath: /root/.cache/huggingface
              - name: shm
                mountPath: /dev/shm
      ```

      </details>

      You can get the full example with steps and sample yaml files from <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve>.

2. Create a Kubernetes Service for vLLM

      Next, create a Kubernetes Service file to expose the `mistral-7b` deployment:

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: Service
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        ports:
        - name: http-mistral-7b
          port: 80
          protocol: TCP
          targetPort: 8000
        # The label selector should match the deployment labels & it is useful for prefix caching feature
        selector:
          app: mistral-7b
        sessionAffinity: None
        type: ClusterIP
      ```

      </details>

3. Deploy and Test

      Apply the deployment and service configurations using `kubectl apply -f <filename>`:

      ```bash
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      To test the deployment, run the following `curl` command:

      ```bash
      curl http://mistral-7b.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "mistralai/Mistral-7B-Instruct-v0.3",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      If the service is correctly deployed, you should receive a response from the vLLM model.

## Drain Shutdown

For production deployments, vLLM supports drain shutdown to enable zero-downtime rolling updates. When enabled, the server drains in-flight requests before terminating instead of abruptly closing connections.

### How It Works

When vLLM receives a `SIGTERM` signal (sent by Kubernetes during pod termination):

1. The server stops accepting new requests (returns `503 Service Unavailable`)
2. The frontend process sends a drain notification to the engine
3. In-flight requests continue processing until completion or timeout
4. If using async KV transfer connectors, pending transfers complete before shutdown
5. The `/live` and `/metrics` endpoints remain accessible during drain

### Configuration

Enable drain shutdown with the following CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--shutdown-mode` | `immediate` | Shutdown mode: `immediate` exits immediately on SIGTERM, `drain` waits for in-flight requests |
| `--shutdown-drain-timeout` | `120` | Seconds to wait for in-flight requests to complete during drain |

### Example Deployment

<details>
<summary>Yaml</summary>

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # should be >= shutdown-drain-timeout to allow drain
      terminationGracePeriodSeconds: 150
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["/bin/sh", "-c"]
        args: [
          "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --shutdown-mode drain --shutdown-drain-timeout 120"
        ]
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 5
        resources:
          limits:
            nvidia.com/gpu: "1"
```

</details>

### Key Considerations

- **`terminationGracePeriodSeconds`**: Set this to at least `drain-timeout` plus a buffer (e.g., 30 seconds) to ensure Kubernetes doesn't force-kill the pod before draining completes.

- **Readiness vs Liveness**: The `/health` endpoint returns `503` during drain (good for readiness probes to remove the pod from service), while `/live` remains accessible (good for liveness probes to avoid restarts during drain).

## Troubleshooting

### Startup Probe or Readiness Probe Failure, container log contains "KeyboardInterrupt: terminated"

If the startup or readiness probe failureThreshold is too low for the time needed to start up the server, Kubernetes scheduler will kill the container. A couple of indications that this has happened:

1. container log contains "KeyboardInterrupt: terminated"
2. `kubectl get events` shows message `Container $NAME failed startup probe, will be restarted`

To mitigate, increase the failureThreshold to allow more time for the model server to start serving. You can identify an ideal failureThreshold by removing the probes from the manifest and measuring how much time it takes for the model server to show it's ready to serve.

## Conclusion

Deploying vLLM with Kubernetes allows for efficient scaling and management of ML models leveraging GPU resources. By following the steps outlined above, you should be able to set up and test a vLLM deployment within your Kubernetes cluster. If you encounter any issues or have suggestions, please feel free to contribute to the documentation.
