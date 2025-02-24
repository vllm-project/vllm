(deployment-k8s)=

# Using Kubernetes

Using Kubernetes to deploy vLLM is a scalable and efficient way to serve machine learning models. 
This guide will walk you through the process of deploying vLLM with [production-stack](https://github.com/vllm-project/production-stack) and native Kubernetes.

Note that though it is doable to deploy vLLM using native Kubernetes, [production-stack](https://github.com/vllm-project/production-stack) allows you to serve LLM models on multi-node Kubernetes cluster, while including a wide range of usability, observability and performance-related features.


## Prerequisites

Before you begin, ensure that you have the following:

- A running Kubernetes cluster
- NVIDIA Kubernetes Device Plugin (`k8s-device-plugin`): This can be found at `https://github.com/NVIDIA/k8s-device-plugin/`
- Available GPU resources in your cluster

If you just start using Kubernetes, don't worry: we provide step-by-step [guide](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md) and [video](https://www.youtube.com/watch?v=EsTJbQtzj0g) in vLLM production stack [repo](https://github.com/vllm-project/production-stack) to help you get started!


## Deployment using helm chart

Helm chart hides Kubernetes complications from you so that yoiu can focus on configurations that really matters for your vLLM cluster. This [bash script](https://github.com/vllm-project/production-stack/blob/main/tutorials/install-helm.sh) helps you install helm.

Here is an minimal helm chart example:
```
servingEngineSpec:
  runtimeClassName: ""
  modelSpec:
  - name: "opt125m"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "facebook/opt-125m"

    replicaCount: 1

    requestCPU: 6
    requestMemory: "16Gi"
    requestGPU: 1

    pvcStorage: "10Gi"
```
In this helm chart:
- **`modelSpec`** includes:
  - `name`: A nickname that you prefer to call the model.
  - `repository`: Docker repository of vLLM.
  - `tag`: Docker image tag.
  - `modelURL`: the LLM model that you want to use.
- **`replicaCount`**: Number of replicas.
- **`requestCPU` and `requestMemory`**: Specifies the CPU and memory resource requests for the pod.
- **`requestGPU`**: Specifies the number of GPUs required.
- **`pvcStorage`**: Allocates persistent storage for the model.

**NOTE:** If you intend to set up two or multiple vllm pods, please refer to this [yaml file](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml).

**NOTE:** Feel free to check more [examples and tutorials](https://github.com/vllm-project/production-stack/tree/main/tutorials) and vLLM production stack [repo](https://github.com/vllm-project/production-stack)!

To run this helm chart, execute the following command:
```bash
sudo helm repo add vllm https://vllm-project.github.io/production-stack
sudo helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```
where `tutorials/assets/values-01-minimal-example.yaml` is the helm chart shown in previous example.


### Validate Installation

Monitor the deployment status using:

```bash
sudo kubectl get pods
```

And you will see that pods for the `vllm` deployment should transition to `Ready` and the `Running` state.

```
NAME                                               READY   STATUS    RESTARTS   AGE
vllm-deployment-router-859d8fb668-2x2b7        1/1     Running   0          2m38s
vllm-opt125m-deployment-vllm-84dfc9bd7-vb9bs   1/1     Running   0          2m38s
```

**NOTE:**: It may take some time for the containers to download the Docker images and LLM weights.

### Send a Query to the Stack

Forward the `vllm-router-service` port to the host machine:

```bash
sudo kubectl port-forward svc/vllm-router-service 30080:80
```

And then you can send out a query to the OpenAI-compatible API to check the available models:

```bash
curl -o- http://localhost:30080/models
```

Expected output:

```json
{
  "object": "list",
  "data": [
    {
      "id": "facebook/opt-125m",
      "object": "model",
      "created": 1737428424,
      "owned_by": "vllm",
      "root": null
    }
  ]
}
```


To send an actual chatting request, you can issue a curl request to the OpenAI `/completion` endpoint:

```bash
curl -X POST http://localhost:30080/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```

Expected output:

```json
{
  "id": "completion-id",
  "object": "text_completion",
  "created": 1737428424,
  "model": "facebook/opt-125m",
  "choices": [
    {
      "text": " there was a brave knight who...",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

This demonstrates the model generating a continuation for the provided prompt.

### Uninstall

To remove the deployment, run:

```bash
sudo helm uninstall vllm
```

## Deployment using native k8s 

1. Create a PVC, Secret and Deployment for vLLM

      PVC is used to store the model cache and it is optional, you can use hostPath or other storage options

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
              - name: HUGGING_FACE_HUB_TOKEN
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

      AMD GPU:

      You can refer to the `deployment.yaml` below if using AMD ROCm GPU like MI300X.

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
              - name: HUGGING_FACE_HUB_TOKEN
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

      You can get the full example with steps and sample yaml files from <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve>.

2. Create a Kubernetes Service for vLLM

      Next, create a Kubernetes Service file to expose the `mistral-7b` deployment:

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

3. Deploy and Test

      Apply the deployment and service configurations using `kubectl apply -f <filename>`:

      ```console
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      To test the deployment, run the following `curl` command:

      ```console
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

## Conclusion

Deploying vLLM with Kubernetes allows for efficient scaling and management of ML models leveraging GPU resources. By following the steps outlined above, you should be able to set up and test a vLLM deployment within your Kubernetes cluster. If you encounter any issues or have suggestions, please feel free to contribute to the documentation.
