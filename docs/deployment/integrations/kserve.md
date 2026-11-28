# KServe

vLLM can be deployed with [KServe](https://github.com/kserve/kserve) on Kubernetes for scalable, production-grade model serving with built-in autoscaling, canary rollouts, and multi-model management.

You can use vLLM with KServe's [Hugging Face serving runtime](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) or via [`LLMInferenceService` that uses llm-d](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview). This guide covers direct deployment using a custom `ServingRuntime`.

For a general introduction to KServe concepts, see the [KServe documentation](https://kserve.github.io/website/latest/).

## Prerequisites

- Kubernetes 1.27+ or OpenShift 4.14+
- KServe 0.12+ installed (Serverless or RawDeployment mode)
- GPU nodes with NVIDIA device plugin configured

## ServingRuntime configuration

KServe uses a `ServingRuntime` custom resource to define how a model server is launched. The following example deploys vLLM using the `vllm/vllm-openai` image.

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-runtime
spec:
  supportedModelFormats:
    - name: huggingface
      version: "1"
      autoSelect: true
  containers:
    - name: kserve-container
      image: vllm/vllm-openai:latest
      command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
      args:
        - "--model=/mnt/models"
        - "--served-model-name={{.Name}}"
        - "--port=8080"
      ports:
        - containerPort: 8080
          protocol: TCP
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
  protocolVersions:
    - v2
    - grpc-v2
```

!!! warning "Use `python3`, not `python`, in the command"
    The `vllm/vllm-openai` image installs vLLM into a virtualenv whose `python` binary is not on the default `PATH` when KServe launches the container. Using `command: ["python"]` results in `executable not found`. Always use `command: ["python3", "-m", ...]` or the full path `/opt/vllm/bin/python3`.

## Deploy an InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-3-8b
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "pvc://model-pvc/llama-3-8b"
      runtime: vllm-runtime
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

Check the rollout status:

```console
kubectl get inferenceservice llama-3-8b
```

## OpenShift AI (RHOAI) specific notes

### GPU architecture support

OpenShift AI ships its own vLLM image (`rhaiis/vllm-cuda-rhel9`). This image may be compiled for a different set of CUDA compute capabilities than the upstream `vllm/vllm-openai` image. Notably, **Volta (sm_70, V100)** may not be included in vendor builds.

If you see `cudaErrorNoKernelImageForDevice` at runtime, your GPU's compute capability is not compiled into the vendor image. Options:

- Use the upstream `vllm/vllm-openai` image directly in your `ServingRuntime` (requires pulling from docker.io).
- Request a custom build targeting your GPU's sm level from your platform team.
- See the [GPU architecture support](../../getting_started/installation/gpu.md#nvidia-gpu-architecture-support) section for the compiled-architecture matrix.

### Flash Attention on Volta GPUs

Flash Attention 2 requires compute capability 8.0+ (Ampere or newer). On Volta (V100, sm_70), vLLM **silently disables Flash Attention** and falls back to PyTorch's Scaled Dot-Product Attention (SDPA). The log will contain:

```text
WARNING  vllm.attention.backends.flash_attn: Flash attention is not supported on Volta. Using PyTorch SDPA instead.
```

The model will run correctly, but throughput and memory efficiency will be lower than on Ampere+ GPUs. Plan GPU capacity accordingly if your deployment targets Volta hardware.

### SecurityContext and fsGroup

On OpenShift, pods run with a namespace-derived UID and fsGroup (e.g., `1000840000`). If you mount a PVC that was previously written by a pod with a different fsGroup, you may see `Permission denied` on the model files. Ensure any model-loading Job that pre-populates the PVC uses the same `securityContext.fsGroup` as the namespace.

See the [KServe PVC documentation](https://kserve.github.io/website/latest/modelserving/storage/pvc/pvc/) for details on loader job patterns.

## Sending inference requests

Once the `InferenceService` is `Ready`, query the OpenAI-compatible endpoint:

```console
ISVC_URL=$(kubectl get inferenceservice llama-3-8b -o jsonpath='{.status.url}')

curl ${ISVC_URL}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "prompt": "San Francisco is a",
    "max_tokens": 64
  }'
```
