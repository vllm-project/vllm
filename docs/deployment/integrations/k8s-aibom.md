# k8s-aibom

[k8s-aibom](https://github.com/GoogleCloudPlatform/k8s-aibom) is an open-source Kubernetes controller that keeps an inventory of the AI workloads actually serving in a cluster. For every vLLM deployment it produces a [CycloneDX ML-BOM](https://cyclonedx.org/capabilities/mlbom/): the vLLM image and resolved digest, the model the server was started with, and, when the model is signed, the verified signer identity. Each attribute carries the evidence it was read from and a confidence tier (declared, inferred, or unresolved).

It is read-only toward your cluster: it watches workloads through the Kubernetes API, adds no sidecars or init containers, runs unprivileged, and never changes scheduling or serving behavior. vLLM is detected whether you run it as a plain Deployment or StatefulSet, under [KubeAI](kubeai.md), [KServe](kserve.md), [KubeRay](kuberay.md), [llm-d](llm-d.md), or the [production stack](production-stack.md), which ships it as an optional chart dependency.

## Install

```bash
helm install k8s-aibom oci://ghcr.io/googlecloudplatform/charts/k8s-aibom \
  --version 1.5.1 \
  --namespace k8s-aibom-system --create-namespace
```

Then opt in each namespace that serves models:

```bash
kubectl label namespace <your-serving-namespace> aibom.k8saibom.dev/enabled=true
```

Documents appear as `AIBOM` resources next to the workloads they describe. The `kubectl aibom` plugin (`kubectl krew install aibom`) summarizes and searches them:

```bash
kubectl aibom summary -A
kubectl aibom find --runtime vllm -A
kubectl aibom find --image vllm/vllm-openai -A
```

See the [k8s-aibom documentation](https://github.com/GoogleCloudPlatform/k8s-aibom#readme) for the document format, signature verification, external sinks, and the full list of detected runtimes.
