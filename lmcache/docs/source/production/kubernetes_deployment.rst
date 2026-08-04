Kubernetes Deployment
=====================

For Kubernetes deployment of vLLM with LMCache integration, we recommend using the `vLLM Production Stack <https://github.com/vllm-project/production-stack>`_ project. This is a specialized production-ready implementation for K8S-native cluster-wide deployment for vllm & lmcache.

For a quick start guide, please refer to the official `documentation <https://docs.vllm.ai/projects/production-stack/en/latest/getting_started/quickstart.html>`_

and replace the Helm values file with (`values-05-cpu-offloading.yaml <https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-05-cpu-offloading.yaml>`_):

.. code-block:: yaml

    servingEngineSpec:
      runtimeClassName: ""
      modelSpec:
      - name: "mistral"
        repository: "lmcache/vllm-openai"
        tag: "latest"
        modelURL: "mistralai/Mistral-7B-Instruct-v0.2"
        replicaCount: 1
        requestCPU: 10
        requestMemory: "40Gi"
        requestGPU: 1
        pvcStorage: "50Gi"
        pvcAccessMode:
          - ReadWriteOnce
        vllmConfig:
          maxModelLen: 32000

        lmcacheConfig:
          enabled: true
          cpuOffloadingBufferSize: "20"

        hf_token: <hf-token>

OR 

refer to a detailed `step-by-step tutorial <https://github.com/vllm-project/production-stack/blob/main/tutorials/05-offload-kv-cache.md>`_ on how to offload KV cache with LMCache in the production stack.
