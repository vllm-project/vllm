.. _deploying_with_kserve:

Deploying with KServe
============================

vLLM can be deployed with `KServe <https://github.com/kserve/kserve>`_ on Kubernetes for highly scalable distributed model serving by defining the following :code:`InferenceService` YAML spec:

.. code-block:: yaml
	apiVersion: serving.kserve.io/v1beta1
	kind: InferenceService
	metadata:
	  name: llama-2-7b
	spec:
	  predictor:
	    containers:
	      - args:
	        - --port
	        - "8080"
	        - --model
	        - /mnt/models
	      command:
	        - python3
	        - -m
	        - vllm.entrypoints.api_server
	      env:
	        - name: STORAGE_URI
	          value: gcs://kfserving-examples/llm/huggingface/llama
	      image: kserve/vllmserver:latest
	      name: kserve-container
	      resources:
	        limits:
	          cpu: "4"
	          memory: 50Gi
	          nvidia.com/gpu: "1"
	        requests:
	          cpu: "1"
	          memory: 50Gi
	          nvidia.com/gpu: "1"

Please see `Deploy the LLaMA model with vLLM Runtime <https://kserve.github.io/website/latest/modelserving/v1beta1/llm/vllm/>`_ for more details.
