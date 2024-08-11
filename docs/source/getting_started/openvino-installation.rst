.. _installation_openvino:

Installation with OpenVINO
==========================

vLLM powered by OpenVINO supports all LLM models from :doc:`vLLM supported models list <../models/supported_models>` and can perform optimal model serving on all x86-64 CPUs with, at least, AVX2 support. OpenVINO vLLM backend supports the following advanced vLLM features:

- Prefix caching (``--enable-prefix-caching``)
- Chunked prefill (``--enable-chunked-prefill``)

**Table of contents**:

- :ref:`Requirements <openvino_backend_requirements>`
- :ref:`Quick start using Dockerfile <openvino_backend_quick_start_dockerfile>`
- :ref:`Build from source <install_openvino_backend_from_source>`
- :ref:`Performance tips <openvino_backend_performance_tips>`
- :ref:`Limitations <openvino_backend_limitations>`

.. _openvino_backend_requirements:

Requirements
------------

* OS: Linux
* Instruction set architecture (ISA) requirement: at least AVX2.

.. _openvino_backend_quick_start_dockerfile:

Quick start using Dockerfile
----------------------------

.. code-block:: console

    $ docker build -f Dockerfile.openvino -t vllm-openvino-env .
    $ docker run -it --rm vllm-openvino-env

.. _install_openvino_backend_from_source:

Install from source
-------------------

- First, install Python. For example, on Ubuntu 22.04, you can run:

  .. code-block:: console

      $ sudo apt-get update  -y
      $ sudo apt-get install python3

- Second, install prerequisites vLLM OpenVINO backend installation:

  .. code-block:: console

      $ pip install --upgrade pip
      $ pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu

- Finally, install vLLM with OpenVINO backend: 

  .. code-block:: console

      $ PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" VLLM_TARGET_DEVICE=openvino python -m pip install -v .

.. _openvino_backend_performance_tips:

Performance tips
----------------

vLLM OpenVINO backend uses the following environment variables to control behavior:

- ``VLLM_OPENVINO_KVCACHE_SPACE`` to specify the KV Cache size (e.g, ``VLLM_OPENVINO_KVCACHE_SPACE=40`` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

- ``VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8`` to control KV cache precision. By default, FP16 / BF16 is used depending on platform.

- ``VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON`` to enable U8 weights compression during model loading stage. By default, compression is turned off.

To enable better TPOT / TTFT latency, you can use vLLM's chunked prefill feature (``--enable-chunked-prefill``). Based on the experiments, the recommended batch size is ``256`` (``--max-num-batched-tokens``)

OpenVINO best known configuration is:

.. code-block:: console

    $ VLLM_OPENVINO_KVCACHE_SPACE=100 VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --enable-chunked-prefill --max-num-batched-tokens 256

.. _openvino_backend_limitations:

Limitations
-----------

- LoRA serving is not supported.

- Only LLM models are currently supported. LLaVa and encoder-decoder models are not currently enabled in vLLM OpenVINO integration.

- Tensor and pipeline parallelism are not currently enabled in vLLM integration.

- Speculative sampling is not tested within vLLM integration.
