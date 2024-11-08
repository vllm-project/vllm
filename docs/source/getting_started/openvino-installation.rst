.. _installation_openvino:

Installation with OpenVINO
==========================

vLLM powered by OpenVINO supports all LLM models from :doc:`vLLM supported models list <../models/supported_models>` and can perform optimal model serving on all x86-64 CPUs with, at least, AVX2 support, as well as on both integrated and discrete Intel® GPUs (`the list of supported GPUs <https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html#gpu>`_). OpenVINO vLLM backend supports the following advanced vLLM features:

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

- [Optional] To use vLLM OpenVINO backend with a GPU device, ensure your system is properly set up. Follow the instructions provided here: `https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`_.

.. _openvino_backend_performance_tips:

Performance tips
----------------

vLLM OpenVINO backend environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``VLLM_OPENVINO_DEVICE`` to specify which device utilize for the inference. If there are multiple GPUs in the system, additional indexes can be used to choose the proper one (e.g, ``VLLM_OPENVINO_DEVICE=GPU.1``). If the value is not specified, CPU device is used by default.

- ``VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON`` to enable U8 weights compression during model loading stage. By default, compression is turned off. You can also export model with different compression techniques using `optimum-cli` and pass exported folder as `<model_id>`

CPU performance tips
~~~~~~~~~~~~~~~~~~~~

CPU uses the following environment variables to control behavior:

- ``VLLM_OPENVINO_KVCACHE_SPACE`` to specify the KV Cache size (e.g, ``VLLM_OPENVINO_KVCACHE_SPACE=40`` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

- ``VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8`` to control KV cache precision. By default, FP16 / BF16 is used depending on platform.

To enable better TPOT / TTFT latency, you can use vLLM's chunked prefill feature (``--enable-chunked-prefill``). Based on the experiments, the recommended batch size is ``256`` (``--max-num-batched-tokens``)

OpenVINO best known configuration for CPU is:

.. code-block:: console

    $ VLLM_OPENVINO_KVCACHE_SPACE=100 VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --enable-chunked-prefill --max-num-batched-tokens 256

GPU performance tips
~~~~~~~~~~~~~~~~~~~~
GPU device implements the logic for automatic detection of available GPU memory and, by default, tries to reserve as much memory as possible for the KV cache (taking into account ``gpu_memory_utilization`` option). However, this behavior can be overridden by explicitly specifying the desired amount of memory for the KV cache using ``VLLM_OPENVINO_KVCACHE_SPACE`` environment variable (e.g, ``VLLM_OPENVINO_KVCACHE_SPACE=8`` means 8 GB space for KV cache).

Currently, the best performance using GPU can be achieved with the default vLLM execution parameters for models with quantized weights (8 and 4-bit integer data types are supported) and `preemption-mode=swap`.

OpenVINO best known configuration for GPU is:

.. code-block:: console

    $ VLLM_OPENVINO_DEVICE=GPU VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json

.. _openvino_backend_limitations:

Limitations
-----------

- LoRA serving is not supported.

- Only LLM models are currently supported. LLaVa and encoder-decoder models are not currently enabled in vLLM OpenVINO integration.

- Tensor and pipeline parallelism are not currently enabled in vLLM integration.
