Welcome to vLLM!
================

.. figure:: ./assets/logos/vllm-logo-text-light.png
  :width: 60%
  :align: center
  :alt: vLLM
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>Easy, fast, and cheap LLM serving for everyone
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>



vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

* State-of-the-art serving throughput
* Efficient management of attention key and value memory with **PagedAttention**
* Continuous batching of incoming requests
* Fast model execution with CUDA/HIP graph
* Quantization: `GPTQ <https://arxiv.org/abs/2210.17323>`_, `AWQ <https://arxiv.org/abs/2306.00978>`_, `SqueezeLLM <https://arxiv.org/abs/2306.07629>`_, FP8 KV Cache
* Optimized CUDA kernels

vLLM is flexible and easy to use with:

* Seamless integration with popular HuggingFace models
* High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
* Tensor parallelism and pipeline parallelism support for distributed inference
* Streaming outputs
* OpenAI-compatible API server
* Support NVIDIA GPUs and AMD GPUs
* (Experimental) Prefix caching support
* (Experimental) Multi-lora support

For more information, check out the following:

* `vLLM announcing blog post <https://vllm.ai>`_ (intro to PagedAttention)
* `vLLM paper <https://arxiv.org/abs/2309.06180>`_ (SOSP 2023)
* `How continuous batching enables 23x throughput in LLM inference while reducing p50 latency <https://www.anyscale.com/blog/continuous-batching-llm-inference>`_ by Cade Daniel et al.
* :ref:`vLLM Meetups <meetups>`.



Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/amd-installation
   getting_started/openvino-installation
   getting_started/cpu-installation
   getting_started/neuron-installation
   getting_started/tpu-installation
   getting_started/xpu-installation
   getting_started/quickstart
   getting_started/debugging
   getting_started/examples/examples_index

.. toctree::
   :maxdepth: 1
   :caption: Serving

   serving/openai_compatible_server
   serving/deploying_with_docker
   serving/distributed_serving
   serving/metrics
   serving/env_vars
   serving/usage_stats
   serving/integrations
   serving/tensorizer
   serving/faq

.. toctree::
   :maxdepth: 1
   :caption: Models

   models/supported_models
   models/adding_model
   models/enabling_multimodal_inputs
   models/engine_args
   models/lora
   models/vlm
   models/spec_decode
   models/performance

.. toctree::
   :maxdepth: 1
   :caption: Quantization

   quantization/supported_hardware
   quantization/auto_awq
   quantization/bnb
   quantization/fp8
   quantization/fp8_e5m2_kvcache
   quantization/fp8_e4m3_kvcache

.. toctree::
   :maxdepth: 1
   :caption: Automatic Prefix Caching

   automatic_prefix_caching/apc
   automatic_prefix_caching/details

.. toctree::
   :maxdepth: 1
   :caption: Performance benchmarks

   performance_benchmark/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   dev/sampling_params
   dev/offline_inference/offline_index
   dev/engine/engine_index
   dev/kernel/paged_attention
   dev/input_processing/model_inputs_index
   dev/multimodal/multimodal_index
   dev/dockerfile/dockerfile

.. toctree::
   :maxdepth: 1
   :caption: Community

   community/meetups
   community/sponsors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
