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
   <a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star skypilot-org/skypilot on GitHub">Star</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch skypilot-org/skypilot on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork skypilot-org/skypilot on GitHub">Fork</a>
   </p>



vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

* State-of-the-art serving throughput
* Efficient management of attention key and value memory with **PagedAttention**
* Dynamic batching of incoming requests
* Optimized CUDA kernels

vLLM is flexible and easy to use with:

* Seamless integration with popular HuggingFace models
* High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
* Tensor parallelism support for distributed inference
* Streaming outputs
* OpenAI-compatible API server

For more information, please refer to our `blog post <>`_.


Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Models

   models/supported_models
   models/adding_model
