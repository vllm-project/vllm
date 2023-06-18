Welcome to vLLM!
================

**vLLM** is a fast and easy-to-use library for LLM inference and serving.
Its core features include:

- State-of-the-art performance in serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Seamless integration with popular HuggingFace models
- Dynamic batching of incoming requests
- Optimized CUDA kernels
- High-throughput serving with various decoding algorithms, including *parallel sampling* and *beam search*
- Tensor parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server

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
