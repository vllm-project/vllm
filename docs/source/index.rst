Welcome to vLLM!
================

**vLLM** is a high-throughput and memory-efficient inference and serving engine for large language models (LLM).
Its core features include:

- **PagedAttention**, efficient management of cached attention keys and values
- Seamless integration with popular HuggingFace models
- Advanced batching mechanism
- Optimized CUDA kernels
- Efficient support for various decoding algorithms such as parallel sampling and beam search
- Tensor parallelism support for multi-GPU inference
- OpenAI-compatible API

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
