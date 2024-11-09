Integration with 🤗 Transformers
===================================

This document describes how vLLM integrates with 🤗 Transformers. We will explain step by step what is happening under the hood when we run `vllm serve`.

Let's say we want to serve the popular llama model from 🤗 Transformers by ``vllm serve meta-llama/Llama-3.1-8B```:

- The ``model`` argument is ``meta-llama/Llama-3.1-8B``.