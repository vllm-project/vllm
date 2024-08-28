.. _bits_and_bytes:

BitsAndBytes
==================

vLLM now supports `BitsAndBytes <https://github.com/TimDettmers/bitsandbytes>`_ for more efficient model inference.
BitsAndBytes quantizes models to reduce memory usage and enhance performance without significantly sacrificing accuracy.
Compared to other quantization methods,  BitsAndBytes eliminates the need for calibrating the quantized model with input data.

Below are the steps to utilize BitsAndBytes with vLLM.

.. code-block:: console

    $ pip install bitsandbytes>=0.42.0

vLLM reads the model's config file and supports both in-flight quantization and pre-quantized checkpoint.

You can find bitsandbytes quantized models on https://huggingface.co/models?other=bitsandbytes.
And usually, these repositories have a config.json file that includes a quantization_config section.

Read quantized checkpoint.
--------------------------

.. code-block:: python

    from vllm import LLM
    import torch
    # unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
    model_id = "unsloth/tinyllama-bnb-4bit"
    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
    quantization="bitsandbytes", load_format="bitsandbytes")

Inflight quantization: load as 4bit quantization
------------------------------------------------

.. code-block:: python

    from vllm import LLM
    import torch
    model_id = "huggyllama/llama-7b"
    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
    quantization="bitsandbytes", load_format="bitsandbytes")

