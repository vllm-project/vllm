.. _bitblas:

BitBLAS
==================

vLLM now supports `BitBLAS <https://github.com/microsoft/BitBLAS>`_ for more efficient and flexible model inference.
Compared to other quantization frameworks, BitBLAS provides more precision combinations.

Below are the steps to utilize BitBLAS with vLLM.

.. code-block:: console

    $ pip install bitblas>=0.0.1.dev15

vLLM reads the model's config file and supports pre-quantized checkpoint.

You can find pre-quantized models on https://huggingface.co/models?other=bitblas or https://huggingface.co/models?other=bitnet or https://huggingface.co/models?other=gptq.

And usually, these repositories have a quantize_config.json file that includes a quantization_config section.

Read bitblas format checkpoint.
--------------------------

.. code-block:: python

    from vllm import LLM
    import torch
    # unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
    model_id = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
    quantization="bitblas")

Read gptq format checkpoint.
--------------------------
.. code-block:: python

    from vllm import LLM
    import torch
    # unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
    model_id = "hxbgsyxh/llama-13b-4bit-g-1"
    llm = LLM(model=model_id, dtype=torch.float16, trust_remote_code=True, \
    quantization="bitblas", max_model_len=1024)

.. From bitnet format
