.. _lora:

Using LoRA adapters
===================

This document shows you how to use `LoRA adapters <https://arxiv.org/abs/2106.09685>`_ with vLLM on top of a base model.
Adapters can be efficiently served on a per request basis with minimal overhead. First we download the adapter(s) and save
them locally with

.. code-block:: python

    from huggingface_hub import snapshot_download

    sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")


Then we instantiate the base model and pass in the ``enable_lora=True`` flag:

.. code-block:: python

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)


We can now submit the prompts and call ``llm.generate`` with the ``lora_request`` parameter. The first parameter
of ``LoRARequest`` is a human identifiable name, the second parameter is a globally unique ID for the adapter and
the third parameter is the path to the LoRA adapter.

.. code-block:: python

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"]
    )

    prompts = [
         "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
         "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
    )


Check out `examples/multilora_inference.py <https://github.com/vllm-project/vllm/blob/main/examples/multilora_inference.py>`_
for an example of how to use LoRA adapters with the async engine and how to use more advanced configuration options.