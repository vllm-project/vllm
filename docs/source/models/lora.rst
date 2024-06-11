.. _lora:

Using LoRA adapters
===================

This document shows you how to use `LoRA adapters <https://arxiv.org/abs/2106.09685>`_ with vLLM on top of a base model.

LoRA adapters can be used with any vLLM model that implements :class:`~vllm.model_executor.models.interfaces.SupportsLoRA`.

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

Serving LoRA Adapters
---------------------
LoRA adapted models can also be served with the Open-AI compatible vLLM server. To do so, we use
``--lora-modules {name}={path} {name}={path}`` to specify each LoRA module when we kickoff the server:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --enable-lora \
        --lora-modules sql-lora=~/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/

The server entrypoint accepts all other LoRA configuration parameters (``max_loras``, ``max_lora_rank``, ``max_cpu_loras``,
etc.), which will apply to all forthcoming requests. Upon querying the ``/models`` endpoint, we should see our LoRA along
with its base model:

.. code-block:: bash

    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-2-7b-hf",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }

Requests can specify the LoRA adapter as if it were any other model via the ``model`` request parameter. The requests will be
processed according to the server-wide LoRA configuration (i.e. in parallel with base model requests, and potentially other
LoRA adapter requests if they were provided and ``max_loras`` is set high enough).

The following is an example request

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "sql-lora",
            "prompt": "San Francisco is a",
            "max_tokens": 7,
            "temperature": 0
        }' | jq
