.. _quickstart:

Quickstart
==========

This guide shows how to use vLLM to:

* run offline batched inference on a dataset;
* build an API server for a large language model;
* start an OpenAI-compatible API server.

Be sure to complete the :ref:`installation instructions <installation>` before continuing with this guide.

Offline Batched Inference
-------------------------

We first show an example of using vLLM for offline batched inference on a dataset. In other words, we use vLLM to generate texts for a list of input prompts.

Import ``LLM`` and ``SamplingParams`` from vLLM. The ``LLM`` class is the main class for running offline inference with vLLM engine. The ``SamplingParams`` class specifies the parameters for the sampling process.

.. code-block:: python

    from vllm import LLM, SamplingParams

Define the list of input prompts and the sampling parameters for generation. The sampling temperature is set to 0.8 and the nucleus sampling probability is set to 0.95. For more information about the sampling parameters, refer to the `class definition <https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py>`_.

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

Initialize vLLM's engine for offline inference with the ``LLM`` class and the `OPT-125M model <https://arxiv.org/abs/2205.01068>`_. The list of supported models can be found at :ref:`supported models <supported_models>`.

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

Call ``llm.generate`` to generate the outputs. It adds the input prompts to vLLM engine's waiting queue and executes the vLLM engine to generate the outputs with high throughput. The outputs are returned as a list of ``RequestOutput`` objects, which include all the output tokens.

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


The code example can also be found in `examples/offline_inference.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py>`_.


API Server
----------

vLLM can be deployed as an LLM service. We provide an example `FastAPI <https://fastapi.tiangolo.com/>`_ server. Check `vllm/entrypoints/api_server.py <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py>`_ for the server implementation. The server uses ``AsyncLLMEngine`` class to support asynchronous processing of incoming requests.

Start the server:

.. code-block:: console

    $ python -m vllm.entrypoints.api_server

By default, this command starts the server at ``http://localhost:8000`` with the OPT-125M model.

Query the model in shell:

.. code-block:: console

    $ curl http://localhost:8000/generate \
    $     -d '{
    $         "prompt": "San Francisco is a",
    $         "use_beam_search": true,
    $         "n": 4,
    $         "temperature": 0
    $     }'

See `examples/api_client.py <https://github.com/vllm-project/vllm/blob/main/examples/api_client.py>`_ for a more detailed client example.

OpenAI-Compatible Server
------------------------

vLLM can be deployed as a server that mimics the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.

Start the server:

.. code-block:: console

    $ python -m vllm.entrypoints.openai.api_server \
    $     --model facebook/opt-125m

By default, it starts the server at ``http://localhost:8000``. You can specify the address with ``--host`` and ``--port`` arguments. The server currently hosts one model at a time (OPT-125M in the above command) and implements `list models <https://platform.openai.com/docs/api-reference/models/list>`_ and `create completion <https://platform.openai.com/docs/api-reference/completions/create>`_ endpoints. We are actively adding support for more endpoints.

This server can be queried in the same format as OpenAI API. For example, list the models:

.. code-block:: console

    $ curl http://localhost:8000/v1/models

Query the model with input prompts:

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "facebook/opt-125m",
    $         "prompt": "San Francisco is a",
    $         "max_tokens": 7,
    $         "temperature": 0
    $     }'

Since this server is compatible with OpenAI API, you can use it as a drop-in replacement for any applications using OpenAI API. For example, another way to query the server is via the ``openai`` python package:

.. code-block:: python

    import openai
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"
    completion = openai.Completion.create(model="facebook/opt-125m",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

For a more detailed client example, refer to `examples/openai_completion_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_completion_client.py>`_.
