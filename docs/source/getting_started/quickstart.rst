Quickstart
==========

This guide shows how to use vLLM for:

* running offline batched inference on a dataset
* build an API server for a large language model
* start an OpenAI-compatible API server

Be sure to complete the `installation instructions <installation.html>`_ before continuing with this guide.

Offline Batched Inference
-------------------------

We first show an example of using vLLM for offline batched inference on a dataset. In other words, we use vLLM to generate texts for a list of input prompts.

First, import ``LLM`` and ``SamplingParams`` from vLLM. The ``LLM`` class is the main class for running offline inference with vLLM engine. The ``SamplingParams`` class specifies the parameters for the sampling process.

.. code-block:: python

    from cacheflow import LLM, SamplingParams

We perform inference on the following list of input prompts. We use a ``SamplingParams`` object to specify the sampling parameters for the generation process. In this example, we set the sampling temperature to 0.8 and nucleus sampling probability to 0.95. For more information about the sampling parameters, please refer to the `class definition <https://github.com/WoosukKwon/cacheflow/blob/main/cacheflow/sampling_params.py>`_.

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

Next, we initialize the LLM server for offline inference with the ``LLM`` class. We use the `OPT model <https://arxiv.org/abs/2205.01068>`_ with 125 million parameters as an example. The list of supported models can be found in the `supported models page </models/supported_models.html>`_.

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

We call ``llm.generate`` to generate the outputs with the given input prompts and sampling parameters. It adds the input prompts to CacheFlow server's waiting queue. Then, the CacheFlow server generates the outputs for the input prompts and returns the outputs with high throughput. The outputs are returned as a list of ``RequestOutput`` objects, which includes the output tokens and log probabilites of each output.

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


The code example can also be found in `examples/offline_inference.py <https://github.com/WoosukKwon/cacheflow/blob/main/examples/offline_inference.py>`_.


Simple FastAPI Server
---------------------

CacheFlow can also be deployed as an LLM server. We provide an example server implementation using `FastAPI <https://fastapi.tiangolo.com/>`_ as an frontend at `cacheflow/entrypoints/simple_fastapi_frontend.py <https://github.com/WoosukKwon/cacheflow/blob/main/cacheflow/entrypoints/simple_fastapi_frontend.py>`_. The server uses ``AsyncLLMServer`` class to support asynchronous processing of incoming requests. To start the server, run the following command:

.. code-block:: bash

    python -m cacheflow.entrypoints.simple_fastapi_frontend

By default, this commands start the server at ``http://localhost:8001`` with the OPT-125M model. To query the model, run the following command:

.. code-block:: bash

    curl http://localhost:8001/generate \
        -d '{
            "prompt": "San Francisco is a",
            "use_beam_search": true,
            "n": 4,
            "temperature": 0
        }'

For a more detailed client example, please refer to `examples/simple_fastapi_client.py <https://github.com/WoosukKwon/cacheflow/blob/main/examples/simple_fastapi_client.py>`_.

OpenAI-Compatible Server
------------------------

CacheFlow can be deployed as a server that mimics the OpenAI API protocol. This allows CacheFlow to be used as a drop-in replacement for applications using OpenAI API. To start an OpenAI-compatible server, run the following command:

.. code-block:: bash

    python -m cacheflow.entrypoints.openai.openai_frontend \
        --model facebook/opt-125m

By default, this commands start the server at ``http://localhost:8000``. You can specify the host and port with ``--host`` and ``--port`` arguments. The server currently hosts one model at a time (OPT-125M in the above command) and implements `list models <https://platform.openai.com/docs/api-reference/models/list>`_ and `create completion <https://platform.openai.com/docs/api-reference/completions/create>`_ endpoints. We are actively adding support for more endpoints.

This server can be queried with the same format as OpenAI API. For example, you can list the models with the following command:

.. code-block:: bash

    curl http://localhost:8000/v1/models

and query the model with the following command:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-125m",
            "prompt": "San Francisco is a",
            "max_tokens": 7,
            "temperature": 0
        }'

Since this server is fully compatible with OpenAI API, you can use it as a drop-in replacement for applications using OpenAI API. For example, you can query the server with ``openai`` python package:

.. code-block:: python

    import openai
    # Modify OpenAI's API key and API base to use CacheFlow's API server.
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"
    completion = openai.Completion.create(model="facebook/opt-125m",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

For a more detailed client example, please refer to `examples/openai_client.py <https://github.com/WoosukKwon/cacheflow/blob/main/examples/openai_client.py>`_.
