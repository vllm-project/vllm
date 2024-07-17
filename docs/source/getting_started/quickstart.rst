.. _quickstart:

Quickstart
==========

This guide shows how to use vLLM to:

* run offline batched inference on a dataset;
* build an API server for a large language model;
* start an OpenAI-compatible API server.

Be sure to complete the :ref:`installation instructions <installation>` before continuing with this guide.

.. note::

    By default, vLLM downloads model from `HuggingFace <https://huggingface.co/>`_. If you would like to use models from `ModelScope <https://www.modelscope.cn>`_ in the following examples, please set the environment variable:

    .. code-block:: shell

        export VLLM_USE_MODELSCOPE=True

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

OpenAI-Compatible Server
------------------------

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.
By default, it starts the server at ``http://localhost:8000``. You can specify the address with ``--host`` and ``--port`` arguments. The server currently hosts one model at a time (OPT-125M in the command below) and implements `list models <https://platform.openai.com/docs/api-reference/models/list>`_, `create chat completion <https://platform.openai.com/docs/api-reference/chat/completions/create>`_, and `create completion <https://platform.openai.com/docs/api-reference/completions/create>`_ endpoints. We are actively adding support for more endpoints.

Start the server:

.. code-block:: console

    $ vllm serve facebook/opt-125m

By default, the server uses a predefined chat template stored in the tokenizer. You can override this template by using the ``--chat-template`` argument:

.. code-block:: console

    $ vllm serve facebook/opt-125m --chat-template ./examples/template_chatml.jinja

This server can be queried in the same format as OpenAI API. For example, list the models:

.. code-block:: console

    $ curl http://localhost:8000/v1/models

You can pass in the argument ``--api-key`` or environment variable ``VLLM_API_KEY`` to enable the server to check for API key in the header.

Using OpenAI Completions API with vLLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(model="facebook/opt-125m",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

For a more detailed client example, refer to `examples/openai_completion_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_completion_client.py>`_.

Using OpenAI Chat API with vLLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The vLLM server is designed to support the OpenAI Chat API, allowing you to engage in dynamic conversations with the model. The chat interface is a more interactive way to communicate with the model, allowing back-and-forth exchanges that can be stored in the chat history. This is useful for tasks that require context or more detailed explanations.

Querying the model using OpenAI Chat API:

You can use the `create chat completion <https://platform.openai.com/docs/api-reference/chat/completions/create>`_ endpoint to communicate with the model in a chat-like interface:

.. code-block:: console

    $ curl http://localhost:8000/v1/chat/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "facebook/opt-125m",
    $         "messages": [
    $             {"role": "system", "content": "You are a helpful assistant."},
    $             {"role": "user", "content": "Who won the world series in 2020?"}
    $         ]
    $     }'

Python Client Example:

Using the `openai` python package, you can also communicate with the model in a chat-like manner:

.. code-block:: python

    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    )
    print("Chat response:", chat_response)

For more in-depth examples and advanced features of the chat API, you can refer to the official OpenAI documentation.
