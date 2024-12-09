.. _quickstart:

==========
Quickstart
==========

This guide will help you quickly get started with vLLM to:

* :ref:`Run offline batched inference <offline_batched_inference>` 
* :ref:`Run OpenAI-compatible inference <openai_compatible_server>`

Prerequisites
--------------
- OS: Linux
- Python: 3.9 -- 3.12
- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

Installation
--------------

You can install vLLM using pip. It's recommended to use `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ to create and manage Python environments.

.. code-block:: console

    $ conda create -n myenv python=3.10 -y
    $ conda activate myenv
    $ pip install vllm

Please refer to the :ref:`installation documentation <installation>` for more details on installing vLLM.

.. _offline_batched_inference:

Offline Batched Inference
-------------------------

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). The example script for this section can be found `here <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py>`__.

The first line of this example imports the classes :class:`~vllm.LLM` and :class:`~vllm.SamplingParams`:

- :class:`~vllm.LLM` is the main class for running offline inference with vLLM engine.
- :class:`~vllm.SamplingParams` specifies the parameters for the sampling process.

.. code-block:: python

    from vllm import LLM, SamplingParams

The next section defines a list of input prompts and sampling parameters for text generation. The `sampling temperature <https://arxiv.org/html/2402.05201v1>`_ is set to ``0.8`` and the `nucleus sampling probability <https://en.wikipedia.org/wiki/Top-p_sampling>`_ is set to ``0.95``. You can find more information about the sampling parameters `here <https://docs.vllm.ai/en/stable/dev/sampling_params.html>`__.

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

The :class:`~vllm.LLM` class initializes vLLM's engine and the `OPT-125M model <https://arxiv.org/abs/2205.01068>`_ for offline inference. The list of supported models can be found :ref:`here <supported_models>`.

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

.. note::

    By default, vLLM downloads models from `HuggingFace <https://huggingface.co/>`_. If you would like to use models from `ModelScope <https://www.modelscope.cn>`_, set the environment variable ``VLLM_USE_MODELSCOPE`` before initializing the engine.

Now, the fun part! The outputs are generated using ``llm.generate``. It adds the input prompts to the vLLM engine's waiting queue and executes the vLLM engine to generate the outputs with high throughput. The outputs are returned as a list of ``RequestOutput`` objects, which include all of the output tokens.

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

.. _openai_compatible_server:

OpenAI-Compatible Server
------------------------

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.
By default, it starts the server at ``http://localhost:8000``. You can specify the address with ``--host`` and ``--port`` arguments. The server currently hosts one model at a time and implements endpoints such as `list models <https://platform.openai.com/docs/api-reference/models/list>`_, `create chat completion <https://platform.openai.com/docs/api-reference/chat/completions/create>`_, and `create completion <https://platform.openai.com/docs/api-reference/completions/create>`_ endpoints. 

Run the following command to start the vLLM server with the `Qwen2.5-1.5B-Instruct <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>`_ model:

.. code-block:: console

    $ vllm serve Qwen/Qwen2.5-1.5B-Instruct

.. note::

    By default, the server uses a predefined chat template stored in the tokenizer. You can learn about overriding it `here <https://github.com/vllm-project/vllm/blob/main/docs/source/serving/openai_compatible_server.md#chat-template>`__.

This server can be queried in the same format as OpenAI API. For example, to list the models:

.. code-block:: console

    $ curl http://localhost:8000/v1/models

You can pass in the argument ``--api-key`` or environment variable ``VLLM_API_KEY`` to enable the server to check for API key in the header.

OpenAI Completions API with vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your server is started, you can query the model with input prompts:

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "Qwen/Qwen2.5-1.5B-Instruct",
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
    completion = client.completions.create(model="Qwen/Qwen2.5-1.5B-Instruct",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

A more detailed client example can be found `here <https://github.com/vllm-project/vllm/blob/main/examples/openai_completion_client.py>`__.

OpenAI Chat Completions API with vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vLLM is designed to also support the OpenAI Chat Completions API. The chat interface is a more dynamic, interactive way to communicate with the model, allowing back-and-forth exchanges that can be stored in the chat history. This is useful for tasks that require context or more detailed explanations.

You can use the `create chat completion <https://platform.openai.com/docs/api-reference/chat/completions/create>`_ endpoint to interact with the model:

.. code-block:: console

    $ curl http://localhost:8000/v1/chat/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "Qwen/Qwen2.5-1.5B-Instruct",
    $         "messages": [
    $             {"role": "system", "content": "You are a helpful assistant."},
    $             {"role": "user", "content": "Who won the world series in 2020?"}
    $         ]
    $     }'

Alternatively, you can use the ``openai`` python package:

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
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    )
    print("Chat response:", chat_response)
