.. _run_on_haystack

Serving with Haystack
============================

`Haystack <https://github.com/deepset-ai/haystack>`_ is an open source LLM framework in Python by `deepset <https://www.deepset.ai/>`_ for building customizable, production-ready LLM applications. It is an end-to-end framework that assists the orchestration of complete NLP applications by providing tooling for each step of the application-building life cycle.

To start using Haystack with vLLM, install:

.. code-block:: console

    $ pip install vllm-haystack

You can use models hosted on a ``vLLM`` server or use locally hosted ``vLLM`` models

Use a Model Hosted on a vLLM Server
-----------------------------------

For models hosted on vLLM server, you need to use ``vLLMInvocationLayer``.

Here is a simple example of how a ``PromptNode`` can be created with ``vLLMInvocationLayer``.

.. code-block:: python

		from haystack.nodes import PromptNode, PromptModel
		from vllm_haystack import vLLMInvocationLayer
		
		
		model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=256, api_key="EMPTY", model_kwargs={
		        "api_base" : API, # Replace this with your API-URL
		        "maximum_context_length": 2048,
		    })
		
		prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=256)

The model will be inferred based on the model served on the vLLM server.

Use a vLLM Model Hosted Locally
-------------------------------
.. note::

    To run vLLM locally, you need to have ``vllm`` installed and a supported GPU.

If you don’t want to use an API-Server, this integration also provides ``vLLMLocalInvocationLayer`` which executes the vLLM on the same node Haystack is running on.

Here is a simple example of how a ``PromptNode`` can be created with the ``vLLMLocalInvocationLayer``.

.. code-block:: python

		from haystack.nodes import PromptNode, PromptModel
		from vllm_haystack import vLLMInvocationLayer
		
		
		model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=256, api_key="EMPTY", model_kwargs={
		        "api_base" : API, # Replace this with your API-URL
		        "maximum_context_length": 2048,
		    })
		
		prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=256)

Refer to `Haystack Integration Page <https://haystack.deepset.ai/integrations/vllm>`_ for more details.
