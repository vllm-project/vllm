.. _run_on_langchain:

Serving with Langchain
============================

vLLM is also available via `Langchain <https://github.com/langchain-ai/langchain>`_ .

To install langchain, run

.. code-block:: console

    $ pip install langchain langchain_community -q

To run inference on a single or multiple GPUs, use ``VLLM`` class from ``langchain``.

.. code-block:: python

    from langchain_community.llms import VLLM

    llm = VLLM(model="mosaicml/mpt-7b",
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=128,
               top_k=10,
               top_p=0.95,
               temperature=0.8,
               # tensor_parallel_size=... # for distributed inference
    )

    print(llm("What is the capital of France ?"))

Please refer to this `Tutorial <https://python.langchain.com/docs/integrations/llms/vllm>`_ for more details.
