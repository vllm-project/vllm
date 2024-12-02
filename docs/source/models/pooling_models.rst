.. _pooling_models:

Pooling Models
==============

vLLM also supports pooling models, including embedding, reranking and reward models.

In vLLM, pooling models implement the :class:`~vllm.model_executor.models.VllmModelForPooling` interface.
These models use a :class:`~vllm.model_executor.layers.Pooler` to aggregate the final hidden states of the input
before returning them.

.. note::

    We currently support pooling models primarily as a matter of convenience.
    As shown in the :ref:`Compatibility Matrix <compatibility_matrix>`, most vLLM features are not applicable to
    pooling models as they only work on the generation or decode stage, so performance may not improve as much.

Offline Inference
-----------------

The :class:`~vllm.LLM` class provides various methods for offline inference.
See :ref:`Engine Arguments <engine_args>` for a list of options when initializing the model.

For pooling models, we support the following :code:`task` options:

- Embedding (:code:`"embed"` / :code:`"embedding"`)
- Classification (:code:`"classify"`/ :code:`"score"`) -- reranking models fall under this category.
- Reward Modeling (:code:`"reward"`)

The selected task determines the default :class:`~vllm.model_executor.layers.Pooler` that is used:

- Embedding: Extract only the hidden states corresponding to the last token, and apply normalization.
- Classification: Extract only the hidden states corresponding to the last token, and apply softmax.
- Reward Modeling: Extract all of the hidden states and return them directly.

When loading `Sentence Transformers <https://huggingface.co/sentence-transformers>`__ models,
we attempt to override the default pooler based on its Sentence Transformers configuration file (:code:`modules.json`).

You can customize the model's pooling method via the :code:`override_pooler_config` option,
which takes priority over both the model's and Sentence Transformers's defaults.

.. tip::

    Technically, any :ref:`generative model <generative_models>` in vLLM can be converted into a pooling model
    by aggregating and returning the hidden states directly, skipping the generation step.
    Nevertheless, to get the best results, you should use pooling models that are specifically trained as such.

``LLM.encode``
^^^^^^^^^^^^^^

The :class:`~vllm.LLM.encode` method is available to all pooling models in vLLM.
It returns the aggregated hidden states directly.

.. code-block:: python

    llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")
    outputs = llm.encode("Hello, my name is")

    outputs = model.encode(prompts)
    for output in outputs:
        embeddings = output.outputs.embedding
        print(f"Prompt: {prompt!r}, Embeddings (size={len(embeddings)}: {embeddings!r}")

A code example can be found in `examples/offline_inference_embedding.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_embedding.py>`_.

``LLM.score``
^^^^^^^^^^^^^

The :class:`~vllm.LLM.score` method outputs similarity scores between sentence pairs.
It is primarily designed for `cross-encoder models <https://www.sbert.net/examples/applications/cross-encoder/README.html>`__.
These types of models serve as rerankers between candidate query-document pairs in RAG systems.

.. note::

    vLLM can only perform the model inference component (e.g. embedding, reranking) of RAG.
    To handle RAG at a higher level, you should use integration frameworks such as `LangChain <https://github.com/langchain-ai/langchain>`_.

You can use `these tests <https://github.com/vllm-project/vllm/blob/main/tests/models/embedding/language/test_scoring.py>`_ as reference.

Online Inference
----------------

Our `OpenAI Compatible Server <../serving/openai_compatible_server>`__ can be used for online inference.
Please click on the above link for more details on how to launch the server.

Embeddings API
^^^^^^^^^^^^^^

Our Embeddings API is similar to ``LLM.encode``, accepting both text and :ref:`multi-modal inputs <multimodal_inputs>`.

The text-only API is compatible with `OpenAI Embeddings API <https://platform.openai.com/docs/api-reference/embeddings>`__
so that you can use OpenAI client to interact with it.
A code example can be found in `examples/openai_embedding_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_embedding_client.py>`_.

The multi-modal API is an extension of the `OpenAI Embeddings API <https://platform.openai.com/docs/api-reference/embeddings>`__
that incorporates `OpenAI Chat Completions API <https://platform.openai.com/docs/api-reference/chat>`__,
so it is not part of the OpenAI standard. Please see :ref:`this page <multimodal_inputs>` for more details on how to use it.

Score API
^^^^^^^^^

Our Score API is similar to ``LLM.score``.
Please see `this page <../serving/openai_compatible_server.html#score-api-for-cross-encoder-models>`__ for more details on how to use it.
