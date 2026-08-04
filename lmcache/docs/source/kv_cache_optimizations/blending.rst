Blending
================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions.
For example, CacheBlend can combine multiple (pre-)computed KV caches, when their corresponding texts are concatenated in the LLM input

Configuring CacheBlend in RAG scenarios
-------------------------------------------------

Here, we will explain the code in our end-to-end `example <https://github.com/LMCache/LMCache/tree/dev/examples/blend_kv_v1/blend.py>`_>.

Below are some blending-related configurations (and explanations):

.. code-block:: python

    # Enable blending in LMCache
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"

    # Separator string between different chunks
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = " # # "

    # Layerwise must be turned on when blending is enabled
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    # Determining which tokens to recompute at layer 1
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"

    # Ratio of tokens to recompute
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    # Optionally, we can use sparse attention to improve generation quality
    # by using more accurate attention mask
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

Firstly, we preprocess texts into tokens, as tokenizing a concatenated string may produce different tokens than concatenating the results of tokenizing each string individually.
For example, assume we have a system prompt and three text chunks. We need to preprocess them into tokens before sending to the LLM:

.. code-block:: python

    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1_prompt = tokenizer.encode("Hello, how are you?" * 500)[1:]
    chunk2_prompt = tokenizer.encode("Hello, what's up?" * 500)[1:]
    chunk3_prompt = tokenizer.encode("Hi, what are you up to?" * 500)[1:]
    blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]
    first_prompt = (
        sys_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + tokenizer.encode("Hello, my name is")[1:]
    )

Then, we can send the tokenized prompt to vLLM. Meanwhile, LMCache will store the KV caches of different chunks according to the ``BLEND_SPECIAL_STR``.

.. code-block:: python

    llm.generate(prompts={"prompt_token_ids": first_prompt})

Similarly, we build another prompt using the same chunks but with different orders.

.. code-block:: python

    second_prompt = (
        sys_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + tokenizer.encode("Hello, how are you?")[1:]
    )
    llm.generate(prompts={"prompt_token_ids": second_prompt})

Even though the second prompt has a different order of chunks, LMCache can still reuse the KV caches of chunk1, chunk2, and chunk3.

