.. _recipe_llama:

Llama
=====

Validated models
----------------

- `meta-llama/Meta-Llama-3.1-8B <https://huggingface.co/meta-llama/Llama-3.1-8B>`_
- `meta-llama/Meta-Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_
- `meta-llama/Meta-Llama-3.1-70B <https://huggingface.co/meta-llama/Llama-3.1-70B>`_
- `meta-llama/Meta-Llama-3.1-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `LlamaForCausalLM in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture `LlamaForCausalLM`).

      **Status:** Validated with LMCache.

      Apply for access on the model card page and add your
      `huggingface token <https://huggingface.co/docs/hub/en/security-tokens>`_
      as an environment variable:

      .. code-block:: bash 

         export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxx
      
      |

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Get the chat templates for tool calling by following the `Llama tool calling guide <https://docs.vllm.ai/en/latest/features/tool_calling/#llama-models-llama3_json>`_ from vLLM.

      Start vLLM with the LMCache MP connector:

      **Meta-Llama-3.1-8B** (1 GPU):

      .. code-block:: bash

         vllm serve meta-llama/Meta-Llama-3.1-8B \
             --trust-remote-code \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |
      
      **Meta-Llama-3.1-8B-Instruct** (1 GPU):

      .. code-block:: bash

         vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser llama3_json \
             --chat-template <path_to_llama3.1_json_template> \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Meta-Llama-3.1-70B** (4 GPUs):

      .. code-block:: bash

         vllm serve meta-llama/Meta-Llama-3.1-70B \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Meta-Llama-3.1-70B-Instruct** (4 GPUs):

      .. code-block:: bash

         vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser llama3_json \
             --chat-template <path_to_llama3.1_json_template> \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      Adjust ``--tensor-parallel-size`` to match your hardware. For the
      generic LMCache + vLLM wiring (ports, remote hosts),
      see :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

Compression support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Status
     - Notes
   * - :doc:`CacheGen <../kv_cache_optimizations/compression/cachegen>`
     - Not validated
     -

Caveats
-------

None known.