.. _supported_models:

Supported Models
================

vLLM supports a variety of generative Transformer models in `HuggingFace Transformers <https://huggingface.co/models>`_.
The following is the list of model architectures that are currently supported by vLLM.
Alongside each architecture, we include some popular models that use it.

.. list-table::
  :widths: 25 25 50
  :header-rows: 1

  * - Architecture
    - Models
    - Example HuggingFace Models
  * - :code:`AquilaForCausalLM`
    - Aquila
    - :code:`BAAI/Aquila-7B`, :code:`BAAI/AquilaChat-7B`, etc.
  * - :code:`BaiChuanForCausalLM`
    - Baichuan
    - :code:`baichuan-inc/Baichuan2-13B-Chat`, :code:`baichuan-inc/Baichuan-7B`, etc.
  * - :code:`ChatGLMModel`
    - ChatGLM
    - :code:`THUDM/chatglm2-6b`, :code:`THUDM/chatglm3-6b`, etc.
  * - :code:`DeciLMForCausalLM`
    - DeciLM
    - :code:`Deci/DeciLM-7B`, :code:`Deci/DeciLM-7B-instruct`, etc.
  * - :code:`BloomForCausalLM`
    - BLOOM, BLOOMZ, BLOOMChat
    - :code:`bigscience/bloom`, :code:`bigscience/bloomz`, etc.
  * - :code:`FalconForCausalLM`
    - Falcon
    - :code:`tiiuae/falcon-7b`, :code:`tiiuae/falcon-40b`, :code:`tiiuae/falcon-rw-7b`, etc.
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl`, etc.
  * - :code:`GPTBigCodeForCausalLM`
    - StarCoder, SantaCoder, WizardCoder
    - :code:`bigcode/starcoder`, :code:`bigcode/gpt_bigcode-santacoder`, :code:`WizardLM/WizardCoder-15B-V1.0`, etc.
  * - :code:`GPTJForCausalLM`
    - GPT-J
    - :code:`EleutherAI/gpt-j-6b`, :code:`nomic-ai/gpt4all-j`, etc.
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b`, etc.
  * - :code:`InternLMForCausalLM`
    - InternLM
    - :code:`internlm/internlm-7b`, :code:`internlm/internlm-chat-7b`, etc.
  * - :code:`LlamaForCausalLM`
    - LLaMA, LLaMA-2, Vicuna, Alpaca, Koala, Guanaco
    - :code:`meta-llama/Llama-2-13b-hf`, :code:`meta-llama/Llama-2-70b-hf`, :code:`openlm-research/open_llama_13b`, :code:`lmsys/vicuna-13b-v1.3`, :code:`young-geng/koala`, etc.
  * - :code:`MistralForCausalLM`
    - Mistral, Mistral-Instruct
    - :code:`mistralai/Mistral-7B-v0.1`, :code:`mistralai/Mistral-7B-Instruct-v0.1`, etc.
  * - :code:`MixtralForCausalLM`
    - Mixtral-8x7B, Mixtral-8x7B-Instruct
    - :code:`mistralai/Mixtral-8x7B-v0.1`, :code:`mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.
  * - :code:`MPTForCausalLM`
    - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
    - :code:`mosaicml/mpt-7b`, :code:`mosaicml/mpt-7b-storywriter`, :code:`mosaicml/mpt-30b`, etc.
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b`, etc.
  * - :code:`PhiForCausalLM`
    - Phi
    - :code:`microsoft/phi-1_5`, :code:`microsoft/phi-2`, etc.
  * - :code:`QWenLMHeadModel`
    - Qwen
    - :code:`Qwen/Qwen-7B`, :code:`Qwen/Qwen-7B-Chat`, etc.
  * - :code:`StableLMEpochForCausalLM`
  * - :code:`Qwen2ForCausalLM`
    - Qwen2
    - :code:`Qwen/Qwen2-7B-beta`, :code:`Qwen/Qwen-7B-Chat-beta`, etc.
    - StableLM
    - :code:`stabilityai/stablelm-3b-4e1t/` , :code:`stabilityai/stablelm-base-alpha-7b-v2`, etc.
  * - :code:`YiForCausalLM`
    - Yi
    - :code:`01-ai/Yi-6B`, :code:`01-ai/Yi-34B`, etc.

If your model uses one of the above model architectures, you can seamlessly run your model with vLLM.
Otherwise, please refer to :ref:`Adding a New Model <adding_a_new_model>` for instructions on how to implement support for your model.
Alternatively, you can raise an issue on our `GitHub <https://github.com/vllm-project/vllm/issues>`_ project.

.. note::
    Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.

.. tip::
    The easiest way to check if your model is supported is to run the program below:

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

    If vLLM successfully generates text, it indicates that your model is supported.

.. tip::
    To use models from `ModelScope <https://www.modelscope.cn>`_ instead of HuggingFace Hub, set an environment variable:

    .. code-block:: shell

       $ export VLLM_USE_MODELSCOPE=True

    And use with :code:`trust_remote_code=True`.

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=..., revision=..., trust_remote_code=True)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)
