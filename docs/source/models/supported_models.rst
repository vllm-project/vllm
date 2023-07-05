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
  * - :code:`BloomForCausalLM`
    - BLOOM, BLOOMZ, BLOOMChat
    - :code:`bigscience/bloom`, :code:`bigscience/bloomz`, etc.
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl`, etc.
  * - :code:`GPTBigCodeForCausalLM`
    - StarCoder, SantaCoder, WizardCoder
    - :code:`bigcode/starcoder`, :code:`bigcode/gpt_bigcode-santacoder`, :code:`WizardLM/WizardCoder-15B-V1.0`, etc.
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b`, etc.
  * - :code:`LlamaForCausalLM`
    - LLaMA, Vicuna, Alpaca, Koala, Guanaco
    - :code:`openlm-research/open_llama_13b`, :code:`lmsys/vicuna-13b-v1.3`, :code:`young-geng/koala`, :code:`JosephusCheung/Guanaco`, etc.
  * - :code:`MPTForCausalLM`
    - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
    - :code:`mosaicml/mpt-7b`, :code:`mosaicml/mpt-7b-storywriter`, :code:`mosaicml/mpt-30b`, etc.
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b`, etc.

If your model uses one of the above model architectures, you can seamlessly run your model with vLLM.
Otherwise, please refer to :ref:`Adding a New Model <adding_a_new_model>` for instructions on how to implement support for your model.
Alternatively, you can raise an issue on our `GitHub <https://github.com/vllm-project/vllm/issues>`_ project.

.. tip::
    The easiest way to check if your model is supported is to run the program below:

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

    If vLLM successfully generates text, it indicates that your model is supported.
