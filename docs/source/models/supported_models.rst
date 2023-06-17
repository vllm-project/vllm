.. _supported_models:

Supported Models
================

vLLM supports a variety of generative Transformer models in `HuggingFace Transformers <https://github.com/huggingface/transformers>`_.
The following is the list of model architectures that are currently supported by vLLM.
Alongside each architecture, we include some popular models that use it.

.. list-table::
  :widths: 25 75
  :header-rows: 1

  * - Architecture
    - Models
  * - :code:`GPT2LMHeadModel`
    - GPT-2
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
  * - :code:`LlamaForCausalLM`
    - LLaMA, Vicuna, Alpaca, Koala
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML

If your model uses one of the above model architectures, you can seamlessly run your model with vLLM.
Otherwise, please refer to :ref:`Adding a New Model <adding_a_new_model>` for instructions on how to implement support for your model.
Alternatively, you can raise an issue on our `GitHub <https://github.com/WoosukKwon/vllm/issues>`_ project.

.. tip::
    The easiest way to check if your model is supported is to run the program below:

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

    If vLLM successfully generates text, it indicates that your model is supported.
