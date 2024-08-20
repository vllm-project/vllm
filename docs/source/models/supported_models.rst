.. _supported_models:

Supported Models
================

vLLM supports a variety of generative Transformer models in `HuggingFace Transformers <https://huggingface.co/models>`_.
The following is the list of model architectures that are currently supported by vLLM.
Alongside each architecture, we include some popular models that use it.

----

Decoder-only Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
  :widths: 25 25 50 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HuggingFace Models
    - :ref:`LoRA <lora>`
  * - :code:`AquilaForCausalLM`
    - Aquila & Aquila2
    - :code:`BAAI/Aquila-7B`, :code:`BAAI/AquilaChat-7B`, etc.
    - ✅︎
  * - :code:`ArcticForCausalLM`
    - Arctic
    - :code:`Snowflake/snowflake-arctic-base`, :code:`Snowflake/snowflake-arctic-instruct`, etc.
    -
  * - :code:`BaiChuanForCausalLM`
    - Baichuan & Baichuan2
    - :code:`baichuan-inc/Baichuan2-13B-Chat`, :code:`baichuan-inc/Baichuan-7B`, etc.
    - ✅︎
  * - :code:`BloomForCausalLM`
    - BLOOM, BLOOMZ, BLOOMChat
    - :code:`bigscience/bloom`, :code:`bigscience/bloomz`, etc.
    -
  * - :code:`ChatGLMModel`
    - ChatGLM
    - :code:`THUDM/chatglm2-6b`, :code:`THUDM/chatglm3-6b`, etc.
    - ✅︎
  * - :code:`CohereForCausalLM`
    - Command-R
    - :code:`CohereForAI/c4ai-command-r-v01`, etc.
    -
  * - :code:`DbrxForCausalLM`
    - DBRX
    - :code:`databricks/dbrx-base`, :code:`databricks/dbrx-instruct`, etc.
    -
  * - :code:`DeciLMForCausalLM`
    - DeciLM
    - :code:`Deci/DeciLM-7B`, :code:`Deci/DeciLM-7B-instruct`, etc.
    -
  * - :code:`FalconForCausalLM`
    - Falcon
    - :code:`tiiuae/falcon-7b`, :code:`tiiuae/falcon-40b`, :code:`tiiuae/falcon-rw-7b`, etc.
    -
  * - :code:`GemmaForCausalLM`
    - Gemma
    - :code:`google/gemma-2b`, :code:`google/gemma-7b`, etc.
    - ✅︎
  * - :code:`Gemma2ForCausalLM`
    - Gemma2
    - :code:`google/gemma-2-9b`, :code:`google/gemma-2-27b`, etc.
    - ✅︎
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl`, etc.
    -
  * - :code:`GPTBigCodeForCausalLM`
    - StarCoder, SantaCoder, WizardCoder
    - :code:`bigcode/starcoder`, :code:`bigcode/gpt_bigcode-santacoder`, :code:`WizardLM/WizardCoder-15B-V1.0`, etc.
    - ✅︎
  * - :code:`GPTJForCausalLM`
    - GPT-J
    - :code:`EleutherAI/gpt-j-6b`, :code:`nomic-ai/gpt4all-j`, etc.
    -
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b`, etc.
    -
  * - :code:`InternLMForCausalLM`
    - InternLM
    - :code:`internlm/internlm-7b`, :code:`internlm/internlm-chat-7b`, etc.
    - ✅︎
  * - :code:`InternLM2ForCausalLM`
    - InternLM2
    - :code:`internlm/internlm2-7b`, :code:`internlm/internlm2-chat-7b`, etc.
    -
  * - :code:`JAISLMHeadModel`
    - Jais
    - :code:`core42/jais-13b`, :code:`core42/jais-13b-chat`, :code:`core42/jais-30b-v3`, :code:`core42/jais-30b-chat-v3`, etc.
    -
  * - :code:`JambaForCausalLM`
    - Jamba
    - :code:`ai21labs/Jamba-v0.1`, etc.
    - ✅︎
  * - :code:`LlamaForCausalLM`
    - Llama 3.1, Llama 3, Llama 2, LLaMA, Yi
    - :code:`meta-llama/Meta-Llama-3.1-405B-Instruct`, :code:`meta-llama/Meta-Llama-3.1-70B`, :code:`meta-llama/Meta-Llama-3-70B-Instruct`, :code:`meta-llama/Llama-2-70b-hf`, :code:`01-ai/Yi-34B`, etc.
    - ✅︎
  * - :code:`MiniCPMForCausalLM`
    - MiniCPM
    - :code:`openbmb/MiniCPM-2B-sft-bf16`, :code:`openbmb/MiniCPM-2B-dpo-bf16`, etc.
    -
  * - :code:`MistralForCausalLM`
    - Mistral, Mistral-Instruct
    - :code:`mistralai/Mistral-7B-v0.1`, :code:`mistralai/Mistral-7B-Instruct-v0.1`, etc.
    - ✅︎
  * - :code:`MixtralForCausalLM`
    - Mixtral-8x7B, Mixtral-8x7B-Instruct
    - :code:`mistralai/Mixtral-8x7B-v0.1`, :code:`mistralai/Mixtral-8x7B-Instruct-v0.1`, :code:`mistral-community/Mixtral-8x22B-v0.1`, etc.
    - ✅︎
  * - :code:`MPTForCausalLM`
    - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
    - :code:`mosaicml/mpt-7b`, :code:`mosaicml/mpt-7b-storywriter`, :code:`mosaicml/mpt-30b`, etc.
    -
  * - :code:`NemotronForCausalLM`
    - Nemotron-3, Nemotron-4, Minitron
    - :code:`nvidia/Minitron-8B-Base`, :code:`mgoin/Nemotron-4-340B-Base-hf-FP8`, etc.
    - ✅︎
  * - :code:`OLMoForCausalLM`
    - OLMo
    - :code:`allenai/OLMo-1B-hf`, :code:`allenai/OLMo-7B-hf`, etc.
    -
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b`, etc.
    -
  * - :code:`OrionForCausalLM`
    - Orion
    - :code:`OrionStarAI/Orion-14B-Base`, :code:`OrionStarAI/Orion-14B-Chat`, etc.
    -
  * - :code:`PhiForCausalLM`
    - Phi
    - :code:`microsoft/phi-1_5`, :code:`microsoft/phi-2`, etc.
    - ✅︎
  * - :code:`Phi3ForCausalLM`
    - Phi-3
    - :code:`microsoft/Phi-3-mini-4k-instruct`, :code:`microsoft/Phi-3-mini-128k-instruct`, :code:`microsoft/Phi-3-medium-128k-instruct`, etc.
    -
  * - :code:`Phi3SmallForCausalLM`
    - Phi-3-Small
    - :code:`microsoft/Phi-3-small-8k-instruct`, :code:`microsoft/Phi-3-small-128k-instruct`, etc.
    -
  * - :code:`PersimmonForCausalLM`
    - Persimmon
    - :code:`adept/persimmon-8b-base`, :code:`adept/persimmon-8b-chat`, etc.
    - 
  * - :code:`QWenLMHeadModel`
    - Qwen
    - :code:`Qwen/Qwen-7B`, :code:`Qwen/Qwen-7B-Chat`, etc.
    -
  * - :code:`Qwen2ForCausalLM`
    - Qwen2
    - :code:`Qwen/Qwen2-beta-7B`, :code:`Qwen/Qwen2-beta-7B-Chat`, etc.
    - ✅︎
  * - :code:`Qwen2MoeForCausalLM`
    - Qwen2MoE
    - :code:`Qwen/Qwen1.5-MoE-A2.7B`, :code:`Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.
    -
  * - :code:`StableLmForCausalLM`
    - StableLM
    - :code:`stabilityai/stablelm-3b-4e1t/` , :code:`stabilityai/stablelm-base-alpha-7b-v2`, etc.
    -
  * - :code:`Starcoder2ForCausalLM`
    - Starcoder2
    - :code:`bigcode/starcoder2-3b`, :code:`bigcode/starcoder2-7b`, :code:`bigcode/starcoder2-15b`, etc.
    -
  * - :code:`XverseForCausalLM`
    - Xverse
    - :code:`xverse/XVERSE-7B-Chat`, :code:`xverse/XVERSE-13B-Chat`, :code:`xverse/XVERSE-65B-Chat`, etc.
    -

.. note::
    Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.

.. _supported_vlms:

Vision Language Models
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
  :widths: 25 25 50 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HuggingFace Models
    - :ref:`LoRA <lora>`
  * - :code:`Blip2ForConditionalGeneration`
    - BLIP-2
    - :code:`Salesforce/blip2-opt-2.7b`, :code:`Salesforce/blip2-opt-6.7b`, etc.
    -
  * - :code:`ChameleonForConditionalGeneration`
    - Chameleon
    - :code:`facebook/chameleon-7b` etc.
    - 
  * - :code:`FuyuForCausalLM`
    - Fuyu
    - :code:`adept/fuyu-8b` etc.
    - 
  * - :code:`InternVLChatModel`
    - InternVL2
    - :code:`OpenGVLab/InternVL2-4B`, :code:`OpenGVLab/InternVL2-8B`, etc.
    - 
  * - :code:`LlavaForConditionalGeneration`
    - LLaVA-1.5
    - :code:`llava-hf/llava-1.5-7b-hf`, :code:`llava-hf/llava-1.5-13b-hf`, etc.
    -
  * - :code:`LlavaNextForConditionalGeneration`
    - LLaVA-NeXT
    - :code:`llava-hf/llava-v1.6-mistral-7b-hf`, :code:`llava-hf/llava-v1.6-vicuna-7b-hf`, etc.
    -
  * - :code:`PaliGemmaForConditionalGeneration`
    - PaliGemma
    - :code:`google/paligemma-3b-pt-224`, :code:`google/paligemma-3b-mix-224`, etc.
    - 
  * - :code:`Phi3VForCausalLM`
    - Phi-3-Vision
    - :code:`microsoft/Phi-3-vision-128k-instruct`, etc.
    -
  * - :code:`MiniCPMV`
    - MiniCPM-V
    - :code:`openbmb/MiniCPM-V-2` (see note), :code:`openbmb/MiniCPM-Llama3-V-2_5`, :code:`openbmb/MiniCPM-V-2_6`, etc.
    -

.. note::
  For :code:`openbmb/MiniCPM-V-2`, the official repo doesn't work yet, so we need to use a fork (:code:`HwwwH/MiniCPM-V-2`) for now.
  For more details, please see: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630

----

If your model uses one of the above model architectures, you can seamlessly run your model with vLLM.
Otherwise, please refer to :ref:`Adding a New Model <adding_a_new_model>` and :ref:`Enabling Multimodal Inputs <enabling_multimodal_inputs>` 
for instructions on how to implement support for your model.
Alternatively, you can raise an issue on our `GitHub <https://github.com/vllm-project/vllm/issues>`_ project.

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


Model Support Policy
=====================

At vLLM, we are committed to facilitating the integration and support of third-party models within our ecosystem. Our approach is designed to balance the need for robustness and the practical limitations of supporting a wide range of models. Here’s how we manage third-party model support:

1. **Community-Driven Support**: We encourage community contributions for adding new models. When a user requests support for a new model, we welcome pull requests (PRs) from the community. These contributions are evaluated primarily on the sensibility of the output they generate, rather than strict consistency with existing implementations such as those in transformers. **Call for contribution:** PRs coming directly from model vendors are greatly appreciated!

2. **Best-Effort Consistency**: While we aim to maintain a level of consistency between the models implemented in vLLM and other frameworks like transformers, complete alignment is not always feasible. Factors like acceleration techniques and the use of low-precision computations can introduce discrepancies. Our commitment is to ensure that the implemented models are functional and produce sensible results.

3. **Issue Resolution and Model Updates**: Users are encouraged to report any bugs or issues they encounter with third-party models. Proposed fixes should be submitted via PRs, with a clear explanation of the problem and the rationale behind the proposed solution. If a fix for one model impacts another, we rely on the community to highlight and address these cross-model dependencies. Note: for bugfix PRs, it is good etiquette to inform the original author to seek their feedback.

4. **Monitoring and Updates**: Users interested in specific models should monitor the commit history for those models (e.g., by tracking changes in the main/vllm/model_executor/models directory). This proactive approach helps users stay informed about updates and changes that may affect the models they use.

5. **Selective Focus**: Our resources are primarily directed towards models with significant user interest and impact. Models that are less frequently used may receive less attention, and we rely on the community to play a more active role in their upkeep and improvement.

Through this approach, vLLM fosters a collaborative environment where both the core development team and the broader community contribute to the robustness and diversity of the third-party models supported in our ecosystem.

Note that, as an inference engine, vLLM does not introduce new models. Therefore, all models supported by vLLM are third-party models in this regard.

We have the following levels of testing for models:

1. **Strict Consistency**: We compare the output of the model with the output of the model in the HuggingFace Transformers library under greedy decoding. This is the most stringent test. Please refer to `test_models.py <https://github.com/vllm-project/vllm/blob/main/tests/models/test_models.py>`_ and `test_big_models.py <https://github.com/vllm-project/vllm/blob/main/tests/models/test_big_models.py>`_ for the models that have passed this test.
2. **Output Sensibility**: We check if the output of the model is sensible and coherent, by measuring the perplexity of the output and checking for any obvious errors. This is a less stringent test.
3. **Runtime Functionality**: We check if the model can be loaded and run without errors. This is the least stringent test. Please refer to `functionality tests <https://github.com/vllm-project/vllm/tree/main/tests>`_ and `examples <https://github.com/vllm-project/vllm/tree/main/examples>`_ for the models that have passed this test.
4. **Community Feedback**: We rely on the community to provide feedback on the models. If a model is broken or not working as expected, we encourage users to raise issues to report it or open pull requests to fix it. The rest of the models fall under this category.
