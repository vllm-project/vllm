.. _supported_models:

Supported Models
================

vLLM supports a variety of generative and embedding models from `HuggingFace (HF) Transformers <https://huggingface.co/models>`_.
This page lists the model architectures that are currently supported by vLLM.
Alongside each architecture, we include some popular models that use it.

For other models, you can check the :code:`config.json` file inside the model repository.
If the :code:`"architectures"` field contains a model architecture listed below, then it should be supported in theory.

.. tip::
    The easiest way to check if your model is really supported at runtime is to run the program below:

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

    If vLLM successfully generates text, it indicates that your model is supported.

Otherwise, please refer to :ref:`Adding a New Model <adding_a_new_model>` and :ref:`Enabling Multimodal Inputs <enabling_multimodal_inputs>` 
for instructions on how to implement your model in vLLM.
Alternatively, you can `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_ to request vLLM support.

.. note::
    To use models from `ModelScope <https://www.modelscope.cn>`_ instead of HuggingFace Hub, set an environment variable:

    .. code-block:: shell

       $ export VLLM_USE_MODELSCOPE=True

    And use with :code:`trust_remote_code=True`.

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=..., revision=..., trust_remote_code=True)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

Text-only Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^

Text Generation
---------------

.. list-table::
  :widths: 25 25 50 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`AquilaForCausalLM`
    - Aquila, Aquila2
    - :code:`BAAI/Aquila-7B`, :code:`BAAI/AquilaChat-7B`, etc.
    - ✅︎
    - ✅︎
  * - :code:`ArcticForCausalLM`
    - Arctic
    - :code:`Snowflake/snowflake-arctic-base`, :code:`Snowflake/snowflake-arctic-instruct`, etc.
    -
    - ✅︎
  * - :code:`BaiChuanForCausalLM`
    - Baichuan2, Baichuan
    - :code:`baichuan-inc/Baichuan2-13B-Chat`, :code:`baichuan-inc/Baichuan-7B`, etc.
    - ✅︎
    - ✅︎
  * - :code:`BloomForCausalLM`
    - BLOOM, BLOOMZ, BLOOMChat
    - :code:`bigscience/bloom`, :code:`bigscience/bloomz`, etc.
    -
    - ✅︎
  * - :code:`BartForConditionalGeneration`
    - BART
    - :code:`facebook/bart-base`, :code:`facebook/bart-large-cnn`, etc.
    - 
    - 
  * - :code:`ChatGLMModel`
    - ChatGLM
    - :code:`THUDM/chatglm2-6b`, :code:`THUDM/chatglm3-6b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`CohereForCausalLM`
    - Command-R
    - :code:`CohereForAI/c4ai-command-r-v01`, etc.
    - ✅︎
    - ✅︎
  * - :code:`DbrxForCausalLM`
    - DBRX
    - :code:`databricks/dbrx-base`, :code:`databricks/dbrx-instruct`, etc.
    -
    - ✅︎
  * - :code:`DeciLMForCausalLM`
    - DeciLM
    - :code:`Deci/DeciLM-7B`, :code:`Deci/DeciLM-7B-instruct`, etc.
    -
    - ✅︎
  * - :code:`DeepseekForCausalLM`
    - DeepSeek
    - :code:`deepseek-ai/deepseek-llm-67b-base`, :code:`deepseek-ai/deepseek-llm-7b-chat` etc.
    - 
    - ✅︎
  * - :code:`DeepseekV2ForCausalLM`
    - DeepSeek-V2
    - :code:`deepseek-ai/DeepSeek-V2`, :code:`deepseek-ai/DeepSeek-V2-Chat` etc.
    - 
    - ✅︎
  * - :code:`ExaoneForCausalLM`
    - EXAONE-3
    - :code:`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`FalconForCausalLM`
    - Falcon
    - :code:`tiiuae/falcon-7b`, :code:`tiiuae/falcon-40b`, :code:`tiiuae/falcon-rw-7b`, etc.
    -
    - ✅︎
  * - :code:`FalconMambaForCausalLM`
    - FalconMamba
    - :code:`tiiuae/falcon-mamba-7b`, :code:`tiiuae/falcon-mamba-7b-instruct`, etc.
    - ✅︎
    -  
  * - :code:`GemmaForCausalLM`
    - Gemma
    - :code:`google/gemma-2b`, :code:`google/gemma-7b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Gemma2ForCausalLM`
    - Gemma2
    - :code:`google/gemma-2-9b`, :code:`google/gemma-2-27b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl`, etc.
    -
    - ✅︎
  * - :code:`GPTBigCodeForCausalLM`
    - StarCoder, SantaCoder, WizardCoder
    - :code:`bigcode/starcoder`, :code:`bigcode/gpt_bigcode-santacoder`, :code:`WizardLM/WizardCoder-15B-V1.0`, etc.
    - ✅︎
    - ✅︎
  * - :code:`GPTJForCausalLM`
    - GPT-J
    - :code:`EleutherAI/gpt-j-6b`, :code:`nomic-ai/gpt4all-j`, etc.
    -
    - ✅︎
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b`, etc.
    -
    - ✅︎
  * - :code:`GraniteForCausalLM`
    - Granite 3.0, PowerLM
    - :code:`ibm-granite/granite-3.0-2b-base`, :code:`ibm-granite/granite-3.0-8b-instruct`, :code:`ibm/PowerLM-3b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`GraniteMoeForCausalLM`
    - Granite 3.0 MoE, PowerMoE
    - :code:`ibm-granite/granite-3.0-1b-a400m-base`, :code:`ibm-granite/granite-3.0-3b-a800m-instruct`, :code:`ibm/PowerMoE-3b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`InternLMForCausalLM`
    - InternLM
    - :code:`internlm/internlm-7b`, :code:`internlm/internlm-chat-7b`, etc.
    - ✅︎
    - ✅︎
  * - :code:`InternLM2ForCausalLM`
    - InternLM2
    - :code:`internlm/internlm2-7b`, :code:`internlm/internlm2-chat-7b`, etc.
    -
    - ✅︎
  * - :code:`JAISLMHeadModel`
    - Jais
    - :code:`inceptionai/jais-13b`, :code:`inceptionai/jais-13b-chat`, :code:`inceptionai/jais-30b-v3`, :code:`inceptionai/jais-30b-chat-v3`, etc.
    -
    - ✅︎
  * - :code:`JambaForCausalLM`
    - Jamba
    - :code:`ai21labs/AI21-Jamba-1.5-Large`, :code:`ai21labs/AI21-Jamba-1.5-Mini`, :code:`ai21labs/Jamba-v0.1`, etc.
    - ✅︎
    - 
  * - :code:`LlamaForCausalLM`
    - Llama 3.1, Llama 3, Llama 2, LLaMA, Yi
    - :code:`meta-llama/Meta-Llama-3.1-405B-Instruct`, :code:`meta-llama/Meta-Llama-3.1-70B`, :code:`meta-llama/Meta-Llama-3-70B-Instruct`, :code:`meta-llama/Llama-2-70b-hf`, :code:`01-ai/Yi-34B`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MambaForCausalLM`
    - Mamba
    - :code:`state-spaces/mamba-130m-hf`, :code:`state-spaces/mamba-790m-hf`, :code:`state-spaces/mamba-2.8b-hf`, etc.
    -
    -
  * - :code:`MiniCPMForCausalLM`
    - MiniCPM
    - :code:`openbmb/MiniCPM-2B-sft-bf16`, :code:`openbmb/MiniCPM-2B-dpo-bf16`, :code:`openbmb/MiniCPM-S-1B-sft`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MiniCPM3ForCausalLM`
    - MiniCPM3
    - :code:`openbmb/MiniCPM3-4B`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MistralForCausalLM`
    - Mistral, Mistral-Instruct
    - :code:`mistralai/Mistral-7B-v0.1`, :code:`mistralai/Mistral-7B-Instruct-v0.1`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MixtralForCausalLM`
    - Mixtral-8x7B, Mixtral-8x7B-Instruct
    - :code:`mistralai/Mixtral-8x7B-v0.1`, :code:`mistralai/Mixtral-8x7B-Instruct-v0.1`, :code:`mistral-community/Mixtral-8x22B-v0.1`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MPTForCausalLM`
    - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
    - :code:`mosaicml/mpt-7b`, :code:`mosaicml/mpt-7b-storywriter`, :code:`mosaicml/mpt-30b`, etc.
    -
    - ✅︎
  * - :code:`NemotronForCausalLM`
    - Nemotron-3, Nemotron-4, Minitron
    - :code:`nvidia/Minitron-8B-Base`, :code:`mgoin/Nemotron-4-340B-Base-hf-FP8`, etc.
    - ✅︎
    - ✅︎
  * - :code:`OLMoForCausalLM`
    - OLMo
    - :code:`allenai/OLMo-1B-hf`, :code:`allenai/OLMo-7B-hf`, etc.
    -
    - ✅︎
  * - :code:`OLMoEForCausalLM`
    - OLMoE
    - :code:`allenai/OLMoE-1B-7B-0924`, :code:`allenai/OLMoE-1B-7B-0924-Instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b`, etc.
    -
    - ✅︎
  * - :code:`OrionForCausalLM`
    - Orion
    - :code:`OrionStarAI/Orion-14B-Base`, :code:`OrionStarAI/Orion-14B-Chat`, etc.
    -
    - ✅︎
  * - :code:`PhiForCausalLM`
    - Phi
    - :code:`microsoft/phi-1_5`, :code:`microsoft/phi-2`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Phi3ForCausalLM`
    - Phi-3
    - :code:`microsoft/Phi-3-mini-4k-instruct`, :code:`microsoft/Phi-3-mini-128k-instruct`, :code:`microsoft/Phi-3-medium-128k-instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Phi3SmallForCausalLM`
    - Phi-3-Small
    - :code:`microsoft/Phi-3-small-8k-instruct`, :code:`microsoft/Phi-3-small-128k-instruct`, etc.
    -
    - ✅︎
  * - :code:`PhiMoEForCausalLM`
    - Phi-3.5-MoE
    - :code:`microsoft/Phi-3.5-MoE-instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`PersimmonForCausalLM`
    - Persimmon
    - :code:`adept/persimmon-8b-base`, :code:`adept/persimmon-8b-chat`, etc.
    - 
    - ✅︎
  * - :code:`QWenLMHeadModel`
    - Qwen
    - :code:`Qwen/Qwen-7B`, :code:`Qwen/Qwen-7B-Chat`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Qwen2ForCausalLM`
    - Qwen2
    - :code:`Qwen/Qwen2-7B-Instruct`, :code:`Qwen/Qwen2-7B`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Qwen2MoeForCausalLM`
    - Qwen2MoE
    - :code:`Qwen/Qwen1.5-MoE-A2.7B`, :code:`Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.
    -
    - ✅︎
  * - :code:`StableLmForCausalLM`
    - StableLM
    - :code:`stabilityai/stablelm-3b-4e1t`, :code:`stabilityai/stablelm-base-alpha-7b-v2`, etc.
    -
    - ✅︎
  * - :code:`Starcoder2ForCausalLM`
    - Starcoder2
    - :code:`bigcode/starcoder2-3b`, :code:`bigcode/starcoder2-7b`, :code:`bigcode/starcoder2-15b`, etc.
    -
    - ✅︎
  * - :code:`SolarForCausalLM`
    - Solar Pro
    - :code:`upstage/solar-pro-preview-instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`XverseForCausalLM`
    - XVERSE
    - :code:`xverse/XVERSE-7B-Chat`, :code:`xverse/XVERSE-13B-Chat`, :code:`xverse/XVERSE-65B-Chat`, etc.
    - ✅︎
    - ✅︎

.. note::
    Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.

Text Embedding
--------------

.. list-table::
  :widths: 25 25 50 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`Gemma2Model`
    - Gemma2-based
    - :code:`BAAI/bge-multilingual-gemma2`, etc.
    - 
    - ✅︎
  * - :code:`LlamaModel`, :code:`LlamaForCausalLM`, :code:`MistralModel`, etc.
    - Llama-based
    - :code:`intfloat/e5-mistral-7b-instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Qwen2Model`, :code:`Qwen2ForCausalLM`
    - Qwen2-based
    - :code:`ssmits/Qwen2-7B-Instruct-embed-base`, :code:`Alibaba-NLP/gte-Qwen2-1.5B-instruct`, etc.
    - ✅︎
    - ✅︎

.. important::
  Some model architectures support both generation and embedding tasks.
  In this case, you have to pass :code:`--task embedding` to run the model in embedding mode.

.. tip::
  You can override the model's pooling method by passing :code:`--override-pooler-config`.

Reward Modeling
---------------

.. list-table::
  :widths: 25 25 50 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`Qwen2ForRewardModel`
    - Qwen2-based
    - :code:`Qwen/Qwen2.5-Math-RM-72B`, etc.
    - ✅︎
    - ✅︎

.. note::
    As an interim measure, these models are supported in both offline and online inference via Embeddings API.

Classification
---------------

.. list-table::
  :widths: 25 25 50 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`Qwen2ForSequenceClassification`
    - Qwen2-based
    - :code:`jason9693/Qwen2.5-1.5B-apeach`, etc.
    - ✅︎
    - ✅︎

.. note::
    As an interim measure, these models are supported in both offline and online inference via Embeddings API.


Multimodal Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following modalities are supported depending on the model:

- **T**\ ext
- **I**\ mage
- **V**\ ideo
- **A**\ udio

Any combination of modalities joined by :code:`+` are supported.

- e.g.: :code:`T + I` means that the model supports text-only, image-only, and text-with-image inputs.

On the other hand, modalities separated by :code:`/` are mutually exclusive.

- e.g.: :code:`T / I` means that the model supports text-only and image-only inputs, but not text-with-image inputs.

.. _supported_vlms:

Text Generation
---------------

.. list-table::
  :widths: 25 25 15 25 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Inputs
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`Blip2ForConditionalGeneration`
    - BLIP-2
    - T + I\ :sup:`E`
    - :code:`Salesforce/blip2-opt-2.7b`, :code:`Salesforce/blip2-opt-6.7b`, etc.
    -
    - ✅︎
  * - :code:`ChameleonForConditionalGeneration`
    - Chameleon
    - T + I
    - :code:`facebook/chameleon-7b` etc.
    - 
    - ✅︎
  * - :code:`FuyuForCausalLM`
    - Fuyu
    - T + I
    - :code:`adept/fuyu-8b` etc.
    - 
    - ✅︎
  * - :code:`ChatGLMModel`
    - GLM-4V
    - T + I
    - :code:`THUDM/glm-4v-9b` etc.
    - ✅︎
    - ✅︎
  * - :code:`H2OVLChatModel`
    - H2OVL
    - T + I\ :sup:`E+`
    - :code:`h2oai/h2ovl-mississippi-800m`, :code:`h2oai/h2ovl-mississippi-2b`, etc.
    - 
    - ✅︎
  * - :code:`Idefics3ForConditionalGeneration`
    - Idefics3
    - T + I
    - :code:`HuggingFaceM4/Idefics3-8B-Llama3` etc.
    - ✅︎
    - 
  * - :code:`InternVLChatModel`
    - InternVL2
    - T + I\ :sup:`E+`
    - :code:`OpenGVLab/Mono-InternVL-2B`, :code:`OpenGVLab/InternVL2-4B`, :code:`OpenGVLab/InternVL2-8B`, etc.
    - 
    - ✅︎
  * - :code:`LlavaForConditionalGeneration`
    - LLaVA-1.5
    - T + I\ :sup:`E+`
    - :code:`llava-hf/llava-1.5-7b-hf`, :code:`llava-hf/llava-1.5-13b-hf`, etc.
    -
    - ✅︎
  * - :code:`LlavaNextForConditionalGeneration`
    - LLaVA-NeXT
    - T + I\ :sup:`E+`
    - :code:`llava-hf/llava-v1.6-mistral-7b-hf`, :code:`llava-hf/llava-v1.6-vicuna-7b-hf`, etc.
    -
    - ✅︎
  * - :code:`LlavaNextVideoForConditionalGeneration`
    - LLaVA-NeXT-Video
    - T + V
    - :code:`llava-hf/LLaVA-NeXT-Video-7B-hf`, etc.
    -
    - ✅︎
  * - :code:`LlavaOnevisionForConditionalGeneration`
    - LLaVA-Onevision
    - T + I\ :sup:`+` + V\ :sup:`+`
    - :code:`llava-hf/llava-onevision-qwen2-7b-ov-hf`, :code:`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc.
    -
    - ✅︎
  * - :code:`MiniCPMV`
    - MiniCPM-V
    - T + I\ :sup:`E+`
    - :code:`openbmb/MiniCPM-V-2` (see note), :code:`openbmb/MiniCPM-Llama3-V-2_5`, :code:`openbmb/MiniCPM-V-2_6`, etc.
    - ✅︎
    - ✅︎
  * - :code:`MllamaForConditionalGeneration`
    - Llama 3.2
    - T + I\ :sup:`+`
    - :code:`meta-llama/Llama-3.2-90B-Vision-Instruct`, :code:`meta-llama/Llama-3.2-11B-Vision`, etc.
    -
    -
  * - :code:`MolmoForCausalLM`
    - Molmo
    - T + I
    - :code:`allenai/Molmo-7B-D-0924`, :code:`allenai/Molmo-72B-0924`, etc.
    -
    - ✅︎
  * - :code:`NVLM_D_Model`
    - NVLM-D 1.0
    - T + I\ :sup:`E+`
    - :code:`nvidia/NVLM-D-72B`, etc.
    - 
    - ✅︎
  * - :code:`PaliGemmaForConditionalGeneration`
    - PaliGemma
    - T + I\ :sup:`E`
    - :code:`google/paligemma-3b-pt-224`, :code:`google/paligemma-3b-mix-224`, etc.
    - 
    - ✅︎
  * - :code:`Phi3VForCausalLM`
    - Phi-3-Vision, Phi-3.5-Vision
    - T + I\ :sup:`E+`
    - :code:`microsoft/Phi-3-vision-128k-instruct`, :code:`microsoft/Phi-3.5-vision-instruct` etc.
    -
    - ✅︎
  * - :code:`PixtralForConditionalGeneration`
    - Pixtral
    - T + I\ :sup:`+`
    - :code:`mistralai/Pixtral-12B-2409`, :code:`mistral-community/pixtral-12b` etc.
    -
    - ✅︎
  * - :code:`QWenLMHeadModel`
    - Qwen-VL
    - T + I\ :sup:`E+`
    - :code:`Qwen/Qwen-VL`, :code:`Qwen/Qwen-VL-Chat`, etc.
    - ✅︎
    - ✅︎
  * - :code:`Qwen2AudioForConditionalGeneration`
    - Qwen2-Audio
    - T + A\ :sup:`+`
    - :code:`Qwen/Qwen2-Audio-7B-Instruct`
    -
    - ✅︎
  * - :code:`Qwen2VLForConditionalGeneration`
    - Qwen2-VL
    - T + I\ :sup:`E+` + V\ :sup:`E+`
    - :code:`Qwen/Qwen2-VL-2B-Instruct`, :code:`Qwen/Qwen2-VL-7B-Instruct`, :code:`Qwen/Qwen2-VL-72B-Instruct`, etc.
    - ✅︎
    - ✅︎
  * - :code:`UltravoxModel`
    - Ultravox
    - T + A\ :sup:`E+`
    - :code:`fixie-ai/ultravox-v0_3`
    -
    - ✅︎

| :sup:`E` Pre-computed embeddings can be inputted for this modality.
| :sup:`+` Multiple items can be inputted per text prompt for this modality.

.. note::
  vLLM currently only supports adding LoRA to the language backbone of multimodal models.               

.. note::
  For :code:`openbmb/MiniCPM-V-2`, the official repo doesn't work yet, so we need to use a fork (:code:`HwwwH/MiniCPM-V-2`) for now.
  For more details, please see: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630

Multimodal Embedding
--------------------

.. list-table::
  :widths: 25 25 15 25 5 5
  :header-rows: 1

  * - Architecture
    - Models
    - Inputs
    - Example HF Models
    - :ref:`LoRA <lora>`
    - :ref:`PP <distributed_serving>`
  * - :code:`LlavaNextForConditionalGeneration`
    - LLaVA-NeXT-based
    - T / I
    - :code:`royokong/e5-v`
    - 
    - ✅︎
  * - :code:`Phi3VForCausalLM`
    - Phi-3-Vision-based
    - T + I
    - :code:`TIGER-Lab/VLM2Vec-Full`
    - 🚧
    - ✅︎
  * - :code:`Qwen2VLForConditionalGeneration`
    - Qwen2-VL-based
    - T + I
    - :code:`MrLight/dse-qwen2-2b-mrl-v1`
    - 
    - ✅︎

.. important::
  Some model architectures support both generation and embedding tasks.
  In this case, you have to pass :code:`--task embedding` to run the model in embedding mode.

.. tip::
  You can override the model's pooling method by passing :code:`--override-pooler-config`.

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

1. **Strict Consistency**: We compare the output of the model with the output of the model in the HuggingFace Transformers library under greedy decoding. This is the most stringent test. Please refer to `models tests <https://github.com/vllm-project/vllm/blob/main/tests/models>`_ for the models that have passed this test.
2. **Output Sensibility**: We check if the output of the model is sensible and coherent, by measuring the perplexity of the output and checking for any obvious errors. This is a less stringent test.
3. **Runtime Functionality**: We check if the model can be loaded and run without errors. This is the least stringent test. Please refer to `functionality tests <https://github.com/vllm-project/vllm/tree/main/tests>`_ and `examples <https://github.com/vllm-project/vllm/tree/main/examples>`_ for the models that have passed this test.
4. **Community Feedback**: We rely on the community to provide feedback on the models. If a model is broken or not working as expected, we encourage users to raise issues to report it or open pull requests to fix it. The rest of the models fall under this category.
