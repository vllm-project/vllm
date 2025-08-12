(supported-models)=

# List of Supported Models

vLLM supports generative and pooling models across various tasks.
If a model supports more than one task, you can set the task via the `--task` argument.

For each task, we list the model architectures that have been implemented in vLLM.
Alongside each architecture, we include some popular models that use it.

## Loading a Model

### HuggingFace Hub

By default, vLLM loads models from [HuggingFace (HF) Hub](https://huggingface.co/models).

To determine whether a given model is supported, you can check the `config.json` file inside the HF repository.
If the `"architectures"` field contains a model architecture listed below, then it should be supported in theory.

````{tip}
The easiest way to check if your model is really supported at runtime is to run the program below:

```python
from vllm import LLM

# For generative models (task=generate) only
llm = LLM(model=..., task="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (task={embed,classify,reward,score}) only
llm = LLM(model=..., task="embed")  # Name or path of your model
output = llm.encode("Hello, my name is")
print(output)
```

If vLLM successfully returns text (for generative models) or hidden states (for pooling models), it indicates that your model is supported.
````

Otherwise, please refer to [Adding a New Model](#new-model) for instructions on how to implement your model in vLLM.
Alternatively, you can [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) to request vLLM support.

### ModelScope

To use models from [ModelScope](https://www.modelscope.cn) instead of HuggingFace Hub, set an environment variable:

```shell
export VLLM_USE_MODELSCOPE=True
```

And use with `trust_remote_code=True`.

```python
from vllm import LLM

llm = LLM(model=..., revision=..., task=..., trust_remote_code=True)

# For generative models (task=generate) only
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (task={embed,classify,reward,score}) only
output = llm.encode("Hello, my name is")
print(output)
```

## List of Text-only Language Models

### Generative Models

See [this page](#generative-models) for more information on how to use generative models.

#### Text Generation (`--task generate`)

```{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

* - Architecture
  - Models
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `AquilaForCausalLM`
  - Aquila, Aquila2
  - `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.
  - ✅︎
  - ✅︎
* - `ArcticForCausalLM`
  - Arctic
  - `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc.
  -
  - ✅︎
* - `BaiChuanForCausalLM`
  - Baichuan2, Baichuan
  - `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.
  - ✅︎
  - ✅︎
* - `BloomForCausalLM`
  - BLOOM, BLOOMZ, BLOOMChat
  - `bigscience/bloom`, `bigscience/bloomz`, etc.
  -
  - ✅︎
* - `BartForConditionalGeneration`
  - BART
  - `facebook/bart-base`, `facebook/bart-large-cnn`, etc.
  -
  -
* - `ChatGLMModel`
  - ChatGLM
  - `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.
  - ✅︎
  - ✅︎
* - `CohereForCausalLM`, `Cohere2ForCausalLM`
  - Command-R
  - `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024`, etc.
  - ✅︎
  - ✅︎
* - `DbrxForCausalLM`
  - DBRX
  - `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc.
  -
  - ✅︎
* - `DeciLMForCausalLM`
  - DeciLM
  - `Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.
  -
  - ✅︎
* - `DeepseekForCausalLM`
  - DeepSeek
  - `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat` etc.
  -
  - ✅︎
* - `DeepseekV2ForCausalLM`
  - DeepSeek-V2
  - `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat` etc.
  -
  - ✅︎
* - `DeepseekV3ForCausalLM`
  - DeepSeek-V3
  - `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3` etc.
  -
  - ✅︎
* - `ExaoneForCausalLM`
  - EXAONE-3
  - `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.
  - ✅︎
  - ✅︎
* - `FalconForCausalLM`
  - Falcon
  - `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.
  -
  - ✅︎
* - `FalconMambaForCausalLM`
  - FalconMamba
  - `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, etc.
  - ✅︎
  - ✅︎
* - `GemmaForCausalLM`
  - Gemma
  - `google/gemma-2b`, `google/gemma-7b`, etc.
  - ✅︎
  - ✅︎
* - `Gemma2ForCausalLM`
  - Gemma2
  - `google/gemma-2-9b`, `google/gemma-2-27b`, etc.
  - ✅︎
  - ✅︎
* - `GlmForCausalLM`
  - GLM-4
  - `THUDM/glm-4-9b-chat-hf`, etc.
  - ✅︎
  - ✅︎
* - `GPT2LMHeadModel`
  - GPT-2
  - `gpt2`, `gpt2-xl`, etc.
  -
  - ✅︎
* - `GPTBigCodeForCausalLM`
  - StarCoder, SantaCoder, WizardCoder
  - `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc.
  - ✅︎
  - ✅︎
* - `GPTJForCausalLM`
  - GPT-J
  - `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.
  -
  - ✅︎
* - `GPTNeoXForCausalLM`
  - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
  - `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.
  -
  - ✅︎
* - `GraniteForCausalLM`
  - Granite 3.0, Granite 3.1, PowerLM
  - `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b`, etc.
  - ✅︎
  - ✅︎
* - `GraniteMoeForCausalLM`
  - Granite 3.0 MoE, PowerMoE
  - `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`, etc.
  - ✅︎
  - ✅︎
* - `GritLM`
  - GritLM
  - `parasail-ai/GritLM-7B-vllm`.
  - ✅︎
  - ✅︎
* - `InternLMForCausalLM`
  - InternLM
  - `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.
  - ✅︎
  - ✅︎
* - `InternLM2ForCausalLM`
  - InternLM2
  - `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.
  - ✅︎
  - ✅︎
* - `JAISLMHeadModel`
  - Jais
  - `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3`, etc.
  -
  - ✅︎
* - `JambaForCausalLM`
  - Jamba
  - `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1`, etc.
  - ✅︎
  - ✅︎
* - `LlamaForCausalLM`
  - Llama 3.1, Llama 3, Llama 2, LLaMA, Yi
  - `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc.
  - ✅︎
  - ✅︎
* - `MambaForCausalLM`
  - Mamba
  - `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf`, etc.
  -
  - ✅︎
* - `MiniCPMForCausalLM`
  - MiniCPM
  - `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft`, etc.
  - ✅︎
  - ✅︎
* - `MiniCPM3ForCausalLM`
  - MiniCPM3
  - `openbmb/MiniCPM3-4B`, etc.
  - ✅︎
  - ✅︎
* - `MistralForCausalLM`
  - Mistral, Mistral-Instruct
  - `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.
  - ✅︎
  - ✅︎
* - `MixtralForCausalLM`
  - Mixtral-8x7B, Mixtral-8x7B-Instruct
  - `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.
  - ✅︎
  - ✅︎
* - `MPTForCausalLM`
  - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
  - `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc.
  -
  - ✅︎
* - `NemotronForCausalLM`
  - Nemotron-3, Nemotron-4, Minitron
  - `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8`, etc.
  - ✅︎
  - ✅︎
* - `OLMoForCausalLM`
  - OLMo
  - `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.
  -
  - ✅︎
* - `OLMo2ForCausalLM`
  - OLMo2
  - `allenai/OLMo2-7B-1124`, etc.
  -
  - ✅︎
* - `OLMoEForCausalLM`
  - OLMoE
  - `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct`, etc.
  - ✅︎
  - ✅︎
* - `OPTForCausalLM`
  - OPT, OPT-IML
  - `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.
  -
  - ✅︎
* - `OrionForCausalLM`
  - Orion
  - `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.
  -
  - ✅︎
* - `PhiForCausalLM`
  - Phi
  - `microsoft/phi-1_5`, `microsoft/phi-2`, etc.
  - ✅︎
  - ✅︎
* - `Phi3ForCausalLM`
  - Phi-3
  - `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct`, etc.
  - ✅︎
  - ✅︎
* - `Phi3SmallForCausalLM`
  - Phi-3-Small
  - `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct`, etc.
  -
  - ✅︎
* - `PhiMoEForCausalLM`
  - Phi-3.5-MoE
  - `microsoft/Phi-3.5-MoE-instruct`, etc.
  - ✅︎
  - ✅︎
* - `PersimmonForCausalLM`
  - Persimmon
  - `adept/persimmon-8b-base`, `adept/persimmon-8b-chat`, etc.
  -
  - ✅︎
* - `QWenLMHeadModel`
  - Qwen
  - `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.
  - ✅︎
  - ✅︎
* - `Qwen2ForCausalLM`
  - QwQ, Qwen2
  - `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc.
  - ✅︎
  - ✅︎
* - `Qwen2MoeForCausalLM`
  - Qwen2MoE
  - `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.
  -
  - ✅︎
* - `StableLmForCausalLM`
  - StableLM
  - `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.
  -
  - ✅︎
* - `Starcoder2ForCausalLM`
  - Starcoder2
  - `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.
  -
  - ✅︎
* - `SolarForCausalLM`
  - Solar Pro
  - `upstage/solar-pro-preview-instruct`, etc.
  - ✅︎
  - ✅︎
* - `TeleChat2ForCausalLM`
  - TeleChat2
  - `TeleAI/TeleChat2-3B`, `TeleAI/TeleChat2-7B`, `TeleAI/TeleChat2-35B`, etc.
  - ✅︎
  - ✅︎
* - `XverseForCausalLM`
  - XVERSE
  - `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.
  - ✅︎
  - ✅︎
```

```{note}
Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.
```

### Pooling Models

See [this page](pooling-models) for more information on how to use pooling models.

```{important}
Since some model architectures support both generative and pooling tasks,
you should explicitly specify the task type to ensure that the model is used in pooling mode instead of generative mode.
```

#### Text Embedding (`--task embed`)

```{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

* - Architecture
  - Models
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `BertModel`
  - BERT-based
  - `BAAI/bge-base-en-v1.5`, etc.
  -
  -
* - `Gemma2Model`
  - Gemma2-based
  - `BAAI/bge-multilingual-gemma2`, etc.
  -
  - ✅︎
* - `GritLM`
  - GritLM
  - `parasail-ai/GritLM-7B-vllm`.
  - ✅︎
  - ✅︎
* - `LlamaModel`, `LlamaForCausalLM`, `MistralModel`, etc.
  - Llama-based
  - `intfloat/e5-mistral-7b-instruct`, etc.
  - ✅︎
  - ✅︎
* - `Qwen2Model`, `Qwen2ForCausalLM`
  - Qwen2-based
  - `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc.
  - ✅︎
  - ✅︎
* - `RobertaModel`, `RobertaForMaskedLM`
  - RoBERTa-based
  - `sentence-transformers/all-roberta-large-v1`, `sentence-transformers/all-roberta-large-v1`, etc.
  -
  -
* - `XLMRobertaModel`
  - XLM-RoBERTa-based
  - `intfloat/multilingual-e5-large`, etc.
  -
  -
```

```{note}
`ssmits/Qwen2-7B-Instruct-embed-base` has an improperly defined Sentence Transformers config.
You should manually set mean pooling by passing `--override-pooler-config '{"pooling_type": "MEAN"}'`.
```

```{note}
Unlike base Qwen2, `Alibaba-NLP/gte-Qwen2-7B-instruct` uses bi-directional attention.
You can set `--hf-overrides '{"is_causal": false}'` to change the attention mask accordingly.

On the other hand, its 1.5B variant (`Alibaba-NLP/gte-Qwen2-1.5B-instruct`) uses causal attention
despite being described otherwise on its model card.

Regardless of the variant, you need to enable `--trust-remote-code` for the correct tokenizer to be
loaded. See [relevant issue on HF Transformers](https://github.com/huggingface/transformers/issues/34882).
```

If your model is not in the above list, we will try to automatically convert the model using
{func}`~vllm.model_executor.models.adapters.as_embedding_model`. By default, the embeddings
of the whole prompt are extracted from the normalized hidden state corresponding to the last token.

#### Reward Modeling (`--task reward`)

```{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

* - Architecture
  - Models
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `InternLM2ForRewardModel`
  - InternLM2-based
  - `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc.
  - ✅︎
  - ✅︎
* - `LlamaForCausalLM`
  - Llama-based
  - `peiyi9979/math-shepherd-mistral-7b-prm`, etc.
  - ✅︎
  - ✅︎
* - `Qwen2ForRewardModel`
  - Qwen2-based
  - `Qwen/Qwen2.5-Math-RM-72B`, etc.
  - ✅︎
  - ✅︎
```

If your model is not in the above list, we will try to automatically convert the model using
{func}`~vllm.model_executor.models.adapters.as_reward_model`. By default, we return the hidden states of each token directly.

```{important}
For process-supervised reward models such as `peiyi9979/math-shepherd-mistral-7b-prm`, the pooling config should be set explicitly,
e.g.: `--override-pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`.
```

#### Classification (`--task classify`)

```{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

* - Architecture
  - Models
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `JambaForSequenceClassification`
  - Jamba
  - `ai21labs/Jamba-tiny-reward-dev`, etc.
  - ✅︎
  - ✅︎
* - `Qwen2ForSequenceClassification`
  - Qwen2-based
  - `jason9693/Qwen2.5-1.5B-apeach`, etc.
  - ✅︎
  - ✅︎
```

If your model is not in the above list, we will try to automatically convert the model using
{func}`~vllm.model_executor.models.adapters.as_classification_model`. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

#### Sentence Pair Scoring (`--task score`)

```{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

* - Architecture
  - Models
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `BertForSequenceClassification`
  - BERT-based
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc.
  -
  -
* - `RobertaForSequenceClassification`
  - RoBERTa-based
  - `cross-encoder/quora-roberta-base`, etc.
  -
  -
* - `XLMRobertaForSequenceClassification`
  - XLM-RoBERTa-based
  - `BAAI/bge-reranker-v2-m3`, etc.
  -
  -
```

(supported-mm-models)=

## List of Multimodal Language Models

The following modalities are supported depending on the model:

- **T**ext
- **I**mage
- **V**ideo
- **A**udio

Any combination of modalities joined by `+` are supported.

- e.g.: `T + I` means that the model supports text-only, image-only, and text-with-image inputs.

On the other hand, modalities separated by `/` are mutually exclusive.

- e.g.: `T / I` means that the model supports text-only and image-only inputs, but not text-with-image inputs.

See [this page](#multimodal-inputs) on how to pass multi-modal inputs to the model.

````{important}
To enable multiple multi-modal items per text prompt, you have to set `limit_mm_per_prompt` (offline inference)
or `--limit-mm-per-prompt` (online serving). For example, to enable passing up to 4 images per text prompt:

Offline inference:
```python
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    limit_mm_per_prompt={"image": 4},
)
```

Online serving:
```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt image=4
```
````

```{note}
vLLM currently only supports adding LoRA to the language backbone of multimodal models.
```

### Generative Models

See [this page](#generative-models) for more information on how to use generative models.

#### Text Generation (`--task generate`)

```{list-table}
:widths: 25 25 15 20 5 5 5
:header-rows: 1

* - Architecture
  - Models
  - Inputs
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
  - [V1](gh-issue:8779)
* - `AriaForConditionalGeneration`
  - Aria
  - T + I<sup>+</sup>
  - `rhymes-ai/Aria`
  -
  - ✅︎
  - ✅︎
* - `Blip2ForConditionalGeneration`
  - BLIP-2
  - T + I<sup>E</sup>
  - `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`, etc.
  -
  - ✅︎
  - ✅︎
* - `ChameleonForConditionalGeneration`
  - Chameleon
  - T + I
  - `facebook/chameleon-7b` etc.
  -
  - ✅︎
  - ✅︎
* - `DeepseekVLV2ForCausalLM`
  - DeepSeek-VL2
  - T + I<sup>+</sup>
  - `deepseek-ai/deepseek-vl2-tiny`(WIP), `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2` etc. (see note)
  -
  - ✅︎
  - ✅︎
* - `FuyuForCausalLM`
  - Fuyu
  - T + I
  - `adept/fuyu-8b` etc.
  -
  - ✅︎
  - ✅︎
* - `ChatGLMModel`
  - GLM-4V
  - T + I
  - `THUDM/glm-4v-9b` etc.
  - ✅︎
  - ✅︎
  -
* - `H2OVLChatModel`
  - H2OVL
  - T + I<sup>E+</sup>
  - `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b`, etc.
  -
  - ✅︎
  -
* - `Idefics3ForConditionalGeneration`
  - Idefics3
  - T + I
  - `HuggingFaceM4/Idefics3-8B-Llama3` etc.
  - ✅︎
  -
  -
* - `InternVLChatModel`
  - InternVL 2.5, Mono-InternVL, InternVL 2.0
  - T + I<sup>E+</sup>
  - `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc.
  -
  - ✅︎
  - ✅︎
* - `LlavaForConditionalGeneration`
  - LLaVA-1.5
  - T + I<sup>E+</sup>
  - `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (see note), etc.
  -
  - ✅︎
  - ✅︎
* - `LlavaNextForConditionalGeneration`
  - LLaVA-NeXT
  - T + I<sup>E+</sup>
  - `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc.
  -
  - ✅︎
  - ✅︎
* - `LlavaNextVideoForConditionalGeneration`
  - LLaVA-NeXT-Video
  - T + V
  - `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc.
  -
  - ✅︎
  - ✅︎
* - `LlavaOnevisionForConditionalGeneration`
  - LLaVA-Onevision
  - T + I<sup>+</sup> + V<sup>+</sup>
  - `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc.
  -
  - ✅︎
  - ✅︎
* - `MiniCPMV`
  - MiniCPM-V
  - T + I<sup>E+</sup>
  - `openbmb/MiniCPM-V-2` (see note), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, etc.
  - ✅︎
  - ✅︎
  -
* - `MllamaForConditionalGeneration`
  - Llama 3.2
  - T + I<sup>+</sup>
  - `meta-llama/Llama-3.2-90B-Vision-Instruct`, `meta-llama/Llama-3.2-11B-Vision`, etc.
  -
  -
  -
* - `MolmoForCausalLM`
  - Molmo
  - T + I
  - `allenai/Molmo-7B-D-0924`, `allenai/Molmo-72B-0924`, etc.
  - ✅︎
  - ✅︎
  - ✅︎
* - `NVLM_D_Model`
  - NVLM-D 1.0
  - T + I<sup>E+</sup>
  - `nvidia/NVLM-D-72B`, etc.
  -
  - ✅︎
  - ✅︎
* - `PaliGemmaForConditionalGeneration`
  - PaliGemma, PaliGemma 2
  - T + I<sup>E</sup>
  - `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448`, etc.
  -
  - ✅︎
  -
* - `Phi3VForCausalLM`
  - Phi-3-Vision, Phi-3.5-Vision
  - T + I<sup>E+</sup>
  - `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct`, etc.
  -
  - ✅︎
  - ✅︎
* - `PixtralForConditionalGeneration`
  - Pixtral
  - T + I<sup>+</sup>
  - `mistralai/Pixtral-12B-2409`, `mistral-community/pixtral-12b` (see note), etc.
  -
  - ✅︎
  - ✅︎
* - `QWenLMHeadModel`
  - Qwen-VL
  - T + I<sup>E+</sup>
  - `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat`, etc.
  - ✅︎
  - ✅︎
  -
* - `Qwen2AudioForConditionalGeneration`
  - Qwen2-Audio
  - T + A<sup>+</sup>
  - `Qwen/Qwen2-Audio-7B-Instruct`
  -
  - ✅︎
  - ✅︎
* - `Qwen2VLForConditionalGeneration`
  - QVQ, Qwen2-VL
  - T + I<sup>E+</sup> + V<sup>E+</sup>
  - `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc.
  - ✅︎
  - ✅︎
  -
* - `UltravoxModel`
  - Ultravox
  - T + A<sup>E+</sup>
  - `fixie-ai/ultravox-v0_3`
  -
  - ✅︎
  - ✅︎
```

<sup>E</sup> Pre-computed embeddings can be inputted for this modality.  
<sup>+</sup> Multiple items can be inputted per text prompt for this modality.

````{note}
The `deepseek-ai/deepseek-vl2-tiny` is not supported yet.

To use `DeepSeek-VL2` series models, you need to install a fork version `deepseek_vl2` package:
```shell
pip install git+https://github.com/Isotr0py/DeepSeek-VL2.git
```

Besides, to run `DeepSeek-VL2` series models, you have to pass `--hf_overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'` when running vLLM.
````

```{note}
To use `TIGER-Lab/Mantis-8B-siglip-llama3`, you have to pass `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'` when running vLLM.
```

```{note}
The official `openbmb/MiniCPM-V-2` doesn't work yet, so we need to use a fork (`HwwwH/MiniCPM-V-2`) for now.
For more details, please see: <gh-pr:4087#issuecomment-2250397630>
```

```{note}
The chat template for Pixtral-HF is incorrect (see [discussion](https://huggingface.co/mistral-community/pixtral-12b/discussions/22)).
A corrected version is available at <gh-file:examples/template_pixtral_hf.jinja>.
```

### Pooling Models

See [this page](pooling-models) for more information on how to use pooling models.

```{important}
Since some model architectures support both generative and pooling tasks,
you should explicitly specify the task type to ensure that the model is used in pooling mode instead of generative mode.
```

#### Text Embedding (`--task embed`)

Any text generation model can be converted into an embedding model by passing `--task embed`.

```{note}
To get the best results, you should use pooling models that are specifically trained as such.
```

The following table lists those that are tested in vLLM.

```{list-table}
:widths: 25 25 15 25 5 5
:header-rows: 1

* - Architecture
  - Models
  - Inputs
  - Example HF Models
  - [LoRA](#lora-adapter)
  - [PP](#distributed-serving)
* - `LlavaNextForConditionalGeneration`
  - LLaVA-NeXT-based
  - T / I
  - `royokong/e5-v`
  -
  - ✅︎
* - `Phi3VForCausalLM`
  - Phi-3-Vision-based
  - T + I
  - `TIGER-Lab/VLM2Vec-Full`
  - 🚧
  - ✅︎
* - `Qwen2VLForConditionalGeneration`
  - Qwen2-VL-based
  - T + I
  - `MrLight/dse-qwen2-2b-mrl-v1`
  -
  - ✅︎
```

_________________

## Model Support Policy

At vLLM, we are committed to facilitating the integration and support of third-party models within our ecosystem. Our approach is designed to balance the need for robustness and the practical limitations of supporting a wide range of models. Here’s how we manage third-party model support:

1. **Community-Driven Support**: We encourage community contributions for adding new models. When a user requests support for a new model, we welcome pull requests (PRs) from the community. These contributions are evaluated primarily on the sensibility of the output they generate, rather than strict consistency with existing implementations such as those in transformers. **Call for contribution:** PRs coming directly from model vendors are greatly appreciated!

2. **Best-Effort Consistency**: While we aim to maintain a level of consistency between the models implemented in vLLM and other frameworks like transformers, complete alignment is not always feasible. Factors like acceleration techniques and the use of low-precision computations can introduce discrepancies. Our commitment is to ensure that the implemented models are functional and produce sensible results.

    ```{tip}
    When comparing the output of `model.generate` from HuggingFace Transformers with the output of `llm.generate` from vLLM, note that the former reads the model's generation config file (i.e., [generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)) and applies the default parameters for generation, while the latter only uses the parameters passed to the function. Ensure all sampling parameters are identical when comparing outputs.
    ```

3. **Issue Resolution and Model Updates**: Users are encouraged to report any bugs or issues they encounter with third-party models. Proposed fixes should be submitted via PRs, with a clear explanation of the problem and the rationale behind the proposed solution. If a fix for one model impacts another, we rely on the community to highlight and address these cross-model dependencies. Note: for bugfix PRs, it is good etiquette to inform the original author to seek their feedback.

4. **Monitoring and Updates**: Users interested in specific models should monitor the commit history for those models (e.g., by tracking changes in the main/vllm/model_executor/models directory). This proactive approach helps users stay informed about updates and changes that may affect the models they use.

5. **Selective Focus**: Our resources are primarily directed towards models with significant user interest and impact. Models that are less frequently used may receive less attention, and we rely on the community to play a more active role in their upkeep and improvement.

Through this approach, vLLM fosters a collaborative environment where both the core development team and the broader community contribute to the robustness and diversity of the third-party models supported in our ecosystem.

Note that, as an inference engine, vLLM does not introduce new models. Therefore, all models supported by vLLM are third-party models in this regard.

We have the following levels of testing for models:

1. **Strict Consistency**: We compare the output of the model with the output of the model in the HuggingFace Transformers library under greedy decoding. This is the most stringent test. Please refer to [models tests](https://github.com/vllm-project/vllm/blob/main/tests/models) for the models that have passed this test.
2. **Output Sensibility**: We check if the output of the model is sensible and coherent, by measuring the perplexity of the output and checking for any obvious errors. This is a less stringent test.
3. **Runtime Functionality**: We check if the model can be loaded and run without errors. This is the least stringent test. Please refer to [functionality tests](gh-dir:tests) and [examples](gh-dir:main/examples) for the models that have passed this test.
4. **Community Feedback**: We rely on the community to provide feedback on the models. If a model is broken or not working as expected, we encourage users to raise issues to report it or open pull requests to fix it. The rest of the models fall under this category.
