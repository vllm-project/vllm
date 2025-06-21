---
title: Supported Models
---
[](){ #supported-models }

vLLM supports [generative](./generative_models.md) and [pooling](./pooling_models.md) models across various tasks.
If a model supports more than one task, you can set the task via the `--task` argument.

For each task, we list the model architectures that have been implemented in vLLM.
Alongside each architecture, we include some popular models that use it.

## Model Implementation

### vLLM

If vLLM natively supports a model, its implementation can be found in <gh-file:vllm/model_executor/models>.

These models are what we list in [supported-text-models][supported-text-models] and [supported-mm-models][supported-mm-models].

[](){ #transformers-backend }

### Transformers

vLLM also supports model implementations that are available in Transformers. This does not currently work for all models, but most decoder language models are supported, and vision language model support is planned!

To check if the modeling backend is Transformers, you can simply do this:

```python
from vllm import LLM
llm = LLM(model=..., task="generate")  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))
```

If it is `TransformersForCausalLM` then it means it's based on Transformers!

!!! tip
    You can force the use of `TransformersForCausalLM` by setting `model_impl="transformers"` for [offline-inference][offline-inference] or `--model-impl transformers` for the [openai-compatible-server][openai-compatible-server].

!!! note
    vLLM may not fully optimise the Transformers implementation so you may see degraded performance if comparing a native model to a Transformers model in vLLM.

#### Custom models

If a model is neither supported natively by vLLM or Transformers, it can still be used in vLLM!

For a model to be compatible with the Transformers backend for vLLM it must:

- be a Transformers compatible custom model (see [Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models)):
    * The model directory must have the correct structure (e.g. `config.json` is present).
    * `config.json` must contain `auto_map.AutoModel`.
- be a Transformers backend for vLLM compatible model (see [writing-custom-models][writing-custom-models]):
    * Customisation should be done in the base model (e.g. in `MyModel`, not `MyModelForCausalLM`).

If the compatible model is:

- on the Hugging Face Model Hub, simply set `trust_remote_code=True` for [offline-inference][offline-inference] or `--trust-remote-code` for the [openai-compatible-server][openai-compatible-server].
- in a local directory, simply pass directory path to `model=<MODEL_DIR>` for [offline-inference][offline-inference] or `vllm serve <MODEL_DIR>` for the [openai-compatible-server][openai-compatible-server].

This means that, with the Transformers backend for vLLM, new models can be used before they are officially supported in Transformers or vLLM!

[](){ #writing-custom-models }

#### Writing custom models

This section details the necessary modifications to make to a Transformers compatible custom model that make it compatible with the Transformers backend for vLLM. (We assume that a Transformers compatible custom model has already been created, see [Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models)).

To make your model compatible with the Transformers backend, it needs:

1. `kwargs` passed down through all modules from `MyModel` to `MyAttention`.
2. `MyAttention` must use `ALL_ATTENTION_FUNCTIONS` to call attention.
3. `MyModel` must contain `_supports_attention_backend = True`.

```python title="modeling_my_model.py"

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

Here is what happens in the background when this model is loaded:

1. The config is loaded.
2. `MyModel` Python class is loaded from the `auto_map` in config, and we check that the model `is_backend_compatible()`.
3. `MyModel` is loaded into `TransformersForCausalLM` (see <gh-file:vllm/model_executor/models/transformers.py>) which sets `self.config._attn_implementation = "vllm"` so that vLLM's attention layer is used.

That's it!

For your model to be compatible with vLLM's tensor parallel and/or pipeline parallel features, you must add `base_model_tp_plan` and/or `base_model_pp_plan` to your model's config class:

```python title="configuration_my_model.py"

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

- `base_model_tp_plan` is a `dict` that maps fully qualified layer name patterns to tensor parallel styles (currently only `"colwise"` and `"rowwise"` are supported).
- `base_model_pp_plan` is a `dict` that maps direct child layer names to `tuple`s of `list`s of `str`s:
    * You only need to do this for layers which are not present on all pipeline stages
    * vLLM assumes that there will be only one `nn.ModuleList`, which is distributed across the pipeline stages
    * The `list` in the first element of the `tuple` contains the names of the input arguments
    * The `list` in the last element of the `tuple` contains the names of the variables the layer outputs to in your modeling code

## Loading a Model

### Hugging Face Hub

By default, vLLM loads models from [Hugging Face (HF) Hub](https://huggingface.co/models). To change the download path for models, you can set the `HF_HOME` environment variable; for more details, refer to [their official documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome).

To determine whether a given model is natively supported, you can check the `config.json` file inside the HF repository.
If the `"architectures"` field contains a model architecture listed below, then it should be natively supported.

Models do not _need_ to be natively supported to be used in vLLM.
The [Transformers backend][transformers-backend] enables you to run models directly using their Transformers implementation (or even remote code on the Hugging Face Model Hub!).

!!! tip
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

Otherwise, please refer to [Adding a New Model][new-model] for instructions on how to implement your model in vLLM.
Alternatively, you can [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) to request vLLM support.

#### Download a model

If you prefer, you can use the Hugging Face CLI to [download a model](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download) or specific files from a model repository:

```console
# Download a model
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

# Specify a custom cache directory
huggingface-cli download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# Download a specific file from a model repo
huggingface-cli download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### List the downloaded models

Use the Hugging Face CLI to [manage models](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#scan-your-cache) stored in local cache:

```console
# List cached models
huggingface-cli scan-cache

# Show detailed (verbose) output
huggingface-cli scan-cache -v

# Specify a custom cache directory
huggingface-cli scan-cache --dir ~/.cache/huggingface/hub
```

#### Delete a cached model

Use the Hugging Face CLI to interactively [delete downloaded model](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#clean-your-cache) from the cache:

```console
# The `delete-cache` command requires extra dependencies to work with the TUI.
# Please run `pip install huggingface_hub[cli]` to install them.

# Launch the interactive TUI to select models to delete
$ huggingface-cli delete-cache
? Select revisions to delete: 1 revisions selected counting for 438.9M.
  ○ None of the following (if selected, nothing will be deleted).
Model BAAI/bge-base-en-v1.5 (438.9M, used 1 week ago)
❯ ◉ a5beb1e3: main # modified 1 week ago

Model BAAI/bge-large-en-v1.5 (1.3G, used 1 week ago)
  ○ d4aa6901: main # modified 1 week ago

Model BAAI/bge-reranker-base (1.1G, used 4 weeks ago)
  ○ 2cfc18c9: main # modified 4 weeks ago

Press <space> to select, <enter> to validate and <ctrl+c> to quit without modification.

# Need to confirm after selected
? Select revisions to delete: 1 revision(s) selected.
? 1 revisions selected counting for 438.9M. Confirm deletion ? Yes
Start deletion.
Done. Deleted 1 repo(s) and 0 revision(s) for a total of 438.9M.
```

#### Using a proxy

Here are some tips for loading/downloading models from Hugging Face using a proxy:

- Set the proxy globally for your session (or set it in the profile file):

```shell
export http_proxy=http://your.proxy.server:port
export https_proxy=http://your.proxy.server:port
```

- Set the proxy for just the current command:

```shell
https_proxy=http://your.proxy.server:port huggingface-cli download <model_name>

# or use vllm cmd directly
https_proxy=http://your.proxy.server:port  vllm serve <model_name> --disable-log-requests
```

- Set the proxy in Python interpreter:

```python
import os

os.environ['http_proxy'] = 'http://your.proxy.server:port'
os.environ['https_proxy'] = 'http://your.proxy.server:port'
```

### ModelScope

To use models from [ModelScope](https://www.modelscope.cn) instead of Hugging Face Hub, set an environment variable:

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

[](){ #feature-status-legend }

## Feature Status Legend

- ✅︎ indicates that the feature is supported for the model.

- 🚧 indicates that the feature is planned but not yet supported for the model.

- ⚠️ indicates that the feature is available but may have known issues or limitations.

[](){ #supported-text-models }

## List of Text-only Language Models

### Generative Models

See [this page][generative-models] for more information on how to use generative models.

#### Text Generation

Specified using `--task generate`.

| Architecture                                      | Models                                              | Example HF Models                                                                                                                                                            | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|---------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|
| `AquilaForCausalLM`                               | Aquila, Aquila2                                     | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.                                                                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `ArcticForCausalLM`                               | Arctic                                              | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc.                                                                                               |                        | ✅︎                          | ✅︎                     |
| `BaiChuanForCausalLM`                             | Baichuan2, Baichuan                                 | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.                                                                                                          | ✅︎                     | ✅︎                          | ✅︎                     |
| `BambaForCausalLM`                                | Bamba                                               | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B`                                                                                                                   | ✅︎                     | ✅︎                          |                       |
| `BloomForCausalLM`                                | BLOOM, BLOOMZ, BLOOMChat                            | `bigscience/bloom`, `bigscience/bloomz`, etc.                                                                                                                                |                        | ✅︎                          |                       |
| `BartForConditionalGeneration`                    | BART                                                | `facebook/bart-base`, `facebook/bart-large-cnn`, etc.                                                                                                                        |                        |                             |                       |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM                                             | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc.                                                                                                       | ✅︎                     | ✅︎                          | ✅︎                     |
| `CohereForCausalLM`, `Cohere2ForCausalLM`         | Command-R                                           | `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024`, etc.                                                                                               | ✅︎                     | ✅︎                          | ✅︎                     |
| `DbrxForCausalLM`                                 | DBRX                                                | `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc.                                                                                                                     |                        | ✅︎                          | ✅︎                     |
| `DeciLMForCausalLM`                               | DeciLM                                              | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc.                                                                                                                               | ✅︎                     | ✅︎                          | ✅︎                     |
| `DeepseekForCausalLM`                             | DeepSeek                                            | `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat` etc.                                                                                                 |                        | ✅︎                          | ✅︎                     |
| `DeepseekV2ForCausalLM`                           | DeepSeek-V2                                         | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat` etc.                                                                                                               |                        | ✅︎                          | ✅︎                     |
| `DeepseekV3ForCausalLM`                           | DeepSeek-V3                                         | `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3` etc.                                                                                                               |                        | ✅︎                          | ✅︎                     |
| `ExaoneForCausalLM`                               | EXAONE-3                                            | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.                                                                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `FalconForCausalLM`                               | Falcon                                              | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.                                                                                                         |                        | ✅︎                          | ✅︎                     |
| `FalconMambaForCausalLM`                          | FalconMamba                                         | `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, etc.                                                                                                            |                        | ✅︎                          | ✅︎                     |
| `FalconH1ForCausalLM`                             | Falcon-H1                                           | `tiiuae/Falcon-H1-34B-Base`, `tiiuae/Falcon-H1-34B-Instruct`, etc.                                                                                                           | ✅︎                     | ✅︎                          |                       |
| `GemmaForCausalLM`                                | Gemma                                               | `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc.                                                                                                                            | ✅︎                     | ✅︎                          | ✅︎                     |
| `Gemma2ForCausalLM`                               | Gemma 2                                             | `google/gemma-2-9b`, `google/gemma-2-27b`, etc.                                                                                                                              | ✅︎                     | ✅︎                          | ✅︎                     |
| `Gemma3ForCausalLM`                               | Gemma 3                                             | `google/gemma-3-1b-it`, etc.                                                                                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `GlmForCausalLM`                                  | GLM-4                                               | `THUDM/glm-4-9b-chat-hf`, etc.                                                                                                                                               | ✅︎                     | ✅︎                          | ✅︎                     |
| `Glm4ForCausalLM`                                 | GLM-4-0414                                          | `THUDM/GLM-4-32B-0414`, etc.                                                                                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `GPT2LMHeadModel`                                 | GPT-2                                               | `gpt2`, `gpt2-xl`, etc.                                                                                                                                                      |                        | ✅︎                          | ✅︎                     |
| `GPTBigCodeForCausalLM`                           | StarCoder, SantaCoder, WizardCoder                  | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc.                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `GPTJForCausalLM`                                 | GPT-J                                               | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.                                                                                                                            |                        | ✅︎                          | ✅︎                     |
| `GPTNeoXForCausalLM`                              | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc. |                        | ✅︎                          | ✅︎                     |
| `GraniteForCausalLM`                              | Granite 3.0, Granite 3.1, PowerLM                   | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b`, etc.                                                                             | ✅︎                     | ✅︎                          | ✅︎                     |
| `GraniteMoeForCausalLM`                           | Granite 3.0 MoE, PowerMoE                           | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`, etc.                                                                | ✅︎                     | ✅︎                          | ✅︎                     |
| `GraniteMoeHybridForCausalLM`                     | Granite 4.0 MoE Hybrid                              | `ibm-granite/granite-4.0-tiny-preview`, etc.                                                                                                                                 | ✅︎                     | ✅︎                          |                       |
| `GraniteMoeSharedForCausalLM`                     | Granite MoE Shared                                  | `ibm-research/moe-7b-1b-active-shared-experts` (test model)                                                                                                                  | ✅︎                     | ✅︎                          | ✅︎                     |
| `GritLM`                                          | GritLM                                              | `parasail-ai/GritLM-7B-vllm`.                                                                                                                                                | ✅︎                     | ✅︎                          |                       |
| `Grok1ModelForCausalLM`                           | Grok1                                               | `hpcai-tech/grok-1`.                                                                                                                                                         | ✅︎                     | ✅︎                          | ✅︎                     |
| `InternLMForCausalLM`                             | InternLM                                            | `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.                                                                                                                    | ✅︎                     | ✅︎                          | ✅︎                     |
| `InternLM2ForCausalLM`                            | InternLM2                                           | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.                                                                                                                  | ✅︎                     | ✅︎                          | ✅︎                     |
| `InternLM3ForCausalLM`                            | InternLM3                                           | `internlm/internlm3-8b-instruct`, etc.                                                                                                                                       | ✅︎                     | ✅︎                          | ✅︎                     |
| `JAISLMHeadModel`                                 | Jais                                                | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3`, etc.                                                         |                        | ✅︎                          | ✅︎                     |
| `JambaForCausalLM`                                | Jamba                                               | `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1`, etc.                                                                                 | ✅︎                     | ✅︎                          |                       |
| `LlamaForCausalLM`                                | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi              | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc.        | ✅︎                     | ✅︎                          | ✅︎                     |
| `MambaForCausalLM`                                | Mamba                                               | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf`, etc.                                                                               |                        | ✅︎                          |                       |
| `Mamba2ForCausalLM`                               | Mamba2                                              | `mistralai/Mamba-Codestral-7B-v0.1`, etc.                                                                                                                                   |                        | ✅︎                          |                       |
| `MiniCPMForCausalLM`                              | MiniCPM                                             | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft`, etc.                                                                               | ✅︎                     | ✅︎                          | ✅︎                     |
| `MiniCPM3ForCausalLM`                             | MiniCPM3                                            | `openbmb/MiniCPM3-4B`, etc.                                                                                                                                                  | ✅︎                     | ✅︎                          | ✅︎                     |
| `MistralForCausalLM`                              | Mistral, Mistral-Instruct                           | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.                                                                                                      | ✅︎                     | ✅︎                          | ✅︎                     |
| `MixtralForCausalLM`                              | Mixtral-8x7B, Mixtral-8x7B-Instruct                 | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.                                                          | ✅︎                     | ✅︎                          | ✅︎                     |
| `MPTForCausalLM`                                  | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter        | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc.                                                                                                   |                        | ✅︎                          | ✅︎                     |
| `NemotronForCausalLM`                             | Nemotron-3, Nemotron-4, Minitron                    | `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8`, etc.                                                                                                         | ✅︎                     | ✅︎                          | ✅︎                     |
| `NemotronHForCausalLM`                            | Nemotron-H                                          | `nvidia/Nemotron-H-8B-Base-8K`, `nvidia/Nemotron-H-47B-Base-8K`, `nvidia/Nemotron-H-56B-Base-8K`, etc.                                                                       | ✅︎                     | ✅︎                          |                       |
| `OLMoForCausalLM`                                 | OLMo                                                | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.                                                                                                                             |                        | ✅︎                          | ✅︎                     |
| `OLMo2ForCausalLM`                                | OLMo2                                               | `allenai/OLMo-2-0425-1B`, etc.                                                                                                                                               |                        | ✅︎                          | ✅︎                     |
| `OLMoEForCausalLM`                                | OLMoE                                               | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct`, etc.                                                                                                        |                        | ✅︎                          | ✅︎                     |
| `OPTForCausalLM`                                  | OPT, OPT-IML                                        | `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.                                                                                                                         |                        | ✅︎                          | ✅︎                     |
| `OrionForCausalLM`                                | Orion                                               | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.                                                                                                             |                        | ✅︎                          | ✅︎                     |
| `PhiForCausalLM`                                  | Phi                                                 | `microsoft/phi-1_5`, `microsoft/phi-2`, etc.                                                                                                                                 | ✅︎                     | ✅︎                          | ✅︎                     |
| `Phi3ForCausalLM`                                 | Phi-4, Phi-3                                        | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct`, etc.   | ✅︎                     | ✅︎                          | ✅︎                     |
| `Phi3SmallForCausalLM`                            | Phi-3-Small                                         | `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct`, etc.                                                                                             |                        | ✅︎                          | ✅︎                     |
| `PhiMoEForCausalLM`                               | Phi-3.5-MoE                                         | `microsoft/Phi-3.5-MoE-instruct`, etc.                                                                                                                                       | ✅︎                     | ✅︎                          | ✅︎                     |
| `PersimmonForCausalLM`                            | Persimmon                                           | `adept/persimmon-8b-base`, `adept/persimmon-8b-chat`, etc.                                                                                                                   |                        | ✅︎                          | ✅︎                     |
| `Plamo2ForCausalLM`                               | PLaMo2                                              | `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b`, etc.                                                                                                                                 |                        |                             |                       |
| `QWenLMHeadModel`                                 | Qwen                                                | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.                                                                                                                                    | ✅︎                     | ✅︎                          | ✅︎                     |
| `Qwen2ForCausalLM`                                | QwQ, Qwen2                                          | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc.                                                                                                      | ✅︎                     | ✅︎                          | ✅︎                     |
| `Qwen2MoeForCausalLM`                             | Qwen2MoE                                            | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.                                                                                                                |                        | ✅︎                          | ✅︎                     |
| `Qwen3ForCausalLM`                                | Qwen3                                               | `Qwen/Qwen3-8B`, etc.                                                                                                                                                        | ✅︎                     | ✅︎                          | ✅︎                     |
| `Qwen3MoeForCausalLM`                             | Qwen3MoE                                            | `Qwen/Qwen3-30B-A3B`, etc.                                                                                                                                                   |                        | ✅︎                          | ✅︎                     |
| `StableLmForCausalLM`                             | StableLM                                            | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.                                                                                                |                        |                             | ✅︎                     |
| `Starcoder2ForCausalLM`                           | Starcoder2                                          | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.                                                                                             |                        | ✅︎                          | ✅︎                     |
| `SolarForCausalLM`                                | Solar Pro                                           | `upstage/solar-pro-preview-instruct`, etc.                                                                                                                                   | ✅︎                     | ✅︎                          | ✅︎                     |
| `TeleChat2ForCausalLM`                            | TeleChat2                                           | `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B`, etc.                                                                                                | ✅︎                     | ✅︎                          | ✅︎                     |
| `TeleFLMForCausalLM`                              | TeleFLM                                             | `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM`, etc.                                                                                                                    | ✅︎                     | ✅︎                          | ✅︎                     |
| `XverseForCausalLM`                               | XVERSE                                              | `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.                                                                                            | ✅︎                     | ✅︎                          | ✅︎                     |
| `MiniMaxM1ForCausalLM`                        | MiniMax-Text                                        | `MiniMaxAI/MiniMax-M1-40k`, `MiniMaxAI/MiniMax-M1-80k`etc.                                                                                                                                  |                        |                             |                       |
| `MiniMaxText01ForCausalLM`                        | MiniMax-Text                                        | `MiniMaxAI/MiniMax-Text-01`, etc.                                                                                                                                            |                        |                             |                       |
| `Zamba2ForCausalLM`                               | Zamba2                                              | `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct`, etc.                                                                              |                        |                             |                       |

!!! note
    Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.

### Pooling Models

See [this page](./pooling_models.md) for more information on how to use pooling models.

!!! important
    Since some model architectures support both generative and pooling tasks,
    you should explicitly specify the task type to ensure that the model is used in pooling mode instead of generative mode.

#### Text Embedding

Specified using `--task embed`.

| Architecture                                           | Models              | Example HF Models                                                                                                   | [LoRA][lora-adapter] | [PP][distributed-serving] | [V1](gh-issue:8779)   |
|--------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------|----------------------|---------------------------|-----------------------|
| `BertModel`                                            | BERT-based          | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc.                                                |                      |                           |                       |
| `Gemma2Model`                                          | Gemma 2-based       | `BAAI/bge-multilingual-gemma2`, etc.                                                                                | ✅︎                   |                           |                       |
| `GritLM`                                               | GritLM              | `parasail-ai/GritLM-7B-vllm`.                                                                                       | ✅︎                   | ✅︎                        |                       |
| `GteModel`                                             | Arctic-Embed-2.0-M  | `Snowflake/snowflake-arctic-embed-m-v2.0`.                                                                          | ︎                     |                           |                       |
| `GteNewModel`                                          | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc.                                                                           | ︎                     | ︎                         |                       |
| `ModernBertModel`                                      | ModernBERT-based    | `Alibaba-NLP/gte-modernbert-base`, etc.                                                                             | ︎                     | ︎                         |                       |
| `NomicBertModel`                                       | Nomic BERT          | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc. | ︎                     | ︎                         |                       |
| `LlamaModel`, `LlamaForCausalLM`, `MistralModel`, etc. | Llama-based         | `intfloat/e5-mistral-7b-instruct`, etc.                                                                             | ✅︎                   | ✅︎                        |                       |
| `Qwen2Model`, `Qwen2ForCausalLM`                       | Qwen2-based         | `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc.              | ✅︎                   | ✅︎                        |                       |
| `Qwen3Model`, `Qwen3ForCausalLM`                       | Qwen3-based         | `Qwen/Qwen3-Embedding-0.6B`, etc.                                                                                   | ✅︎                   | ✅︎                        |                       |
| `RobertaModel`, `RobertaForMaskedLM`                   | RoBERTa-based       | `sentence-transformers/all-roberta-large-v1`, etc.                                                                  |                      |                           |                       |

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` has an improperly defined Sentence Transformers config.
    You need to manually set mean pooling by passing `--override-pooler-config '{"pooling_type": "MEAN"}'`.

!!! note
    For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
    See [relevant issue on HF Transformers](https://github.com/huggingface/transformers/issues/34882).

!!! note
    `jinaai/jina-embeddings-v3` supports multiple tasks through lora, while vllm temporarily only supports text-matching tasks by merging lora weights.

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewModel`. The name `NewModel` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewModel"]}'` to specify the use of the `GteNewModel` architecture.

If your model is not in the above list, we will try to automatically convert the model using
[as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model]. By default, the embeddings
of the whole prompt are extracted from the normalized hidden state corresponding to the last token.

#### Reward Modeling

Specified using `--task reward`.

| Architecture              | Models          | Example HF Models                                                      | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|---------------------------|-----------------|------------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|
| `InternLM2ForRewardModel` | InternLM2-based | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc. | ✅︎                     | ✅︎                          |                       |
| `LlamaForCausalLM`        | Llama-based     | `peiyi9979/math-shepherd-mistral-7b-prm`, etc.                         | ✅︎                     | ✅︎                          |                       |
| `Qwen2ForRewardModel`     | Qwen2-based     | `Qwen/Qwen2.5-Math-RM-72B`, etc.                                       | ✅︎                     | ✅︎                          |                       |

If your model is not in the above list, we will try to automatically convert the model using
[as_reward_model][vllm.model_executor.models.adapters.as_reward_model]. By default, we return the hidden states of each token directly.

!!! important
    For process-supervised reward models such as `peiyi9979/math-shepherd-mistral-7b-prm`, the pooling config should be set explicitly,
    e.g.: `--override-pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`.

#### Classification

Specified using `--task classify`.

| Architecture                     | Models   | Example HF Models                      | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|----------------------------------|----------|----------------------------------------|------------------------|-----------------------------|-----------------------|
| `JambaForSequenceClassification` | Jamba    | `ai21labs/Jamba-tiny-reward-dev`, etc. | ✅︎                     | ✅︎                          |                       |

If your model is not in the above list, we will try to automatically convert the model using
[as_classification_model][vllm.model_executor.models.adapters.as_classification_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

#### Sentence Pair Scoring

Specified using `--task score`.

| Architecture                          | Models            | Example HF Models                                                                    | [V1](gh-issue:8779)   |
|---------------------------------------|-------------------|--------------------------------------------------------------------------------------|-----------------------|
| `BertForSequenceClassification`       | BERT-based        | `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc.                                         |                       |
| `Qwen3ForSequenceClassification`      | Qwen3-based       | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B` (see note), etc. |                       |
| `RobertaForSequenceClassification`    | RoBERTa-based     | `cross-encoder/quora-roberta-base`, etc.                                             |                       |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa-based | `BAAI/bge-reranker-v2-m3`, etc.                                                      |                       |

!!! note
    Load the official original `Qwen3 Reranker` by using the following command. More information can be found at: <gh-file:examples/offline_inference/qwen3_reranker.py>.

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```
[](){ #supported-mm-models }

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

See [this page][multimodal-inputs] on how to pass multi-modal inputs to the model.

!!! important
    **To enable multiple multi-modal items per text prompt in vLLM V0**, you have to set `limit_mm_per_prompt` (offline inference)
    or `--limit-mm-per-prompt` (online serving). For example, to enable passing up to 4 images per text prompt:

    Offline inference:

    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen2-VL-7B-Instruct",
        limit_mm_per_prompt={"image": 4},
    )
    ```

    Online serving:

    ```bash
    vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt '{"image":4}'
    ```

    **This is no longer required if you are using vLLM V1.**

!!! note
    vLLM currently only supports adding LoRA to the language backbone of multimodal models.

### Generative Models

See [this page][generative-models] for more information on how to use generative models.

#### Text Generation

Specified using `--task generate`.

| Architecture                                 | Models                                                                   | Inputs                                                                | Example HF Models                                                                                                                                       | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|----------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|
| `AriaForConditionalGeneration`               | Aria                                                                     | T + I<sup>+</sup>                                                     | `rhymes-ai/Aria`                                                                                                                                        |                        |                             | ✅︎                    |
| `AyaVisionForConditionalGeneration`          | Aya Vision                                                               | T + I<sup>+</sup>                                                     | `CohereForAI/aya-vision-8b`, `CohereForAI/aya-vision-32b`, etc.                                                                                         |                        | ✅︎                          | ✅︎                    |
| `Blip2ForConditionalGeneration`              | BLIP-2                                                                   | T + I<sup>E</sup>                                                     | `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`, etc.                                                                                          |                        | ✅︎                          | ✅︎                    |
| `ChameleonForConditionalGeneration`          | Chameleon                                                                | T + I                                                                 | `facebook/chameleon-7b` etc.                                                                                                                            |                        | ✅︎                          | ✅︎                    |
| `DeepseekVLV2ForCausalLM`<sup>^</sup>        | DeepSeek-VL2                                                             | T + I<sup>+</sup>                                                     | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2` etc.                                                      |                        | ✅︎                          | ✅︎                    |
| `Florence2ForConditionalGeneration`          | Florence-2                                                               | T + I                                                                 | `microsoft/Florence-2-base`, `microsoft/Florence-2-large` etc.                                                                                          |                        |                             |                       |
| `FuyuForCausalLM`                            | Fuyu                                                                     | T + I                                                                 | `adept/fuyu-8b` etc.                                                                                                                                    |                        | ✅︎                          | ✅︎                    |
| `Gemma3ForConditionalGeneration`             | Gemma 3                                                                  | T + I<sup>+</sup>                                                     | `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, etc.                                                                                                   | ✅︎                     | ✅︎                          | ⚠️                    |
| `GLM4VForCausalLM`<sup>^</sup>               | GLM-4V                                                                   | T + I                                                                 | `THUDM/glm-4v-9b`, `THUDM/cogagent-9b-20241220` etc.                                                                                                    | ✅︎                     | ✅︎                          | ✅︎                    |
| `GraniteSpeechForConditionalGeneration`      | Granite Speech                                                           | T + A                                                                 | `ibm-granite/granite-speech-3.3-8b`                                                                                                                     | ✅︎                     | ✅︎                          | ✅︎                    |
| `H2OVLChatModel`                             | H2OVL                                                                    | T + I<sup>E+</sup>                                                    | `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b`, etc.                                                                                      |                        | ✅︎                          | ✅︎\*                  |
| `Idefics3ForConditionalGeneration`           | Idefics3                                                                 | T + I                                                                 | `HuggingFaceM4/Idefics3-8B-Llama3` etc.                                                                                                                 | ✅︎                     |                             |  ✅︎                   |
| `InternVLChatModel`                          | InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>)                                 | `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc.  | ✅︎                     | ✅︎                          | ✅︎                    |
| `KimiVLForConditionalGeneration`             | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking                               | T + I<sup>+</sup>                                                     | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking`                                                                                    |                        |                             | ✅︎                    |
| `Llama4ForConditionalGeneration`             | Llama 4                                                                  | T + I<sup>+</sup>                                                     | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, etc. |                        | ✅︎                          | ✅︎                    |
| `LlavaForConditionalGeneration`              | LLaVA-1.5                                                                | T + I<sup>E+</sup>                                                    | `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (see note), etc.                                                                        |                        | ✅︎                          | ✅︎                    |
| `LlavaNextForConditionalGeneration`          | LLaVA-NeXT                                                               | T + I<sup>E+</sup>                                                    | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc.                                                                           |                        | ✅︎                          | ✅︎                    |
| `LlavaNextVideoForConditionalGeneration`     | LLaVA-NeXT-Video                                                         | T + V                                                                 | `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc.                                                                                                                 |                        | ✅︎                          | ✅︎                    |
| `LlavaOnevisionForConditionalGeneration`     | LLaVA-Onevision                                                          | T + I<sup>+</sup> + V<sup>+</sup>                                     | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc.                                                            |                        | ✅︎                          | ✅︎                    |
| `MiniCPMO`                                   | MiniCPM-O                                                                | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup>                  | `openbmb/MiniCPM-o-2_6`, etc.                                                                                                                           | ✅︎                     | ✅︎                          | ✅︎                    |
| `MiniCPMV`                                   | MiniCPM-V                                                                | T + I<sup>E+</sup> + V<sup>E+</sup>                                   | `openbmb/MiniCPM-V-2` (see note), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, etc.                                                         | ✅︎                     |                             | ✅︎                    |
| `MiniMaxVL01ForConditionalGeneration`        | MiniMax-VL                                                               | T + I<sup>E+</sup>                                                    | `MiniMaxAI/MiniMax-VL-01`, etc.                                                                                                                         |                        | ✅︎                          | ✅︎                    |
| `Mistral3ForConditionalGeneration`           | Mistral3                                                                 | T + I<sup>+</sup>                                                     | `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, etc.                                                                                                   | ✅︎                     | ✅︎                          | ✅︎                    |
| `MllamaForConditionalGeneration`             | Llama 3.2                                                                | T + I<sup>+</sup>                                                     | `meta-llama/Llama-3.2-90B-Vision-Instruct`, `meta-llama/Llama-3.2-11B-Vision`, etc.                                                                     |                        |                             |                       |
| `MolmoForCausalLM`                           | Molmo                                                                    | T + I<sup>+</sup>                                                     | `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`, etc.                                                                                              | ✅︎                     | ✅︎                          | ✅︎                    |
| `NVLM_D_Model`                               | NVLM-D 1.0                                                               | T + I<sup>+</sup>                                                     | `nvidia/NVLM-D-72B`, etc.                                                                                                                               |                        | ✅︎                          | ✅︎                    |
| `Ovis`                                       | Ovis2, Ovis1.6                                                           | T + I<sup>+</sup>                                                     | `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B`, etc.                                                                                                 |                        | ✅︎                          | ✅︎                    |
| `PaliGemmaForConditionalGeneration`          | PaliGemma, PaliGemma 2                                                   | T + I<sup>E</sup>                                                     | `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448`, etc.                                                  |                        | ✅︎                          | ⚠️                    |
| `Phi3VForCausalLM`                           | Phi-3-Vision, Phi-3.5-Vision                                             | T + I<sup>E+</sup>                                                    | `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct`, etc.                                                                       |                        | ✅︎                          |  ✅︎                   |
| `Phi4MMForCausalLM`                          | Phi-4-multimodal                                                         | T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup> | `microsoft/Phi-4-multimodal-instruct`, etc.                                                                                                             | ✅︎                     | ✅︎                          | ✅︎                    |
| `PixtralForConditionalGeneration`            | Pixtral                                                                  | T + I<sup>+</sup>                                                     | `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistral-community/pixtral-12b`, etc.                                                                  |                        | ✅︎                          | ✅︎                    |
| `QwenVLForConditionalGeneration`<sup>^</sup> | Qwen-VL                                                                  | T + I<sup>E+</sup>                                                    | `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat`, etc.                                                                                                               | ✅︎                     | ✅︎                          | ✅︎                    |
| `Qwen2AudioForConditionalGeneration`         | Qwen2-Audio                                                              | T + A<sup>+</sup>                                                     | `Qwen/Qwen2-Audio-7B-Instruct`                                                                                                                          |                        | ✅︎                          | ✅︎                    |
| `Qwen2VLForConditionalGeneration`            | QVQ, Qwen2-VL                                                            | T + I<sup>E+</sup> + V<sup>E+</sup>                                   | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc.                                                                 | ✅︎                     | ✅︎                          | ✅︎                    |
| `Qwen2_5_VLForConditionalGeneration`         | Qwen2.5-VL                                                               | T + I<sup>E+</sup> + V<sup>E+</sup>                                   | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc.                                                                                     | ✅︎                     | ✅︎                          | ✅︎                    |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni                                                             | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup>                   | `Qwen/Qwen2.5-Omni-7B`                                                                                                                                  |                        | ✅︎                          | ✅︎\*                  |
| `SkyworkR1VChatModel`                        | Skywork-R1V-38B                                                          | T + I                                                                 | `Skywork/Skywork-R1V-38B`                                                                                                                               |                        | ✅︎                          | ✅︎                    |
| `SmolVLMForConditionalGeneration`            | SmolVLM2                                                                 | T + I                                                                 | `SmolVLM2-2.2B-Instruct`                                                                                                                                | ✅︎                     |                             | ✅︎                    |
| `TarsierForConditionalGeneration`            | Tarsier                                                                  | T + I<sup>E+</sup>                                                    | `omni-search/Tarsier-7b`,`omni-search/Tarsier-34b`                                                                                                      |                        | ✅︎                          | ✅︎                    |
| `Tarsier2ForConditionalGeneration`<sup>^</sup>            | Tarsier2                                                                  | T + I<sup>E+</sup> + V<sup>E+</sup>                                                    | `omni-research/Tarsier2-Recap-7b`,`omni-research/Tarsier2-7b-0115`                                                                                                      |                        | ✅︎                          | ✅︎                    |

<sup>^</sup> You need to set the architecture name via `--hf-overrides` to match the one in vLLM.  
&nbsp;&nbsp;&nbsp;&nbsp;• For example, to use DeepSeek-VL2 series models:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`--hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'`  
<sup>E</sup> Pre-computed embeddings can be inputted for this modality.  
<sup>+</sup> Multiple items can be inputted per text prompt for this modality.

!!! warning
    Both V0 and V1 support `Gemma3ForConditionalGeneration` for text-only inputs.
    However, there are differences in how they handle text + image inputs:

    V0 correctly implements the model's attention pattern:
    - Uses bidirectional attention between the image tokens corresponding to the same image
    - Uses causal attention for other tokens
    - Implemented via (naive) PyTorch SDPA with masking tensors
    - Note: May use significant memory for long prompts with image

    V1 currently uses a simplified attention pattern:
    - Uses causal attention for all tokens, including image tokens
    - Generates reasonable outputs but does not match the original model's attention for text + image inputs, especially when `{"do_pan_and_scan": true}`
    - Will be updated in the future to support the correct behavior

    This limitation exists because the model's mixed attention pattern (bidirectional for images, causal otherwise) is not yet supported by vLLM's attention backends.

!!! note
    Only `InternVLChatModel` with Qwen2.5 text backbone (`OpenGVLab/InternVL3-2B`, `OpenGVLab/InternVL2.5-1B` etc) has video inputs support currently.

!!! note
    `h2oai/h2ovl-mississippi-2b` will be available in V1 once we support head size 80.

!!! note
    To use `TIGER-Lab/Mantis-8B-siglip-llama3`, you have to pass `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'` when running vLLM.

!!! warning
    The output quality of `AllenAI/Molmo-7B-D-0924` (especially in object localization tasks) has deteriorated in recent updates.

    For the best results, we recommend using the following dependency versions (tested on A10 and L40):

    ```text
    # Core vLLM-compatible dependencies with Molmo accuracy setup (tested on L40)
    torch==2.5.1
    torchvision==0.20.1
    transformers==4.48.1
    tokenizers==0.21.0
    tiktoken==0.7.0
    vllm==0.7.0

    # Optional but recommended for improved performance and stability
    triton==3.1.0
    xformers==0.0.28.post3
    uvloop==0.21.0
    protobuf==5.29.3
    openai==1.60.2
    opencv-python-headless==4.11.0.86
    pillow==10.4.0

    # Installed FlashAttention (for float16 only)
    flash-attn>=2.5.6  # Not used in float32, but should be documented
    ```

    **Note:** Make sure you understand the security implications of using outdated packages.

!!! note
    The official `openbmb/MiniCPM-V-2` doesn't work yet, so we need to use a fork (`HwwwH/MiniCPM-V-2`) for now.
    For more details, please see: <gh-pr:4087#issuecomment-2250397630>

!!! warning
    Our PaliGemma implementations have the same problem as Gemma 3 (see above) for both V0 and V1.

!!! note
    To use Qwen2.5-Omni, you have to install Hugging Face Transformers library from source via
    `pip install git+https://github.com/huggingface/transformers.git`.

    Read audio from video pre-processing is currently supported on V0 (but not V1), because overlapping modalities is not yet supported in V1.
    `--mm-processor-kwargs '{"use_audio_in_video": true}'`.

#### Transcription

Specified using `--task transcription`.

Speech2Text models trained specifically for Automatic Speech Recognition.

| Architecture                                 | Models           | Example HF Models                                                | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|----------------------------------------------|------------------|------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|
| `WhisperForConditionalGeneration`            | Whisper          | `openai/whisper-small`, `openai/whisper-large-v3-turbo`, etc.    |                        |                             |                       |

### Pooling Models

See [this page](./pooling_models.md) for more information on how to use pooling models.

!!! important
    Since some model architectures support both generative and pooling tasks,
    you should explicitly specify the task type to ensure that the model is used in pooling mode instead of generative mode.

#### Text Embedding

Specified using `--task embed`.

Any text generation model can be converted into an embedding model by passing `--task embed`.

!!! note
    To get the best results, you should use pooling models that are specifically trained as such.

The following table lists those that are tested in vLLM.

| Architecture                        | Models             | Inputs   | Example HF Models        | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
|-------------------------------------|--------------------|----------|--------------------------|------------------------|-----------------------------|-----------------------|
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT-based   | T / I    | `royokong/e5-v`          |                        |                             |                       |
| `Phi3VForCausalLM`                  | Phi-3-Vision-based | T + I    | `TIGER-Lab/VLM2Vec-Full` | 🚧                     | ✅︎                          |                       |

---

## Model Support Policy

At vLLM, we are committed to facilitating the integration and support of third-party models within our ecosystem. Our approach is designed to balance the need for robustness and the practical limitations of supporting a wide range of models. Here’s how we manage third-party model support:

1. **Community-Driven Support**: We encourage community contributions for adding new models. When a user requests support for a new model, we welcome pull requests (PRs) from the community. These contributions are evaluated primarily on the sensibility of the output they generate, rather than strict consistency with existing implementations such as those in transformers. **Call for contribution:** PRs coming directly from model vendors are greatly appreciated!

2. **Best-Effort Consistency**: While we aim to maintain a level of consistency between the models implemented in vLLM and other frameworks like transformers, complete alignment is not always feasible. Factors like acceleration techniques and the use of low-precision computations can introduce discrepancies. Our commitment is to ensure that the implemented models are functional and produce sensible results.

    !!! tip
        When comparing the output of `model.generate` from Hugging Face Transformers with the output of `llm.generate` from vLLM, note that the former reads the model's generation config file (i.e., [generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)) and applies the default parameters for generation, while the latter only uses the parameters passed to the function. Ensure all sampling parameters are identical when comparing outputs.

3. **Issue Resolution and Model Updates**: Users are encouraged to report any bugs or issues they encounter with third-party models. Proposed fixes should be submitted via PRs, with a clear explanation of the problem and the rationale behind the proposed solution. If a fix for one model impacts another, we rely on the community to highlight and address these cross-model dependencies. Note: for bugfix PRs, it is good etiquette to inform the original author to seek their feedback.

4. **Monitoring and Updates**: Users interested in specific models should monitor the commit history for those models (e.g., by tracking changes in the main/vllm/model_executor/models directory). This proactive approach helps users stay informed about updates and changes that may affect the models they use.

5. **Selective Focus**: Our resources are primarily directed towards models with significant user interest and impact. Models that are less frequently used may receive less attention, and we rely on the community to play a more active role in their upkeep and improvement.

Through this approach, vLLM fosters a collaborative environment where both the core development team and the broader community contribute to the robustness and diversity of the third-party models supported in our ecosystem.

Note that, as an inference engine, vLLM does not introduce new models. Therefore, all models supported by vLLM are third-party models in this regard.

We have the following levels of testing for models:

1. **Strict Consistency**: We compare the output of the model with the output of the model in the HuggingFace Transformers library under greedy decoding. This is the most stringent test. Please refer to [models tests](https://github.com/vllm-project/vllm/blob/main/tests/models) for the models that have passed this test.
2. **Output Sensibility**: We check if the output of the model is sensible and coherent, by measuring the perplexity of the output and checking for any obvious errors. This is a less stringent test.
3. **Runtime Functionality**: We check if the model can be loaded and run without errors. This is the least stringent test. Please refer to [functionality tests](gh-dir:tests) and [examples](gh-dir:examples) for the models that have passed this test.
4. **Community Feedback**: We rely on the community to provide feedback on the models. If a model is broken or not working as expected, we encourage users to raise issues to report it or open pull requests to fix it. The rest of the models fall under this category.
