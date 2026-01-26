# Supported Models

vLLM supports [generative](./generative_models.md) and [pooling](./pooling_models.md) models across various tasks.

For each task, we list the model architectures that have been implemented in vLLM.
Alongside each architecture, we include some popular models that use it.

## Model Implementation

### vLLM

If vLLM natively supports a model, its implementation can be found in [vllm/model_executor/models](../../vllm/model_executor/models).

These models are what we list in [supported text models](#list-of-text-only-language-models) and [supported multimodal models](#list-of-multimodal-language-models).

### Transformers

vLLM also supports model implementations that are available in Transformers. You should expect the performance of a Transformers model implementation used in vLLM to be within <5% of the performance of a dedicated vLLM model implementation. We call this feature the "Transformers modeling backend".

Currently, the Transformers modeling backend works for the following:

- Modalities: embedding models, language models and vision-language models*
- Architectures: encoder-only, decoder-only, mixture-of-experts
- Attention types: full attention and/or sliding attention

_*Vision-language models currently accept only image inputs. Support for video inputs will be added in a future release._

If the Transformers model implementation follows all the steps in [writing a custom model](#writing-custom-models) then, when used with the Transformers modeling backend, it will be compatible with the following features of vLLM:

- All the features listed in the [compatibility matrix](../features/README.md#feature-x-feature)
- Any combination of the following vLLM parallelisation schemes:
    - Data parallel
    - Tensor parallel
    - Expert parallel
    - Pipeline parallel

Checking if the modeling backend is Transformers is as simple as:

```python
from vllm import LLM
llm = LLM(model=...)  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))
```

If the printed type starts with `Transformers...` then it's using the Transformers model implementation!

If a model has a vLLM implementation but you would prefer to use the Transformers implementation via the Transformers modeling backend, set `model_impl="transformers"` for [offline inference](../serving/offline_inference.md) or `--model-impl transformers` for the [online serving](../serving/openai_compatible_server.md).

!!! note
    For vision-language models, if you are loading with `dtype="auto"`, vLLM loads the whole model with config's `dtype` if it exists. In contrast the native Transformers will respect the `dtype` attribute of each backbone in the model. That might cause a slight difference in performance.

#### Custom models

If a model is neither supported natively by vLLM nor Transformers, it can still be used in vLLM!

For a model to be compatible with the Transformers modeling backend for vLLM it must:

- be a Transformers compatible custom model (see [Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models)):
    - The model directory must have the correct structure (e.g. `config.json` is present).
    - `config.json` must contain `auto_map.AutoModel`.
- be a Transformers modeling backend for vLLM compatible model (see [Writing custom models](#writing-custom-models)):
    - Customisation should be done in the base model (e.g. in `MyModel`, not `MyModelForCausalLM`).

If the compatible model is:

- on the Hugging Face Model Hub, simply set `trust_remote_code=True` for [offline-inference](../serving/offline_inference.md) or `--trust-remote-code` for the [openai-compatible-server](../serving/openai_compatible_server.md).
- in a local directory, simply pass directory path to `model=<MODEL_DIR>` for [offline-inference](../serving/offline_inference.md) or `vllm serve <MODEL_DIR>` for the [openai-compatible-server](../serving/openai_compatible_server.md).

This means that, with the Transformers modeling backend for vLLM, new models can be used before they are officially supported in Transformers or vLLM!

#### Writing custom models

This section details the necessary modifications to make to a Transformers compatible custom model that make it compatible with the Transformers modeling backend for vLLM. (We assume that a Transformers compatible custom model has already been created, see [Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models)).

To make your model compatible with the Transformers modeling backend, it needs:

1. `kwargs` passed down through all modules from `MyModel` to `MyAttention`.
    - If your model is encoder-only:
        1. Add `is_causal = False` to `MyAttention`.
    - If your model is mixture-of-experts (MoE):
        1. Your sparse MoE block must have an attribute called `experts`.
        2. The class of `experts` (`MyExperts`) must either:
            - Inherit from `nn.ModuleList` (naive).
            - Or contain all 3D `nn.Parameters` (packed).
        3. `MyExperts.forward` must accept `hidden_states`, `top_k_index`, `top_k_weights`.
2. `MyAttention` must use `ALL_ATTENTION_FUNCTIONS` to call attention.
3. `MyModel` must contain `_supports_attention_backend = True`.

<details class="code">
<summary>modeling_my_model.py</summary>

```python

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):
    is_causal = False  # Only do this for encoder-only models

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

# Only do this for mixture-of-experts models
class MyExperts(nn.ModuleList):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ...

# Only do this for mixture-of-experts models
class MySparseMoEBlock(nn.Module):
    def __init__(self, config):
        ...
        self.experts = MyExperts(config)
        ...

    def forward(self, hidden_states: torch.Tensor):
        ...
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

</details>

Here is what happens in the background when this model is loaded:

1. The config is loaded.
2. `MyModel` Python class is loaded from the `auto_map` in config, and we check that the model `is_backend_compatible()`.
3. `MyModel` is loaded into one of the Transformers modeling backend classes in [vllm/model_executor/models/transformers](../../vllm/model_executor/models/transformers) which sets `self.config._attn_implementation = "vllm"` so that vLLM's attention layer is used.

That's it!

For your model to be compatible with vLLM's tensor parallel and/or pipeline parallel features, you must add `base_model_tp_plan` and/or `base_model_pp_plan` to your model's config class:

<details class="code">
<summary>configuration_my_model.py</summary>

```python

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

</details>

- `base_model_tp_plan` is a `dict` that maps fully qualified layer name patterns to tensor parallel styles (currently only `"colwise"` and `"rowwise"` are supported).
- `base_model_pp_plan` is a `dict` that maps direct child layer names to `tuple`s of `list`s of `str`s:
    - You only need to do this for layers which are not present on all pipeline stages
    - vLLM assumes that there will be only one `nn.ModuleList`, which is distributed across the pipeline stages
    - The `list` in the first element of the `tuple` contains the names of the input arguments
    - The `list` in the last element of the `tuple` contains the names of the variables the layer outputs to in your modeling code

## Loading a Model

### Hugging Face Hub

By default, vLLM loads models from [Hugging Face (HF) Hub](https://huggingface.co/models). To change the download path for models, you can set the `HF_HOME` environment variable; for more details, refer to [their official documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome).

To determine whether a given model is natively supported, you can check the `config.json` file inside the HF repository.
If the `"architectures"` field contains a model architecture listed below, then it should be natively supported.

Models do not _need_ to be natively supported to be used in vLLM.
The [Transformers modeling backend](#transformers) enables you to run models directly using their Transformers implementation (or even remote code on the Hugging Face Model Hub!).

!!! tip
    The easiest way to check if your model is really supported at runtime is to run the program below:

    ```python
    from vllm import LLM

    # For generative models (runner=generate) only
    llm = LLM(model=..., runner="generate")  # Name or path of your model
    output = llm.generate("Hello, my name is")
    print(output)

    # For pooling models (runner=pooling) only
    llm = LLM(model=..., runner="pooling")  # Name or path of your model
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    If vLLM successfully returns text (for generative models) or hidden states (for pooling models), it indicates that your model is supported.

Otherwise, please refer to [Adding a New Model](../contributing/model/README.md) for instructions on how to implement your model in vLLM.
Alternatively, you can [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) to request vLLM support.

#### Download a model

If you prefer, you can use the Hugging Face CLI to [download a model](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download) or specific files from a model repository:

```bash
# Download a model
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

# Specify a custom cache directory
huggingface-cli download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# Download a specific file from a model repo
huggingface-cli download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### List the downloaded models

Use the Hugging Face CLI to [manage models](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#scan-your-cache) stored in local cache:

```bash
# List cached models
huggingface-cli scan-cache

# Show detailed (verbose) output
huggingface-cli scan-cache -v

# Specify a custom cache directory
huggingface-cli scan-cache --dir ~/.cache/huggingface/hub
```

#### Delete a cached model

Use the Hugging Face CLI to interactively [delete downloaded model](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#clean-your-cache) from the cache:

<details>
<summary>Commands</summary>

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

</details>

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
https_proxy=http://your.proxy.server:port  vllm serve <model_name>
```

- Set the proxy in Python interpreter:

```python
import os

os.environ["http_proxy"] = "http://your.proxy.server:port"
os.environ["https_proxy"] = "http://your.proxy.server:port"
```

### ModelScope

To use models from [ModelScope](https://www.modelscope.cn) instead of Hugging Face Hub, set an environment variable:

```shell
export VLLM_USE_MODELSCOPE=True
```

And use with `trust_remote_code=True`.

```python
from vllm import LLM

llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)

# For generative models (runner=generate) only
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (runner=pooling) only
output = llm.encode("Hello, my name is")
print(output)
```

## Feature Status Legend

- ✅︎ indicates that the feature is supported for the model.

- 🚧 indicates that the feature is planned but not yet supported for the model.

- ⚠️ indicates that the feature is available but may have known issues or limitations.

## List of Text-only Language Models

### Generative Models

See [this page](generative_models.md) for more information on how to use generative models.

#### Text Generation

These models primarily accept the [`LLM.generate`](./generative_models.md#llmgenerate) API. Chat/Instruct models additionally support the [`LLM.chat`](./generative_models.md#llmchat) API.

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `AfmoeForCausalLM` | Afmoe | TBA | ✅︎ | ✅︎ |
| `ApertusForCausalLM` | Apertus | `swiss-ai/Apertus-8B-2509`, `swiss-ai/Apertus-70B-Instruct-2509`, etc. | ✅︎ | ✅︎ |
| `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc. | ✅︎ | ✅︎ |
| `ArceeForCausalLM` | Arcee (AFM) | `arcee-ai/AFM-4.5B-Base`, etc. | ✅︎ | ✅︎ |
| `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc. | | ✅︎ |
| `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ |
| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | ✅︎ | ✅︎ |
| `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0`, etc. | ✅︎ | ✅︎ |
| `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ |
| `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `thu-coai/ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ |
| `CohereForCausalLM`, `Cohere2ForCausalLM` | Command-R, Command-A | `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r7b-12-2024`, `CohereLabs/c4ai-command-a-03-2025`, `CohereLabs/command-a-reasoning-08-2025`, etc. | ✅︎ | ✅︎ |
| `DbrxForCausalLM` | DBRX | `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc. | | ✅︎ |
| `DeciLMForCausalLM` | DeciLM | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc. | ✅︎ | ✅︎ |
| `DeepseekForCausalLM` | DeepSeek | `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`, etc. | ✅︎ | ✅︎ |
| `Dots1ForCausalLM` | dots.llm1 | `rednote-hilab/dots.llm1.base`, `rednote-hilab/dots.llm1.inst`, etc. | | ✅︎ |
| `DotsOCRForCausalLM` | dots_ocr | `rednote-hilab/dots.ocr` | ✅︎ | ✅︎ |
| `Ernie4_5ForCausalLM` | Ernie4.5 | `baidu/ERNIE-4.5-0.3B-PT`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_MoeForCausalLM` | Ernie4.5MoE | `baidu/ERNIE-4.5-21B-A3B-PT`, `baidu/ERNIE-4.5-300B-A47B-PT`, etc. |✅︎| ✅︎ |
| `ExaoneForCausalLM` | EXAONE-3 | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc. | ✅︎ | ✅︎ |
| `ExaoneMoeCausalLM` | K-EXAONE | `LGAI-EXAONE/K-EXAONE-236B-A23B`, etc. | | |
| `Exaone4ForCausalLM` | EXAONE-4 | `LGAI-EXAONE/EXAONE-4.0-32B`, etc. | ✅︎ | ✅︎ |
| `Fairseq2LlamaForCausalLM` | Llama (fairseq2 format) | `mgleize/fairseq2-dummy-Llama-3.2-1B`, etc. | ✅︎ | ✅︎ |
| `FalconForCausalLM` | Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc. | | ✅︎ |
| `FalconMambaForCausalLM` | FalconMamba | `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, etc. | | ✅︎ |
| `FalconH1ForCausalLM` | Falcon-H1 | `tiiuae/Falcon-H1-34B-Base`, `tiiuae/Falcon-H1-34B-Instruct`, etc. | ✅︎ | ✅︎ |
| `FlexOlmoForCausalLM` | FlexOlmo | `allenai/FlexOlmo-7x7B-1T`, `allenai/FlexOlmo-7x7B-1T-RT`, etc. | | ✅︎ |
| `GemmaForCausalLM` | Gemma | `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma2ForCausalLM` | Gemma 2 | `google/gemma-2-9b`, `google/gemma-2-27b`, etc. | ✅︎ | ✅︎ |
| `Gemma3ForCausalLM` | Gemma 3 | `google/gemma-3-1b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma3nForCausalLM` | Gemma 3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ |
| `Glm4ForCausalLM` | GLM-4-0414 | `zai-org/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ |
| `Glm4MoeForCausalLM` | GLM-4.5, GLM-4.6, GLM-4.7 | `zai-org/GLM-4.5`, etc. | ✅︎ | ✅︎ |
| `GPT2LMHeadModel` | GPT-2 | `openai-community/gpt2`, `openai-community/gpt2-xl`, etc. | | ✅︎ |
| `GPTBigCodeForCausalLM` | StarCoder, SantaCoder, WizardCoder | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc. | ✅︎ | ✅︎ |
| `GPTJForCausalLM` | GPT-J | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc. | | ✅︎ |
| `GPTNeoXForCausalLM` | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc. | | ✅︎ |
| `GptOssForCausalLM` | GPT-OSS | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | ✅︎ | ✅︎ |
| `GraniteForCausalLM` | Granite 3.0, Granite 3.1, PowerLM | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeForCausalLM` | Granite 3.0 MoE, PowerMoE | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeHybridForCausalLM` | Granite 4.0 MoE Hybrid | `ibm-granite/granite-4.0-tiny-preview`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeSharedForCausalLM` | Granite MoE Shared | `ibm-research/moe-7b-1b-active-shared-experts` (test model) | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `Grok1ModelForCausalLM` | Grok1 | `hpcai-tech/grok-1`. | ✅︎ | ✅︎ |
| `Grok1ForCausalLM` | Grok2 | `xai-org/grok-2` | ✅︎ | ✅︎ |
| `HunYuanDenseV1ForCausalLM` | Hunyuan Dense | `tencent/Hunyuan-7B-Instruct` | ✅︎ | ✅︎ |
| `HunYuanMoEV1ForCausalLM` | Hunyuan-A13B | `tencent/Hunyuan-A13B-Instruct`, `tencent/Hunyuan-A13B-Pretrain`, `tencent/Hunyuan-A13B-Instruct-FP8`, etc. | ✅︎ | ✅︎ |
| `HCXVisionForCausalLM` | HyperCLOVAX-SEED-Vision-Instruct-3B | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | | |
| `InternLMForCausalLM` | InternLM | `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc. | ✅︎ | ✅︎ |
| `InternLM2ForCausalLM` | InternLM2 | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc. | ✅︎ | ✅︎ |
| `InternLM3ForCausalLM` | InternLM3 | `internlm/internlm3-8b-instruct`, etc. | ✅︎ | ✅︎ |
| `IQuestCoderForCausalLM` | IQuestCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Instruct`, etc. | | |
| `IQuestLoopCoderForCausalLM` | IQuestLoopCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct`, etc. | | |
| `JAISLMHeadModel` | Jais | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3`, etc. | | ✅︎ |
| `Jais2ForCausalLM` | Jais2 | `inceptionai/Jais-2-8B-Chat`, `inceptionai/Jais-2-70B-Chat`, etc. | | ✅︎ |
| `JambaForCausalLM` | Jamba | `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1`, etc. | ✅︎ | ✅︎ |
| `KimiLinearForCausalLM` | Kimi-Linear-48B-A3B-Base, Kimi-Linear-48B-A3B-Instruct | `moonshotai/Kimi-Linear-48B-A3B-Base`, `moonshotai/Kimi-Linear-48B-A3B-Instruct` | | ✅︎ |
| `Lfm2ForCausalLM`  | LFM2  | `LiquidAI/LFM2-1.2B`, `LiquidAI/LFM2-700M`, `LiquidAI/LFM2-350M`, etc. | ✅︎ | ✅︎ |
| `Lfm2MoeForCausalLM`  | LFM2MoE  | `LiquidAI/LFM2-8B-A1B-preview`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc. | ✅︎ | ✅︎ |
| `MambaForCausalLM` | Mamba | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf`, etc. | | ✅︎ |
| `Mamba2ForCausalLM` | Mamba2 | `mistralai/Mamba-Codestral-7B-v0.1`, etc. | | ✅︎ |
| `MiMoForCausalLM` | MiMo | `XiaomiMiMo/MiMo-7B-RL`, etc. | ✅︎ | ✅︎ |
| `MiMoV2FlashForCausalLM` | MiMoV2Flash | `XiaomiMiMo/MiMo-V2-Flash`, etc. | ︎| ✅︎ |
| `MiniCPMForCausalLM` | MiniCPM | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft`, etc. | ✅︎ | ✅︎ |
| `MiniCPM3ForCausalLM` | MiniCPM3 | `openbmb/MiniCPM3-4B`, etc. | ✅︎ | ✅︎ |
| `MiniMaxM2ForCausalLM` | MiniMax-M2, MiniMax-M2.1 |`MiniMaxAI/MiniMax-M2`, etc. | ✅︎ | ✅︎ |
| `MistralForCausalLM` | Ministral-3, Mistral, Mistral-Instruct | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc. | ✅︎ | ✅︎ |
| `MistralLarge3ForCausalLM` | Mistral-Large-3-675B-Base-2512, Mistral-Large-3-675B-Instruct-2512 | `mistralai/Mistral-Large-3-675B-Base-2512`, `mistralai/Mistral-Large-3-675B-Instruct-2512`, etc. | ✅︎ | ✅︎ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc. | ✅︎ | ✅︎ |
| `MPTForCausalLM` | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc. | | ✅︎ |
| `NemotronForCausalLM` | Nemotron-3, Nemotron-4, Minitron | `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8`, etc. | ✅︎ | ✅︎ |
| `NemotronHForCausalLM` | Nemotron-H | `nvidia/Nemotron-H-8B-Base-8K`, `nvidia/Nemotron-H-47B-Base-8K`, `nvidia/Nemotron-H-56B-Base-8K`, etc. | ✅︎ | ✅︎ |
| `OLMoForCausalLM` | OLMo | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc. | ✅︎ | ✅︎ |
| `OLMo2ForCausalLM` | OLMo2 | `allenai/OLMo-2-0425-1B`, etc. | ✅︎ | ✅︎ |
| `OLMo3ForCausalLM` | OLMo3 | `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-32B-Think`, etc. | ✅︎ | ✅︎ |
| `OLMoEForCausalLM` | OLMoE | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct`, etc. | | ✅︎ |
| `OPTForCausalLM` | OPT, OPT-IML | `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc. | ✅︎ | ✅︎ |
| `OrionForCausalLM` | Orion | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc. | | ✅︎ |
| `OuroForCausalLM` | ouro | `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, etc. | ✅︎ | |
| `PanguEmbeddedForCausalLM` |openPangu-Embedded-7B | `FreedomIntelligence/openPangu-Embedded-7B-V1.1` | ✅︎ | ✅︎ |
| `PanguProMoEV2ForCausalLM` |openpangu-pro-moe-v2 | | ✅︎ | ✅︎ |
| `PanguUltraMoEForCausalLM` |openpangu-ultra-moe-718b-model | `FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1` | ✅︎ | ✅︎ |
| `PhiForCausalLM` | Phi | `microsoft/phi-1_5`, `microsoft/phi-2`, etc. | ✅︎ | ✅︎ |
| `Phi3ForCausalLM` | Phi-4, Phi-3 | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct`, etc. | ✅︎ | ✅︎ |
| `PhiMoEForCausalLM` | Phi-3.5-MoE | `microsoft/Phi-3.5-MoE-instruct`, etc. | ✅︎ | ✅︎ |
| `PersimmonForCausalLM` | Persimmon | `adept/persimmon-8b-base`, `adept/persimmon-8b-chat`, etc. | | ✅︎ |
| `Plamo2ForCausalLM` | PLaMo2 | `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b`, etc. | ✅ | ✅︎ |
| `Plamo3ForCausalLM` | PLaMo3 | `pfnet/plamo-3-nict-2b-base`, `pfnet/plamo-3-nict-8b-base`, etc. | ✅ | ✅︎ |
| `QWenLMHeadModel` | Qwen | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc. | ✅︎ | ✅︎ |
| `Qwen2MoeForCausalLM` | Qwen2MoE | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B`, etc. | ✅︎ | ✅︎ |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B`, etc. | ✅︎ | ✅︎ |
| `Qwen3NextForCausalLM` | Qwen3NextMoE | `Qwen/Qwen3-Next-80B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `SeedOssForCausalLM` | SeedOss | `ByteDance-Seed/Seed-OSS-36B-Instruct`, etc. | ✅︎ | ✅︎ |
| `SolarForCausalLM` | Solar Pro | `upstage/solar-pro-preview-instruct`, etc. | ✅︎ | ✅︎ |
| `StableLmForCausalLM` | StableLM | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc. | | |
| `Starcoder2ForCausalLM` | Starcoder2 | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc. | | ✅︎ |
| `Step1ForCausalLM` | Step-Audio | `stepfun-ai/Step-Audio-EditX`, etc. | ✅︎ | ✅︎ |
| `TeleChat2ForCausalLM` | TeleChat2 | `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B`, etc. | ✅︎ | ✅︎ |
| `TeleFLMForCausalLM` | TeleFLM | `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM`, etc. | ✅︎ | ✅︎ |
| `XverseForCausalLM` | XVERSE | `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc. | ✅︎ | ✅︎ |
| `MiniMaxM1ForCausalLM` | MiniMax-Text | `MiniMaxAI/MiniMax-M1-40k`, `MiniMaxAI/MiniMax-M1-80k`, etc. | | |
| `MiniMaxText01ForCausalLM` | MiniMax-Text | `MiniMaxAI/MiniMax-Text-01`, etc. | | |
| `Zamba2ForCausalLM` | Zamba2 | `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct`, etc. | | |
| `LongcatFlashForCausalLM` | LongCat-Flash | `meituan-longcat/LongCat-Flash-Chat`, `meituan-longcat/LongCat-Flash-Chat-FP8` | ✅︎ | ✅︎ |

!!! note
    Grok2 requires `tokenizer.tok.json` with `tiktoken` installed. You can optionally override MoE router renormalization with `moe_router_renormalize`.

Some models are supported only via the [Transformers modeling backend](#transformers). The purpose of the table below is to acknowledge models which we officially support in this way. The logs will say that the Transformers modeling backend is being used, and you will see no warning that this is fallback behaviour. This means that, if you have issues with any of the models listed below, please [make an issue](https://github.com/vllm-project/vllm/issues/new/choose) and we'll do our best to fix it!

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `SmolLM3ForCausalLM` | SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | ✅︎ | ✅︎ |

!!! note
    Currently, the ROCm version of vLLM supports Mistral and Mixtral only for context lengths up to 4096.

### Pooling Models

See [this page](./pooling_models.md) for more information on how to use pooling models.

!!! important
    Since some model architectures support both generative and pooling tasks,
    you should explicitly specify `--runner pooling` to ensure that the model is used in pooling mode instead of generative mode.

#### Embedding

These models primarily support the [`LLM.embed`](./pooling_models.md#llmembed) API.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `BertModel`<sup>C</sup> | BERT-based | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc. | | |
| `BertSpladeSparseEmbeddingModel` | SPLADE | `naver/splade-v3` | | |
| `Gemma2Model`<sup>C</sup> | Gemma 2-based | `BAAI/bge-multilingual-gemma2`, etc. | ✅︎ | ✅︎ |
| `Gemma3TextModel`<sup>C</sup> | Gemma 3-based | `google/embeddinggemma-300m`, etc. | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `GteModel`<sup>C</sup> | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0`. |  |  |
| `GteNewModel`<sup>C</sup> | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc. |  |  |
| `ModernBertModel`<sup>C</sup> | ModernBERT-based | `Alibaba-NLP/gte-modernbert-base`, etc. |  |  |
| `NomicBertModel`<sup>C</sup> | Nomic BERT | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc. |  |  |
| `LlamaBidirectionalModel`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-embed-1b-v2`, etc. | ✅︎ | ✅︎ |
| `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup>, etc. | Llama-based | `intfloat/e5-mistral-7b-instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2Model`<sup>C</sup>, `Qwen2ForCausalLM`<sup>C</sup> | Qwen2-based | `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc. | ✅︎ | ✅︎ |
| `Qwen3Model`<sup>C</sup>, `Qwen3ForCausalLM`<sup>C</sup> | Qwen3-based | `Qwen/Qwen3-Embedding-0.6B`, etc. | ✅︎ | ✅︎ |
| `RobertaModel`, `RobertaForMaskedLM` | RoBERTa-based | `sentence-transformers/all-roberta-large-v1`, etc. | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` has an improperly defined Sentence Transformers config.
    You need to manually set mean pooling by passing `--pooler-config '{"pooling_type": "MEAN"}'`.

!!! note
    For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
    See [relevant issue on HF Transformers](https://github.com/huggingface/transformers/issues/34882).

!!! note
    `jinaai/jina-embeddings-v3` supports multiple tasks through LoRA, while vllm temporarily only supports text-matching tasks by merging LoRA weights.

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewModel`. The name `NewModel` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewModel"]}'` to specify the use of the `GteNewModel` architecture.

If your model is not in the above list, we will try to automatically convert the model using
[as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model]. By default, the embeddings
of the whole prompt are extracted from the normalized hidden state corresponding to the last token.

#### Classification

These models primarily support the [`LLM.classify`](./pooling_models.md#llmclassify) API.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `JambaForSequenceClassification` | Jamba | `ai21labs/Jamba-tiny-reward-dev`, etc. | ✅︎ | ✅︎ |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

#### Cross-encoder / Reranker

Cross-encoder and reranker models are a subset of classification models that accept two prompts as input.
These models primarily support the [`LLM.score`](./pooling_models.md#llmscore) API.

| Architecture | Models | Example HF Models | Score template (see note) | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|---------------------------|-----------------------------|-----------------------------------------|
| `BertForSequenceClassification` | BERT-based | `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc. | N/A | | |
| `GemmaForSequenceClassification` | Gemma-based | `BAAI/bge-reranker-v2-gemma`(see note), etc. | [bge-reranker-v2-gemma.jinja](../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-reranker-base`, etc. | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-rerank-1b-v2`, etc. | [nemotron-rerank.jinja](../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `mixedbread-ai/mxbai-rerank-base-v2`(see note), etc. | [mxbai_rerank_v2.jinja](../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(see note), etc. | [qwen3_reranker.jinja](../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | RoBERTa-based | `cross-encoder/quora-roberta-base`, etc. | N/A | | |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa-based | `BAAI/bge-reranker-v2-m3`, etc. | N/A | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Some models require a specific prompt format to work correctly.

    You can find Example HF Models's corresponding score template in [examples/pooling/score/template/](../../examples/pooling/score/template)

    Examples : [examples/pooling/score/using_template_offline.py](../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../examples/pooling/score/using_template_online.py)

!!! note
    Load the official original `BAAI/bge-reranker-v2-gemma` by using the following command.

    ```bash
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}'
    ```

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewForSequenceClassification`. The name `NewForSequenceClassification` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'` to specify the use of the `GteNewForSequenceClassification` architecture.

!!! note
    Load the official original `mxbai-rerank-v2` by using the following command.

    ```bash
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}'
    ```

!!! note
    Load the official original `Qwen3 Reranker` by using the following command. More information can be found at: [examples/pooling/score/qwen3_reranker_offline.py](../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../examples/pooling/score/qwen3_reranker_online.py).

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

#### Reward Modeling

These models primarily support the [`LLM.reward`](./pooling_models.md#llmreward) API.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `InternLM2ForRewardModel` | InternLM2-based | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama-based | `peiyi9979/math-shepherd-mistral-7b-prm`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-RM-72B`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForProcessRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-PRM-7B`, etc. | ✅︎ | ✅︎ |

!!! important
    For process-supervised reward models such as `peiyi9979/math-shepherd-mistral-7b-prm`, the pooling config should be set explicitly,
    e.g.: `--pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`.

#### Token Classification

These models primarily support the [`LLM.encode`](./pooling_models.md#llmencode) API.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|-----------------------------|-----------------------------------------|
| `BertForTokenClassification` | bert-based | `boltuix/NeuroBERT-NER` (see note), etc. |  |  |
| `ModernBertForTokenClassification` | ModernBERT-based | `disham993/electrical-ner-ModernBERT-base` |  |  |

!!! note
    Named Entity Recognition (NER) usage, please refer to [examples/pooling/token_classify/ner_offline.py](../../examples/pooling/token_classify/ner_offline.py), [examples/pooling/token_classify/ner_online.py](../../examples/pooling/token_classify/ner_online.py).

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

See [this page](../features/multimodal_inputs.md) on how to pass multi-modal inputs to the model.

!!! tip
    For hybrid-only models such as Llama-4, Step3 and Mistral-3, a text-only mode can be enabled by setting all supported multimodal modalities to 0 (e.g, `--limit-mm-per-prompt '{"image":0}`) so that their multimodal modules will not be loaded to free up more GPU memory for KV cache.

!!! note
    vLLM currently supports adding LoRA adapters to the language backbone for most multimodal models. Additionally, vLLM now experimentally supports adding LoRA to the tower and connector modules for some multimodal models. See [this page](../features/lora.md).

### Generative Models

See [this page](generative_models.md) for more information on how to use generative models.

#### Text Generation

These models primarily accept the [`LLM.generate`](./generative_models.md#llmgenerate) API. Chat/Instruct models additionally support the [`LLM.chat`](./generative_models.md#llmchat) API.

| Architecture | Models | Inputs | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `AriaForConditionalGeneration` | Aria | T + I<sup>+</sup> | `rhymes-ai/Aria` | | |
| `AudioFlamingo3ForConditionalGeneration` | AudioFlamingo3 | T + A<sup>+</sup> | `nvidia/audio-flamingo-3-hf`, `nvidia/music-flamingo-hf` | ✅︎ | ✅︎ |
| `AyaVisionForConditionalGeneration` | Aya Vision | T + I<sup>+</sup> | `CohereLabs/aya-vision-8b`, `CohereLabs/aya-vision-32b`, etc. | | ✅︎ |
| `BagelForConditionalGeneration` | BAGEL | T + I<sup>+</sup> | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ |
| `BeeForConditionalGeneration` | Bee-8B | T + I<sup>E+</sup> | `Open-Bee/Bee-8B-RL`, `Open-Bee/Bee-8B-SFT` | | ✅︎ |
| `Blip2ForConditionalGeneration` | BLIP-2 | T + I<sup>E</sup> | `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`, etc. | ✅︎ | ✅︎ |
| `ChameleonForConditionalGeneration` | Chameleon | T + I | `facebook/chameleon-7b`, etc. | | ✅︎ |
| `Cohere2VisionForConditionalGeneration` | Command A Vision | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025`, etc. | | ✅︎ |
| `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | ✅︎ | ✅︎ |
| `Eagle2_5_VLForConditionalGeneration` | Eagle2.5-VL | T + I<sup>E+</sup> | `nvidia/Eagle2.5-8B`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
| `FuyuForCausalLM` | Fuyu | T + I | `adept/fuyu-8b`, etc. | | ✅︎ |
| `Gemma3ForConditionalGeneration` | Gemma 3 | T + I<sup>E+</sup> | `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma3nForConditionalGeneration` | Gemma 3n | T + I + A | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `GLM4VForCausalLM`<sup>^</sup> | GLM-4V | T + I | `zai-org/glm-4v-9b`, `zai-org/cogagent-9b-20241220`, etc. | ✅︎ | ✅︎ |
| `Glm4vForConditionalGeneration` | GLM-4.1V-Thinking | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.1V-9B-Thinking`, etc. | ✅︎ | ✅︎ |
| `Glm4vMoeForConditionalGeneration` | GLM-4.5V | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.5V`, etc. | ✅︎ | ✅︎ |
| `GlmOcrForConditionalGeneration` | GLM-OCR | T + I<sup>E+</sup>  | `zai-org/GLM-OCR`, etc. | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | T + A | `ibm-granite/granite-speech-3.3-8b` | ✅︎ | ✅︎ |
| `H2OVLChatModel` | H2OVL | T + I<sup>E+</sup> | `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b`, etc. | | ✅︎ |
| `HunYuanVLForConditionalGeneration` | HunyuanOCR | T + I<sup>E+</sup> | `tencent/HunyuanOCR`, etc. | ✅︎ | ✅︎ |
| `Idefics3ForConditionalGeneration` | Idefics3 | T + I | `HuggingFaceM4/Idefics3-8B-Llama3`, etc. | ✅︎ | |
| `IsaacForConditionalGeneration` | Isaac | T + I<sup>+</sup> | `PerceptronAI/Isaac-0.1` | ✅︎ | ✅︎ |
| `InternS1ForConditionalGeneration` | Intern-S1 | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1`, `internlm/Intern-S1-mini`, etc. | ✅︎ | ✅︎ |
| `InternVLChatModel` | InternVL 3.5, InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>) | `OpenGVLab/InternVL3_5-14B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc. | ✅︎ | ✅︎ |
| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ |
| `KananaVForConditionalGeneration` | Kanana-V | T + I<sup>+</sup> | `kakaocorp/kanana-1.5-v-3b-instruct`, etc. | | ✅︎ |
| `KeyeForConditionalGeneration` | Keye-VL-8B-Preview | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-8B-Preview` | ✅︎ | ✅︎ |
| `KeyeVL1_5ForConditionalGeneration` | Keye-VL-1_5-8B | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-1_5-8B` | ✅︎ | ✅︎ |
| `KimiVLForConditionalGeneration` | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking | T + I<sup>+</sup> | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking` | | ✅︎ |
| `LightOnOCRForConditionalGeneration`  | LightOnOCR-1B  | T + I<sup>+</sup> | `lightonai/LightOnOCR-1B`, etc | ✅︎ | ✅︎ |
| `Lfm2VlForConditionalGeneration` | LFM2-VL | T + I<sup>+</sup> | `LiquidAI/LFM2-VL-450M`, `LiquidAI/LFM2-VL-3B`, `LiquidAI/LFM2-VL-8B-A1B`, etc. | ✅︎ | ✅︎ |
| `Llama4ForConditionalGeneration` | Llama 4 | T + I<sup>+</sup> | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, etc. | ✅︎ | ✅︎ |
| `Llama_Nemotron_Nano_VL` | Llama Nemotron Nano VL | T + I<sup>E+</sup> | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | ✅︎ | ✅︎ |
| `LlavaForConditionalGeneration` | LLaVA-1.5, Pixtral (HF Transformers) | T + I<sup>E+</sup> | `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (see note), `mistral-community/pixtral-12b`, etc. | ✅︎ | ✅︎ |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | T + I<sup>E+</sup> | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc. | | ✅︎ |
| `LlavaNextVideoForConditionalGeneration` | LLaVA-NeXT-Video | T + V | `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc. | | ✅︎ |
| `LlavaOnevisionForConditionalGeneration` | LLaVA-Onevision | T + I<sup>+</sup> + V<sup>+</sup> | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc. | | ✅︎ |
| `MiDashengLMModel` | MiDashengLM | T + A<sup>+</sup> | `mispeech/midashenglm-7b` | | ✅︎ |
| `MiniCPMO` | MiniCPM-O | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup> | `openbmb/MiniCPM-o-2_6`, etc. | ✅︎ | ✅︎ |
| `MiniCPMV` | MiniCPM-V | T + I<sup>E+</sup> + V<sup>E+</sup> | `openbmb/MiniCPM-V-2` (see note), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, `openbmb/MiniCPM-V-4`, `openbmb/MiniCPM-V-4_5`, etc. | ✅︎ | |
| `MiniMaxVL01ForConditionalGeneration` | MiniMax-VL | T + I<sup>E+</sup> | `MiniMaxAI/MiniMax-VL-01`, etc. | | ✅︎ |
| `Mistral3ForConditionalGeneration` | Mistral3 (HF Transformers) | T + I<sup>+</sup> | `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, etc. | ✅︎ | ✅︎ |
| `MolmoForCausalLM` | Molmo | T + I<sup>+</sup> | `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`, etc. | ✅︎ | ✅︎ |
| `Molmo2ForConditionalGeneration` | Molmo2 | T + I<sup>+</sup> / V | `allenai/Molmo2-4B`, `allenai/Molmo2-8B`, `allenai/Molmo2-O-7B` | ✅︎ | ✅︎ |
| `NVLM_D_Model` | NVLM-D 1.0 | T + I<sup>+</sup> | `nvidia/NVLM-D-72B`, etc. | | ✅︎ |
| `OpenCUAForConditionalGeneration` | OpenCUA-7B | T + I<sup>E+</sup> | `xlangai/OpenCUA-7B` | ✅︎ | ✅︎ |
| `Ovis` | Ovis2, Ovis1.6 | T + I<sup>+</sup> | `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B`, etc. | | ✅︎ |
| `Ovis2_5` | Ovis2.5 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.5-9B`, etc. | | |
| `PaddleOCRVLForConditionalGeneration` | Paddle-OCR | T + I<sup>+</sup> | `PaddlePaddle/PaddleOCR-VL`, etc. | | |
| `PaliGemmaForConditionalGeneration` | PaliGemma, PaliGemma 2 | T + I<sup>E</sup> | `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448`, etc. | ✅︎ | ✅︎ |
| `Phi3VForCausalLM` | Phi-3-Vision, Phi-3.5-Vision | T + I<sup>E+</sup> | `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct`, etc. | | ✅︎ |
| `Phi4MMForCausalLM` | Phi-4-multimodal | T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup> | `microsoft/Phi-4-multimodal-instruct`, etc. | ✅︎ | ✅︎ |
| `PixtralForConditionalGeneration` | Ministral 3 (Mistral format), Mistral 3 (Mistral format), Mistral Large 3 (Mistral format), Pixtral (Mistral format) | T + I<sup>+</sup> | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistralai/Mistral-Large-3-675B-Instruct-2512` `mistralai/Pixtral-12B-2409` etc. | ✅︎ | ✅︎ |
| `QwenVLForConditionalGeneration`<sup>^</sup> | Qwen-VL | T + I<sup>E+</sup> | `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen2AudioForConditionalGeneration` | Qwen2-Audio | T + A<sup>+</sup> | `Qwen/Qwen2-Audio-7B-Instruct` | | ✅︎ |
| `Qwen2VLForConditionalGeneration` | QVQ, Qwen2-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen2.5-Omni-3B`, `Qwen/Qwen2.5-Omni-7B` | ✅︎ | ✅︎ |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-4B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Thinking` | ✅︎ | ✅︎ |
| `RForConditionalGeneration` | R-VL-4B | T + I<sup>E+</sup> | `YannQi/R-4B` | | ✅︎ |
| `SkyworkR1VChatModel` | Skywork-R1V-38B | T + I | `Skywork/Skywork-R1V-38B` | | ✅︎ |
| `SmolVLMForConditionalGeneration` | SmolVLM2 | T + I | `SmolVLM2-2.2B-Instruct` | ✅︎ | |
| `Step3VLForConditionalGeneration` | Step3-VL | T + I<sup>+</sup> | `stepfun-ai/step3` | | ✅︎ |
| `StepVLForConditionalGeneration` | Step3-VL-10B | T + I<sup>+</sup> | `stepfun-ai/Step3-VL-10B` | | ✅︎ |
| `TarsierForConditionalGeneration` | Tarsier | T + I<sup>E+</sup> | `omni-search/Tarsier-7b`, `omni-search/Tarsier-34b` | | ✅︎ |
| `Tarsier2ForConditionalGeneration`<sup>^</sup> | Tarsier2 | T + I<sup>E+</sup> + V<sup>E+</sup> | `omni-research/Tarsier2-Recap-7b`, `omni-research/Tarsier2-7b-0115` | | ✅︎ |
| `UltravoxModel` | Ultravox | T + A<sup>E+</sup> | `fixie-ai/ultravox-v0_5-llama-3_2-1b` | ✅︎ | ✅︎ |

Some models are supported only via the [Transformers modeling backend](#transformers). The purpose of the table below is to acknowledge models which we officially support in this way. The logs will say that the Transformers modeling backend is being used, and you will see no warning that this is fallback behaviour. This means that, if you have issues with any of the models listed below, please [make an issue](https://github.com/vllm-project/vllm/issues/new/choose) and we'll do our best to fix it!

| Architecture | Models | Inputs | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|--------|-------------------|-----------------------------|-----------------------------------------|
| `Emu3ForConditionalGeneration` | Emu3 | T + I | `BAAI/Emu3-Chat-hf` | ✅︎ | ✅︎ |

<sup>^</sup> You need to set the architecture name via `--hf-overrides` to match the one in vLLM.
&nbsp;&nbsp;&nbsp;&nbsp;• For example, to use DeepSeek-VL2 series models:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`--hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'`
<sup>E</sup> Pre-computed embeddings can be inputted for this modality.
<sup>+</sup> Multiple items can be inputted per text prompt for this modality.

!!! note
    `Gemma3nForConditionalGeneration` is only supported on V1 due to shared KV caching and it depends on `timm>=1.0.17` to make use of its
    MobileNet-v5 vision backbone.
  
    Performance is not yet fully optimized mainly due to:
  
    - Both audio and vision MM encoders use `transformers.AutoModel` implementation.  
    - There's no PLE caching or out-of-memory swapping support, as described in [Google's blog](https://developers.googleblog.com/en/introducing-gemma-3n/). These features might be too model-specific for vLLM, and swapping in particular may be better suited for constrained setups.

!!! note
    For `InternVLChatModel`, only InternVL2.5 with Qwen2.5 text backbone (`OpenGVLab/InternVL2.5-1B` etc.), InternVL3 and InternVL3.5 have video inputs support currently.

!!! note
    To use `TIGER-Lab/Mantis-8B-siglip-llama3`, you have to pass `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'` when running vLLM.

!!! note
    The official `openbmb/MiniCPM-V-2` doesn't work yet, so we need to use a fork (`HwwwH/MiniCPM-V-2`) for now.
    For more details, please see: <https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630>

#### Transcription

Speech2Text models trained specifically for Automatic Speech Recognition.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `Gemma3nForConditionalGeneration` | Gemma3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `GlmAsrForConditionalGeneration` | GLM-ASR | `zai-org/GLM-ASR-Nano-2512` | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | `ibm-granite/granite-speech-3.3-2b`, `ibm-granite/granite-speech-3.3-8b`, etc. | ✅︎ | ✅︎ |
| `VoxtralForConditionalGeneration` | Voxtral (Mistral format) | `mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-Small-24B-2507`, etc. | ✅︎ | ✅︎ |
| `WhisperForConditionalGeneration` | Whisper | `openai/whisper-small`, `openai/whisper-large-v3-turbo`, etc. | | |

!!! note
    `VoxtralForConditionalGeneration` requires `mistral-common[audio]` to be installed.

### Pooling Models

See [this page](./pooling_models.md) for more information on how to use pooling models.

#### Embedding

These models primarily support the [`LLM.embed`](./pooling_models.md#llmembed) API.

!!! note
    To get the best results, you should use pooling models that are specifically trained as such.

The following table lists those that are tested in vLLM.

| Architecture | Models | Inputs | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `CLIPModel` | CLIP | T / I | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, etc. | | |
| `LlavaNextForConditionalGeneration`<sup>C</sup> | LLaVA-NeXT-based | T / I | `royokong/e5-v` | | ✅︎ |
| `Phi3VForCausalLM`<sup>C</sup> | Phi-3-Vision-based | T + I | `TIGER-Lab/VLM2Vec-Full` | | ✅︎ |
| `Qwen3VLForConditionalGeneration`<sup>C</sup> | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
| `SiglipModel` | SigLIP, SigLIP2 | T / I | `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

---

#### Cross-encoder / Reranker

Cross-encoder and reranker models are a subset of classification models that accept two prompts as input.
These models primarily support the [`LLM.score`](./pooling_models.md#llmscore) API.

| Architecture | Models | Inputs | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `JinaVLForSequenceClassification` | JinaVL-based | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`(see note), etc. | ✅︎ | ✅︎ |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Similar to Qwen3-Reranker, you need to use the following `--hf_overrides` to load the official original `Qwen3-VL-Reranker`.

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

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
3. **Runtime Functionality**: We check if the model can be loaded and run without errors. This is the least stringent test. Please refer to [functionality tests](../../tests) and [examples](../../examples) for the models that have passed this test.
4. **Community Feedback**: We rely on the community to provide feedback on the models. If a model is broken or not working as expected, we encourage users to raise issues to report it or open pull requests to fix it. The rest of the models fall under this category.
