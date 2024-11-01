# OpenAI Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) API.

You can start the server using Python, or using [Docker](deploying_with_docker.rst):
```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

To call the server, you can use the official OpenAI Python client library, or any other HTTP client.
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

## API Reference

We currently support the following OpenAI APIs:

- [Completions API](https://platform.openai.com/docs/api-reference/completions)
  - *Note: `suffix` parameter is not supported.*
- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
  - [Vision](https://platform.openai.com/docs/guides/vision)-related parameters are supported; see [Using VLMs](../models/vlm.rst).
    - *Note: `image_url.detail` parameter is not supported.*
  - We also support `audio_url` content type for audio files.
    - Refer to [vllm.entrypoints.chat_utils](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/chat_utils.py) for the exact schema.
    - *TODO: Support `input_audio` content type as defined [here](https://github.com/openai/openai-python/blob/v1.52.2/src/openai/types/chat/chat_completion_content_part_input_audio_param.py).*
  - *Note: `parallel_tool_calls` and `user` parameters are ignored.*
- [Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
  - Instead of `inputs`, you can pass in a list of `messages` (same schema as Chat Completions API),
    which will be treated as a single prompt to the model according to its chat template.
    - This enables multi-modal inputs to be passed to embedding models, see [Using VLMs](../models/vlm.rst).
  - *Note: You should run `vllm serve` with `--task embedding` to ensure that the model is being run in embedding mode.*

## Extra Parameters

vLLM supports a set of parameters that are not part of the OpenAI API.
In order to use them, you can pass them as extra parameters in the OpenAI client.
Or directly merge them into the JSON payload if you are using HTTP call directly.

```python
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
  ],
  extra_body={
    "guided_choice": ["positive", "negative"]
  }
)
```

### Extra Parameters for Completions API

The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-sampling-params
:end-before: end-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-extra-params
:end-before: end-completion-extra-params
```

### Extra Parameters for Chat Completions API

The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-sampling-params
:end-before: end-chat-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-extra-params
:end-before: end-chat-completion-extra-params
```

### Extra Parameters for Embeddings API

The following [pooling parameters (click through to see documentation)](../dev/pooling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-embedding-pooling-params
:end-before: end-embedding-pooling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-embedding-extra-params
:end-before: end-embedding-extra-params
```

## Chat Template

In order for the language model to support chat protocol, vLLM requires the model to include
a chat template in its tokenizer configuration. The chat template is a Jinja2 template that
specifies how are roles, messages, and other chat-specific tokens are encoded in the input.

An example chat template for `NousResearch/Meta-Llama-3-8B-Instruct` can be found [here](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models)

Some models do not provide a chat template even though they are instruction/chat fine-tuned. For those model,
you can manually specify their chat template in the `--chat-template` parameter with the file path to the chat
template, or the template in string form. Without a chat template, the server will not be able to process chat
and all chat requests will error.

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM community provides a set of chat templates for popular models. You can find them in the examples
directory [here](https://github.com/vllm-project/vllm/tree/main/examples/)

With the inclusion of multi-modal chat APIs, the OpenAI spec now accepts chat messages in a new format which specifies 
both a `type` and a `text` field. An example is provided below:
```python
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": [{"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"}]}
  ]
)
```
Most chat templates for LLMs expect the `content` to be a `string` but there are some newer models like 
`meta-llama/Llama-Guard-3-1B` that expect the content to be parsed with the new OpenAI spec. In order to choose which
format the content needs to be parsed in by vLLM, please use the `--chat-template-text-format` argument to specify
between `string` or `openai`. The default value is `string` and vLLM internally converts both spec formats to match 
this, unless explicitly specified.


## Command line arguments for the server

```{argparse}
:module: vllm.entrypoints.openai.cli_args
:func: create_parser_for_docs
:prog: vllm serve
```
## Tool Calling in the Chat Completion API
### Named Function Calling
vLLM supports only named function calling in the chat completion API by default. It does so using Outlines, so this is 
enabled by default, and will work with any supported model. You are guaranteed a validly-parsable function call - not a 
high-quality one. 

To use a named function, you need to define the functions in the `tools` parameter of the chat completion request, and 
specify the `name` of one of the tools in the `tool_choice` parameter of the chat completion request. 

### Config file

The `serve` module can also accept arguments from a config file in
`yaml` format. The arguments in the yaml must be specified using the 
long form of the argument outlined [here](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server): 

For example:

```yaml
# config.yaml

host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

```bash
$ vllm serve SOME_MODEL --config config.yaml
```
---
**NOTE**  
In case an argument is supplied simultaneously using command line and the config file, the value from the commandline will take precedence.
The order of priorities is `command line > config file values > defaults`.

---

## Tool calling in the chat completion API
vLLM supports only named function calling in the chat completion API. The `tool_choice` options `auto` and `required` are **not yet supported** but on the roadmap.

It is the callers responsibility to prompt the model with the tool information, vLLM will not automatically manipulate the prompt.

vLLM will use guided decoding to ensure the response matches the tool parameter object defined by the JSON schema in the `tools` parameter.


### Automatic Function Calling
To enable this feature, you should set the following flags:
* `--enable-auto-tool-choice` -- **mandatory** Auto tool choice. tells vLLM that you want to enable the model to generate its own tool calls when it 
deems appropriate.
* `--tool-call-parser` -- select the tool parser to use (listed below). Additional tool parsers 
will continue to be added in the future, and also can register your own tool parsers in the `--tool-parser-plugin`.
* `--tool-parser-plugin` -- **optional** tool parser plugin used to register user defined tool parsers into vllm, the registered tool parser name can be specified in `--tool-call-parser`.
* `--chat-template` -- **optional** for auto tool choice. the path to the chat template which handles `tool`-role messages and `assistant`-role messages 
that contain previously generated tool calls. Hermes, Mistral and Llama models have tool-compatible chat templates in their 
`tokenizer_config.json` files, but you can specify a custom template. This argument can be set to `tool_use` if your model has a tool use-specific chat 
template configured in the `tokenizer_config.json`. In this case, it will be used per the `transformers` specification. More on this [here](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)
from HuggingFace; and you can find an example of this in a `tokenizer_config.json` [here](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)

If your favorite tool-calling model is not supported, please feel free to contribute a parser & tool use chat template! 


#### Hermes Models (`hermes`)

All Nous Research Hermes-series models newer than Hermes 2 Pro should be supported.
* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`


_Note that the Hermes 2 **Theta** models are known to have degraded tool call quality & capabilities due to the merge 
step in their creation_. 

Flags: `--tool-call-parser hermes`


#### Mistral Models (`mistral`)

Supported models:
* `mistralai/Mistral-7B-Instruct-v0.3` (confirmed)
* Additional mistral function-calling models are compatible as well.

Known issues:
1. Mistral 7B struggles to generate parallel tool calls correctly. 
2. Mistral's `tokenizer_config.json` chat template requires tool call IDs that are exactly 9 digits, which is 
much shorter than what vLLM generates. Since an exception is thrown when this condition 
is not met, the following additional chat templates are provided:

* `examples/tool_chat_template_mistral.jinja` - this is the "official" Mistral chat template, but tweaked so that
it works with vLLM's tool call IDs (provided `tool_call_id` fields are truncated to the last 9 digits)
* `examples/tool_chat_template_mistral_parallel.jinja` - this is a "better" version that adds a tool-use system prompt
when tools are provided, that results in much better reliability when working with parallel tool calling.


Recommended flags: `--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`


#### Llama Models (`llama3_json`)

Supported models:
* `meta-llama/Meta-Llama-3.1-8B-Instruct`
* `meta-llama/Meta-Llama-3.1-70B-Instruct`
* `meta-llama/Meta-Llama-3.1-405B-Instruct`
* `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`

The tool calling that is supported is the [JSON based tool calling](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling).
Other tool calling formats like the built in python tool calling or custom tool calling are not supported.

Known issues:
1. Parallel tool calls are not supported. 
2. The model can generate parameters with a wrong format, such as generating
   an array serialized as string instead of an array.

The `tool_chat_template_llama3_json.jinja` file contains the "official" Llama chat template, but tweaked so that
it works better with vLLM.

Recommended flags: `--tool-call-parser llama3_json --chat-template examples/tool_chat_template_llama3_json.jinja`


#### InternLM Models (`internlm`)

Supported models:
* `internlm/internlm2_5-7b-chat` (confirmed)
* Additional internlm2.5 function-calling models are compatible as well

Known issues:
* Although this implementation also supports InternLM2, the tool call results are not stable when testing with the `internlm/internlm2-chat-7b` model.

Recommended flags: `--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`


#### Jamba Models (`jamba`)
AI21's Jamba-1.5 models are supported.
* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`


Flags: `--tool-call-parser jamba`


#### IBM Granite (`granite-20b-fc`)

Supported models:
* `ibm-granite/granite-20b-functioncalling`

Flags: `--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

The example chat template deviates slightly from the original on Huggingface, which is not vLLM compatible. It blends function description elements from the Hermes template and follows the same system prompt as "Response Generation" mode from [the paper](https://arxiv.org/abs/2407.00121). Parallel function calls are supported.


### How to write a tool parser plugin

A tool parser plugin is a Python file containing one or more ToolParser implementations. You can write a ToolParser similar to the `Hermes2ProToolParser` in vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py.

Here is a summary of a plugin file:

```python

# import the required packages

# define a tool parser and register it to vllm
# the name list in register_module can be used
# in --tool-call-parser. you can define as many
# tool parsers as you want here.
@ToolParserManager.register_module(["example"])
class ExampleToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    # adjust request. e.g.: set skip special tokens
    # to False for tool call output.
    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        return request

    # implement the tool call parse for stream call
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        return delta

    # implement the tool parse for non-stream call
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=text)


```

Then you can use this plugin in the command line like this.
```
    --enable-auto-tool-choice \
    --tool-parser-plugin <absolute path of the plugin file>
    --tool-call-parser example \
    --chat-template <your chat template> \
```

