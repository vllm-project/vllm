# Tool Calling

vLLM currently supports named function calling, as well as the `auto`, `required` (as of `vllm>=0.8.3`), and `none` options for the `tool_choice` field in the chat completion API.

## Quickstart

Start the server with tool calling enabled. This example uses Meta's Llama 3.1 8B model, so we need to use the `llama3_json` tool calling chat template from the vLLM examples directory:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

Next, make a request that triggers the model to use the available tools:

??? code

    ```python
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    def get_weather(location: str, unit: str):
        return f"Getting the weather for {location} in {unit}..."
    tool_functions = {"get_weather": get_weather}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location", "unit"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
    ```

Example output:

```text
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

This example demonstrates:

* Setting up the server with tool calling enabled
* Defining an actual function to handle tool calls
* Making a request with `tool_choice="auto"`
* Handling the structured response and executing the corresponding function

You can also specify a particular function using named function calling by setting `tool_choice={"type": "function", "function": {"name": "get_weather"}}`. Note that this will use the structured outputs backend - so the first time this is used, there will be several seconds of latency (or more) as the FSM is compiled for the first time before it is cached for subsequent requests.

Remember that it's the caller's responsibility to:

1. Define appropriate tools in the request
2. Include relevant context in the chat messages
3. Handle the tool calls in your application logic

For more advanced usage, including parallel tool calls and different model-specific parsers, see the sections below.

## Named Function Calling

vLLM supports named function calling in the chat completion API by default. This should work with most structured outputs backends supported by vLLM. You are guaranteed a validly-parsable function call - not a
high-quality one.

vLLM will use structured outputs to ensure the response matches the tool parameter object defined by the JSON schema in the `tools` parameter.
For best results, we recommend ensuring that the expected output format / schema is specified in the prompt to ensure that the model's intended generation is aligned with the schema that it's being forced to generate by the structured outputs backend.

To use a named function, you need to define the functions in the `tools` parameter of the chat completion request, and
specify the `name` of one of the tools in the `tool_choice` parameter of the chat completion request.

## Required Function Calling

vLLM supports the `tool_choice='required'` option in the chat completion API. Similar to the named function calling, it also uses structured outputs, so this is enabled by default and will work with any supported model. However, support for alternative decoding backends are on the [roadmap](../usage/v1_guide.md#features) for the V1 engine.

When tool_choice='required' is set, the model is guaranteed to generate one or more tool calls based on the specified tool list in the `tools` parameter. The number of tool calls depends on the user's query. The output format strictly follows the schema defined in the `tools` parameter.

## None Function Calling

vLLM supports the `tool_choice='none'` option in the chat completion API. When this option is set, the model will not generate any tool calls and will respond with regular text content only, even if tools are defined in the request.

!!! note
    When tools are specified in the request, vLLM includes tool definitions in the prompt by default, regardless of the `tool_choice` setting. To exclude tool definitions when `tool_choice='none'`, use the `--exclude-tools-when-tool-choice-none` option.

## Automatic Function Calling

To enable this feature, you should set the following flags:

* `--enable-auto-tool-choice` -- **mandatory** Auto tool choice. It tells vLLM that you want to enable the model to generate its own tool calls when it
deems appropriate.
* `--tool-call-parser` -- select the tool parser to use (listed below). Additional tool parsers
will continue to be added in the future. You can also register your own tool parsers in the `--tool-parser-plugin`.
* `--tool-parser-plugin` -- **optional** tool parser plugin used to register user defined tool parsers into vllm, the registered tool parser name can be specified in `--tool-call-parser`.
* `--chat-template` -- **optional** for auto tool choice. It's the path to the chat template which handles `tool`-role messages and `assistant`-role messages
that contain previously generated tool calls. Hermes, Mistral and Llama models have tool-compatible chat templates in their
`tokenizer_config.json` files, but you can specify a custom template. This argument can be set to `tool_use` if your model has a tool use-specific chat
template configured in the `tokenizer_config.json`. In this case, it will be used per the `transformers` specification. More on this [here](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)
from HuggingFace; and you can find an example of this in a `tokenizer_config.json` [here](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json).

If your favorite tool-calling model is not supported, please feel free to contribute a parser & tool use chat template!

### Hermes Models (`hermes`)

All Nous Research Hermes-series models newer than Hermes 2 Pro should be supported.

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_Note that the Hermes 2 **Theta** models are known to have degraded tool call quality and capabilities due to the merge
step in their creation_.

Flags: `--tool-call-parser hermes`

### Mistral Models (`mistral`)

Supported models:

* `mistralai/Mistral-7B-Instruct-v0.3` (confirmed)
* Additional Mistral function-calling models are compatible as well.

Known issues:

1. Mistral 7B struggles to generate parallel tool calls correctly.
2. **For Transformers tokenization backend only**: Mistral's `tokenizer_config.json` chat template requires tool call IDs that are exactly 9 digits, which is
   much shorter than what vLLM generates. Since an exception is thrown when this condition
   is not met, the following additional chat templates are provided:

    * [examples/tool_chat_template_mistral.jinja](../../examples/tool_chat_template_mistral.jinja) - this is the "official" Mistral chat template, but tweaked so that
      it works with vLLM's tool call IDs (provided `tool_call_id` fields are truncated to the last 9 digits)
    * [examples/tool_chat_template_mistral_parallel.jinja](../../examples/tool_chat_template_mistral_parallel.jinja) - this is a "better" version that adds a tool-use system prompt
      when tools are provided, that results in much better reliability when working with parallel tool calling.

Recommended flags:

1. To use the official Mistral AI's format:

    `--tool-call-parser mistral`

2. To use the Transformers format when available:

    `--tokenizer_mode hf --config_format hf --load_format hf --tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

!!! note
    Models officially released by Mistral AI have two possible formats:

    1. The official format that is used by default with `auto` or `mistral` arguments:

        `--tokenizer_mode mistral --config_format mistral --load_format mistral`
        This format uses [mistral-common](https://github.com/mistralai/mistral-common), the Mistral AI's tokenizer backend.

    2. The Transformers format, when available, that is used with `hf` arguments:

        `--tokenizer_mode hf --config_format hf --load_format hf --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama Models (`llama3_json`)

Supported models:

All Llama 3.1, 3.2 and 4 models should be supported.

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

The tool calling that is supported is the [JSON-based tool calling](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling). For [pythonic tool calling](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling) introduced by the Llama-3.2 models, see the `pythonic` tool parser below. As for Llama 4 models, it is recommended to use the `llama4_pythonic` tool parser.

Other tool calling formats like the built-in python tool calling or custom tool calling are not supported.

Known issues:

1. Parallel tool calls are not supported for Llama 3, but it is supported in Llama 4 models.
2. The model can generate parameters in an incorrect format, such as generating
   an array serialized as string instead of an array.

VLLM provides two JSON-based chat templates for Llama 3.1 and 3.2:

* [examples/tool_chat_template_llama3.1_json.jinja](../../examples/tool_chat_template_llama3.1_json.jinja) - this is the "official" chat template for the Llama 3.1
models, but tweaked so that it works better with vLLM.
* [examples/tool_chat_template_llama3.2_json.jinja](../../examples/tool_chat_template_llama3.2_json.jinja) - this extends upon the Llama 3.1 chat template by adding support for
images.

Recommended flags: `--tool-call-parser llama3_json --chat-template {see_above}`

VLLM also provides a pythonic and JSON-based chat template for Llama 4, but pythonic tool calling is recommended:

* [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) - this is based on the [official chat template](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/) for the Llama 4 models.

For Llama 4 model, use `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja`.

### IBM Granite

Supported models:

* `ibm-granite/granite-4.0-h-small` and other Granite 4.0 models

    Recommended flags: `--tool-call-parser hermes`

* `ibm-granite/granite-3.0-8b-instruct`

    Recommended flags: `--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

    [examples/tool_chat_template_granite.jinja](../../examples/tool_chat_template_granite.jinja): this is a modified chat template from the original on Hugging Face. Parallel function calls are supported.

* `ibm-granite/granite-3.1-8b-instruct`

    Recommended flags: `--tool-call-parser granite`

    The chat template from Huggingface can be used directly. Parallel function calls are supported.

* `ibm-granite/granite-20b-functioncalling`

    Recommended flags: `--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

    [examples/tool_chat_template_granite_20b_fc.jinja](../../examples/tool_chat_template_granite_20b_fc.jinja): this is a modified chat template from the original on Hugging Face, which is not vLLM-compatible. It blends function description elements from the Hermes template and follows the same system prompt as "Response Generation" mode from [the paper](https://arxiv.org/abs/2407.00121). Parallel function calls are supported.

### InternLM Models (`internlm`)

Supported models:

* `internlm/internlm2_5-7b-chat` (confirmed)
* Additional internlm2.5 function-calling models are compatible as well

Known issues:

* Although this implementation also supports InternLM2, the tool call results are not stable when testing with the `internlm/internlm2-chat-7b` model.

Recommended flags: `--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba Models (`jamba`)

AI21's Jamba-1.5 models are supported.

* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`

Flags: `--tool-call-parser jamba`

### xLAM Models (`xlam`)

The xLAM tool parser is designed to support models that generate tool calls in various JSON formats. It detects function calls in several different output styles:

1. Direct JSON arrays: Output strings that are JSON arrays starting with `[` and ending with `]`
2. Thinking tags: Using `<think>...</think>` tags containing JSON arrays
3. Code blocks: JSON in code blocks (```json ...```)
4. Tool calls tags: Using `[TOOL_CALLS]` or `<tool_call>...</tool_call>` tags

Parallel function calls are supported, and the parser can effectively separate text content from tool calls.

Supported models:

* Salesforce Llama-xLAM models: `Salesforce/Llama-xLAM-2-8B-fc-r`, `Salesforce/Llama-xLAM-2-70B-fc-r`
* Qwen-xLAM models: `Salesforce/xLAM-1B-fc-r`, `Salesforce/xLAM-3B-fc-r`, `Salesforce/Qwen-xLAM-32B-fc-r`

Flags:

* For Llama-based xLAM models: `--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_llama.jinja`
* For Qwen-based xLAM models: `--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_qwen.jinja`

### Qwen Models

For Qwen2.5, the chat template in tokenizer_config.json has already included support for the Hermes-style tool use. Therefore, you can use the `hermes` parser to enable tool calls for Qwen models. For more detailed information, please refer to the official [Qwen documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)

* `Qwen/Qwen2.5-*`
* `Qwen/QwQ-32B`

Flags: `--tool-call-parser hermes`

### MiniMax Models (`minimax_m1`)

Supported models:

* `MiniMaxAi/MiniMax-M1-40k` (use with [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja))
* `MiniMaxAi/MiniMax-M1-80k` (use with [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja))

Flags: `--tool-call-parser minimax --chat-template examples/tool_chat_template_minimax_m1.jinja`

### DeepSeek-V3 Models (`deepseek_v3`)

Supported models:

* `deepseek-ai/DeepSeek-V3-0324` (use with [examples/tool_chat_template_deepseekv3.jinja](../../examples/tool_chat_template_deepseekv3.jinja))
* `deepseek-ai/DeepSeek-R1-0528` (use with [examples/tool_chat_template_deepseekr1.jinja](../../examples/tool_chat_template_deepseekr1.jinja))

Flags: `--tool-call-parser deepseek_v3 --chat-template {see_above}`

### DeepSeek-V3.1 Models (`deepseek_v31`)

Supported models:

* `deepseek-ai/DeepSeek-V3.1` (use with [examples/tool_chat_template_deepseekv31.jinja](../../examples/tool_chat_template_deepseekv31.jinja))

Flags: `--tool-call-parser deepseek_v31 --chat-template {see_above}`

### Kimi-K2 Models (`kimi_k2`)

Supported models:

* `moonshotai/Kimi-K2-Instruct`

Flags: `--tool-call-parser kimi_k2`

### Hunyuan Models (`hunyuan_a13b`)

Supported models:

* `tencent/Hunyuan-A13B-Instruct` (The chat template is already included in the Hugging Face model files.)

Flags:

* For non-reasoning: `--tool-call-parser hunyuan_a13b`
* For reasoning: `--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`

### LongCat-Flash-Chat Models (`longcat`)

Supported models:

* `meituan-longcat/LongCat-Flash-Chat`
* `meituan-longcat/LongCat-Flash-Chat-FP8`

Flags: `--tool-call-parser longcat`

### GLM-4.5 Models (`glm45`)

Supported models:

* `zai-org/GLM-4.5`
* `zai-org/GLM-4.5-Air`
* `zai-org/GLM-4.6`
* `zai-org/GLM-4.6-Air`

Flags: `--tool-call-parser glm45`

### Qwen3-Coder Models (`qwen3_xml`)

Supported models:

* `Qwen/Qwen3-480B-A35B-Instruct`
* `Qwen/Qwen3-Coder-30B-A3B-Instruct`

Flags: `--tool-call-parser qwen3_xml`

### Olmo 3 Models (`olmo3`)

Olmo 3 models output tool calls in a format that is very similar to the one expected by the `pythonic` parser (see below), with a few differences. Each tool call is a pythonic string, but the parallel tool calls are newline-delimited, and the calls are wrapped within XML tags as `<function_calls>..</function_calls>`. In addition, the parser also allows JSON boolean and null literals (`true`, `false`, and `null`) in addition to the pythonic ones (`True`, `False`, and `None`).

Supported models:

* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-32B-Think`

Flags: `--tool-call-parser olmo3`

### Gigachat 3 Models (`gigachat3`)

Use chat template from the Hugging Face model files.

Supported models:

* `ai-sage/GigaChat3-702B-A36B-preview`
* `ai-sage/GigaChat3-702B-A36B-preview-bf16`
* `ai-sage/GigaChat3-10B-A1.8B`
* `ai-sage/GigaChat3-10B-A1.8B-bf16`

Flags: `--tool-call-parser gigachat3`

### Models with Pythonic Tool Calls (`pythonic`)

A growing number of models output a python list to represent tool calls instead of using JSON. This has the advantage of inherently supporting parallel tool calls and removing ambiguity around the JSON schema required for tool calls. The `pythonic` tool parser can support such models.

As a concrete example, these models may look up the weather in San Francisco and Seattle by generating:

```python
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]
```

Limitations:

* The model must not generate both text and tool calls in the same generation. This may not be hard to change for a specific model, but the community currently lacks consensus on which tokens to emit when starting and ending tool calls.  (In particular, the Llama 3.2 models emit no such tokens.)
* Llama's smaller models struggle to use tools effectively.

Example supported models:

* `meta-llama/Llama-3.2-1B-Instruct` ⚠️ (use with [examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja))
* `meta-llama/Llama-3.2-3B-Instruct` ⚠️ (use with [examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja))
* `Team-ACE/ToolACE-8B` (use with [examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja))
* `fixie-ai/ultravox-v0_4-ToolACE-8B` (use with [examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja))
* `meta-llama/Llama-4-Scout-17B-16E-Instruct` ⚠️ (use with [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja))
* `meta-llama/Llama-4-Maverick-17B-128E-Instruct` ⚠️ (use with [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja))

Flags: `--tool-call-parser pythonic --chat-template {see_above}`

!!! warning
    Llama's smaller models frequently fail to emit tool calls in the correct format. Results may vary depending on the model.

## How to Write a Tool Parser Plugin

A tool parser plugin is a Python file containing one or more ToolParser implementations. You can write a ToolParser similar to the `Hermes2ProToolParser` in [vllm/tool_parsers/hermes_tool_parser.py](../../vllm/tool_parsers/hermes_tool_parser.py).

Here is a summary of a plugin file:

??? code

    ```python

    # import the required packages

    # define a tool parser and register it to vllm
    # the name list in register_module can be used
    # in --tool-call-parser. you can define as many
    # tool parsers as you want here.
    class ExampleToolParser(ToolParser):
        def __init__(self, tokenizer: TokenizerLike):
            super().__init__(tokenizer)

        # adjust request. e.g.: set skip special tokens
        # to False for tool call output.
        def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
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
        ) -> DeltaMessage | None:
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
    # register the tool parser to ToolParserManager
    ToolParserManager.register_lazy_module(
        name="example",
        module_path="vllm.tool_parsers.example",
        class_name="ExampleToolParser",
    )

    ```

Then you can use this plugin in the command line like this.

```bash
    --enable-auto-tool-choice \
    --tool-parser-plugin <absolute path of the plugin file>
    --tool-call-parser example \
    --chat-template <your chat template> \
```
