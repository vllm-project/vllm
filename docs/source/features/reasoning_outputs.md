(reasoning-outputs)=

# Reasoning Outputs

vLLM offers support for reasoning models like [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), which are designed to generate outputs containing both reasoning steps and final conclusions.

Reasoning models return an additional `reasoning_content` field in their outputs, which contains the reasoning steps that led to the final conclusion. This field is not present in the outputs of other models.

## Supported Models

vLLM currently supports the following reasoning models:

| Model Series | Parser Name | Structured Output Support | Tool Calling |
|--------------|-------------|------------------|-------------|
| [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `deepseek_r1` | `guided_json`, `guided_regex` | ❌ |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `guided_json`, `guided_regex` | ✅ |
| [IBM Granite 3.2 language models](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a) | `granite` | ❌ | ❌ |
| [Qwen3 series](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `qwen3` | `guided_json`, `guided_regex` | ✅ |

:::{note}
IBM Granite 3.2 reasoning is disabled by default; to enable it, you must also pass `thinking=True` in your `chat_template_kwargs`.
:::

## Quickstart

To use reasoning models, you need to specify the `--reasoning-parser` flags when making a request to the chat completion endpoint. The `--reasoning-parser` flag specifies the reasoning parser to use for extracting reasoning content from the model output.

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

Next, make a request to the model that should return the reasoning content in the response.

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
# For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content:", reasoning_content)
print("content:", content)
```

The `reasoning_content` field contains the reasoning steps that led to the final conclusion, while the `content` field contains the final conclusion.

## Streaming chat completions

Streaming chat completions are also supported for reasoning models. The `reasoning_content` field is available in the `delta` field in [chat completion response chunks](https://platform.openai.com/docs/api-reference/chat/streaming).

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1694268190,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "reasoning_content": "is",
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
```

OpenAI Python client library does not officially support `reasoning_content` attribute for streaming output. But the client supports extra attributes in the response. You can use `hasattr` to check if the `reasoning_content` attribute is present in the response. For example:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
# For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
stream = client.chat.completions.create(model=model,
                                        messages=messages,
                                        stream=True)

print("client: Start streaming chat completions...")
printed_reasoning_content = False
printed_content = False

for chunk in stream:
    reasoning_content = None
    content = None
    # Check the content is reasoning_content or content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    elif hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content

    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("reasoning_content:", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    elif content is not None:
        if not printed_content:
            printed_content = True
            print("\ncontent:", end="", flush=True)
        # Extract and print the content
        print(content, end="", flush=True)
```

Remember to check whether the `reasoning_content` exists in the response before accessing it. You could checkout the [example](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py).

## Structured output

The reasoning content is also available in the structured output. The structured output engine like `xgrammar` will use the reasoning content to generate structured output. It is only supported in v0 engine now.

```bash
VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

Please note that the `VLLM_USE_V1` environment variable must be set to `0` to use the v0 engine.

```python
from openai import OpenAI
from pydantic import BaseModel

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


class People(BaseModel):
    name: str
    age: int


json_schema = People.model_json_schema()

prompt = ("Generate a JSON with the name and age of one random person.")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)
```

## Tool Calling

The reasoning content is also available when both tool calling and the reasoning parser are enabled. Additionally, tool calling only parses functions from the `content` field, not from the `reasoning_content`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [{
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
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)

print(response)
tool_call = response.choices[0].message.tool_calls[0].function

print(f"reasoning_content: {response.choices[0].message.reasoning_content}")
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```

For more examples, please refer to <gh-file:examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py>.

## Limitations

- The reasoning content is only available for online serving's chat completion endpoint (`/v1/chat/completions`).

## How to support a new reasoning model

You can add a new `ReasoningParser` similar to <gh-file:vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py>.

```python
# import the required packages

from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)

# define a reasoning parser and register it to vllm
# the name list in register_module can be used
# in --reasoning-parser.
@ReasoningParserManager.register_module(["example"])
class ExampleParser(ReasoningParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """
```

Additionally, to enable structured output, you'll need to create a new `Reasoner` similar to the one in <gh-file:vllm/model_executor/guided_decoding/reasoner/deepseek_reasoner.py>.

```python
@dataclass
class DeepSeekReasoner(Reasoner):
    """
    Reasoner for DeepSeek R series models.
    """
    start_token_id: int
    end_token_id: int

    start_token: str = "<think>"
    end_token: str = "</think>"

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
        return cls(start_token_id=tokenizer.encode(
            "<think>", add_special_tokens=False)[0],
                   end_token_id=tokenizer.encode("</think>",
                                                 add_special_tokens=False)[0])

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
    ...
```

The structured output engine like [xgrammar](https://github.com/mlc-ai/xgrammar) will use `end_token_id` to check if the reasoning content is present in the model output and skip the structured output if it is the case.

Finally, you can enable reasoning for the model by using the `--reasoning-parser` flags.

```bash
vllm serve <model_tag> --reasoning-parser example
```
