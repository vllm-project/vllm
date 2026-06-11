# Reasoning Examples

This directory contains examples demonstrating how to use reasoning models with vLLM. Reasoning models like Qwen3-4B-Thinking, DeepSeek-R1, and others generate both reasoning (thinking) content and final answers.

## Available Examples

### Offline Examples

These examples run without starting a server and directly use the vLLM Python API:

#### `offline_chat_completion_with_reasoning.py`

Demonstrates offline chat completion with reasoning extraction for models like Qwen3-4B-Thinking.

**Usage:**

```bash
# Run with default model (Qwen3-4B-Thinking-2507)
python examples/reasoning/offline_chat_completion_with_reasoning.py

# Run with custom model
python examples/reasoning/offline_chat_completion_with_reasoning.py \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --max-tokens 512 \
    --temperature 0.7

# Disable thinking/reasoning mode
python examples/reasoning/offline_chat_completion_with_reasoning.py \
    --disable-thinking

# Use with tensor parallelism
python examples/reasoning/offline_chat_completion_with_reasoning.py \
    --tensor-parallel-size 2
```

**Features:**
- Multi-turn conversation with reasoning extraction
- Shows raw model output and separated reasoning/content
- Configurable thinking mode (enabled by default)
- Demonstrates the `Qwen3ReasoningParser` for extracting reasoning

### Server-Based Examples

These examples require starting a vLLM server first:

#### `openai_chat_completion_with_reasoning.py`

Demonstrates basic chat completion with reasoning using the OpenAI-compatible API.

**Start the server:**

```bash
# For Qwen3-4B-Thinking model
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}'

# For DeepSeek-R1 model
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

**Run the example:**

```bash
python examples/reasoning/openai_chat_completion_with_reasoning.py
```

#### `openai_chat_completion_with_reasoning_streaming.py`

Demonstrates streaming chat completion with reasoning.

**Start the server** (same as above), then:

```bash
python examples/reasoning/openai_chat_completion_with_reasoning_streaming.py
```

#### `openai_chat_completion_tool_calls_with_reasoning.py`

Demonstrates tool calling combined with reasoning.

**Start the server:**

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3coder \
    --default-chat-template-kwargs '{"enable_thinking": true}'
```

**Run the example:**

```bash
python examples/reasoning/openai_chat_completion_tool_calls_with_reasoning.py
```

## Supported Models

The following reasoning models are supported by vLLM:

### Qwen3 Family
- `Qwen/Qwen3-4B-Thinking-2507`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- Other Qwen3/Qwen3.5 models with thinking capabilities

**Reasoning parser:** `qwen3`

**Features:**
- Uses `<think>` and `</think>` tags for reasoning
- Supports enabling/disabling thinking via `enable_thinking` parameter
- Can be combined with tool calling

### DeepSeek-R1 Family
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

**Reasoning parser:** `deepseek_r1`

### Other Models

vLLM supports many other reasoning models. Check `vllm/reasoning/` for the full list of available reasoning parsers including:
- `deepseek_v3`
- `gemma4`
- `granite`
- `mistral`
- `minimax_m2`
- And more!

## Requirements

### Environment Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

### For Server-Based Examples

Install the OpenAI Python client:

```bash
uv pip install openai
```

## Key Concepts

### Reasoning vs Content

Reasoning models generate two types of output:

1. **Reasoning** (thinking): Internal thought process, problem analysis, step-by-step reasoning
2. **Content**: The final answer to present to the user

The reasoning parsers automatically separate these two parts.

### Enabling/Disabling Thinking

For Qwen3 models, you can control thinking mode:

**Server mode:**
```bash
# Enable thinking (default)
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}'

# Disable thinking
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Offline mode:**
```python
# Enable thinking (default)
llm = LLM(
    model="Qwen/Qwen3-4B-Thinking-2507",
    chat_template_kwargs={"enable_thinking": True}
)

# Disable thinking
llm = LLM(
    model="Qwen/Qwen3-4B-Thinking-2507",
    chat_template_kwargs={"enable_thinking": False}
)
```

### Reasoning Parsers

Each model family has its own reasoning parser that knows how to extract reasoning and content:

- `Qwen3ReasoningParser`: For Qwen3/Qwen3.5 models (uses `<think>` tags)
- `DeepSeekR1ReasoningParser`: For DeepSeek-R1 models
- And more in `vllm/reasoning/`

## Tips

1. **Offline vs Server**: Use offline examples for simpler integration and direct API access. Use server-based examples when you need OpenAI API compatibility.

2. **Token Budget**: Reasoning models can generate long thinking sequences. Set appropriate `max_tokens` values (e.g., 512-2048).

3. **Temperature**: Lower temperatures (0.0-0.3) work well for reasoning tasks. Higher temperatures (0.7-1.0) encourage more creative thinking.

4. **Multi-turn Conversations**: Always pass the `content` (not the reasoning) when building conversation history for subsequent turns.

5. **Tool Calling**: Qwen3 models support both reasoning and tool calling simultaneously. The reasoning parser handles tool calls that appear within or after the thinking block.

## Troubleshooting

### Model Not Found

Make sure you have access to the model on Hugging Face and the model is downloaded:

```bash
huggingface-cli login
```

### Out of Memory

Try reducing `max_tokens` or using tensor parallelism:

```bash
# Offline
python example.py --max-tokens 256 --tensor-parallel-size 2

# Server
vllm serve model_name --tensor-parallel-size 2
```

### No Reasoning Extracted

1. Check that you're using the correct reasoning parser for your model
2. Verify that thinking is enabled (for Qwen3 models)
3. Some prompts may not trigger reasoning - try more complex questions

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Reasoning Parsers Source Code](../../vllm/reasoning/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat)
