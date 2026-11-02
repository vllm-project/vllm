# Codex

[Codex](https://github.com/openai/codex) is OpenAI's official agentic coding tool that lives in your terminal. It can understand your codebase, edit files, run commands, and help you write code more efficiently.

By pointing Codex at a vLLM server, you can use your own models as the backend instead of the OpenAI API. This is useful for:

- Running fully local/private coding assistance
- Using open-weight models with tool calling capabilities
- Testing and developing with custom models

## How It Works

vLLM implements the OpenAI-Responses API, which is the same API that Codex uses to communicate with OpenAI's servers. By configuring Codex to point at your vLLM server, Codex sends its requests to vLLM instead of OpenAI. vLLM then translates these requests to work with your local model and returns responses in the format Codex expects.

This means any model served by vLLM with proper tool calling support can act as a drop-in replacement for OpenAI models in Codex.

## Requirements

Codex requires a model with strong tool calling capabilities. The model must support the OpenAI-Responses tool calling API. See [Tool Calling](../../features/tool_calling.md) for details on enabling tool calling for your model.

## Installation

First, install Codex by following the [official installation guide](https://github.com/openai/codex).

## Starting the vLLM Server

Start vLLM with a tool-calling capable model - here's an example using `Qwen/Qwen3-27B`:

```bash
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

```

For other models, you'll need to enable tool calling explicitly with `--enable-auto-tool-choice` and the right `--tool-call-parser`. Refer to the [Tool Calling documentation](../../features/tool_calling.md) for the correct flags for your model.

## Configuring Codex

Codex is configured via a TOML file located at `~/.codex/config.toml`. Create or edit this file to point Codex at your vLLM server:

```toml
model = "my-model"
model_provider = "vllm"

[model_providers.vllm]
name = "vLLM"
env_key = "VLLM_API_KEY"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

The configuration fields:

| Field | Description |
| ----- | ----------- |
| `model` | The model name to use. Must match the `--served-model-name` you passed to vLLM. |
| `model_provider` | Set to `"vllm"` to use your local vLLM server. |
| `[model_providers.vllm]` | Configuration section for the vLLM provider. |
| `name` | A display name for your vLLM provider. |
| `env_key` | The name of an environment variable that Codex will read for the API key. vLLM does not require authentication by default, so this can be any value. |
| `base_url` | The URL of your vLLM server's OpenAI-compatible API endpoint (default is `http://localhost:8000/v1`). |
| `wire_api` | The API style to use. Set to `"responses"` for the OpenAI Responses API |

!!! tip
    You can set the `env_key` to any dummy environment variable since vLLM doesn't require authentication by default:
    ```bash
    export VLLM_API_KEY=dummy
    ```

!!! warning
    When using the `responses` API, ensure your vLLM version supports the OpenAI Responses API.

## Testing the Setup

Once Codex is configured, launch it in your project directory:

```bash
codex
```

Try a simple prompt to verify the connection, such as asking it to explain a file in your project. If the model responds correctly, your setup is working. You can now use Codex with your vLLM-served model for coding tasks.

## Troubleshooting

**Connection refused**: Ensure vLLM is running and accessible at the specified URL. Check that the port matches and that `base_url` includes the `/v1` path suffix.

**Tool calls not working**: Verify that your model supports tool calling and that you've enabled it with the correct `--tool-call-parser` flag. See [Tool Calling](../../features/tool_calling.md).

**Model not found**: Ensure the `model` field in `~/.codex/config.toml` matches the `--served-model-name` you passed to vLLM.
