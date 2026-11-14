# Atomic Agent

[Atomic Agent](https://github.com/AtomicBot-ai/atomic-agent) is an open source (MIT) local-first AI agent for the terminal and the desktop. It reads and edits files, runs shell commands you approve, drives a browser and calls MCP tools. It can use a vLLM server as its model through its OpenAI-compatible provider.

## Prerequisites

Set up the vLLM environment:

```bash
pip install vllm
```

## Deploy

1. Start the vLLM server with a model that supports tool calling. Atomic Agent sends OpenAI `tools` in every request, so enable automatic tool choice and pick the parser for your model, e.g.

    ```bash
    vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
    ```

1. Install Atomic Agent:

    ```bash
    curl -fsSL https://atomicagent.io/install | sh
    ```

1. Add vLLM as a provider in `~/.atomic-agent/config.json` and make it the active one:

    ```json
    {
      "llm": {
        "activeTextProvider": "vllm",
        "providers": [
          {
            "id": "vllm",
            "kind": "openai-compatible",
            "baseUrl": "http://{vllm server host}:{vllm server port}",
            "defaultChatModel": "Qwen/Qwen3-8B"
          }
        ]
      }
    }
    ```

    - `baseUrl` works with or without a trailing `/v1`.
    - If you started vLLM with `--api-key`, add `"apiKey": "..."` to the provider.
    - Instead of editing the file, you can run `/llm` in the agent, press `n` in the Cloud pane and choose "OpenAI-compatible API (custom base URL)".

1. Start the agent and chat:

    ```bash
    atomic-agent
    ```

## Notes

- If your model returns tool calls as `<tool_call>` text instead of structured calls, Atomic Agent decodes them automatically.
- For an older vLLM build that rejects `tool` role messages, add `"messageShape": "flat"` to the provider.
- Atomic Agent's [Fusion](https://atomicagent.io/fusion/) mode can use a vLLM provider as the orchestrator or as the workers, next to another provider.
