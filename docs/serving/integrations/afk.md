# AFK

[AFK](https://afk.mooglest.com) is a browser-based command center for persistent coding-agent sessions. It can connect to a vLLM server through vLLM's OpenAI-compatible API, letting AFK sessions use models that you host yourself.

This is useful for:

- Running coding-agent sessions with local or self-hosted models
- Keeping model traffic on infrastructure that you control
- Testing open-weight or custom models in an agent workflow
- Supervising long-running coding tasks from a browser UI

## How It Works

vLLM exposes an OpenAI-compatible API server. AFK can use that endpoint as a local model connection by setting the Base URL to the vLLM server's `/v1` endpoint.

AFK sessions run through a connected daemon. If you use a URL such as `http://localhost:8000/v1`, `localhost` must be reachable from the daemon machine that will run the session.

## Requirements

AFK coding sessions use tool calls for file edits, shell commands, and other actions. Use a model with strong tool calling support, and start vLLM with the tool-calling options required by that model. See [Tool Calling](../../features/tool_calling.md) for details.

## Starting the vLLM Server

Start vLLM with an OpenAI-compatible server endpoint. For example:

```bash
vllm serve openai/gpt-oss-120b \
    --served-model-name my-model \
    --enable-auto-tool-choice \
    --tool-call-parser openai
```

The default server listens on port `8000`, so the OpenAI-compatible endpoint is:

```text
http://localhost:8000/v1
```

For other models, choose the appropriate `--tool-call-parser` and serving flags for that model.

## Configuring AFK

1. Open [AFK](https://afk.mooglest.com) and sign in.
2. Install and connect an AFK daemon on the machine that can reach your vLLM server and project files.
3. Open **Account → LLM**.
4. Add a **Local model** connection.
5. Set **Base URL** to your vLLM OpenAI-compatible endpoint:

   ```text
   http://localhost:8000/v1
   ```

6. Bind the connection to the daemon that can reach vLLM.
7. Save the connection.

!!! tip
    If your vLLM server is on another machine, use a network URL that the daemon can reach, such as `http://vllm-host:8000/v1`.

!!! note
    If your vLLM endpoint requires an API key, configure AFK with an OpenAI-compatible connection using the vLLM Base URL and the API key expected by your gateway or server.

## Starting an AFK Session

Click **New session** in AFK, then:

1. Select the daemon and project directory.
2. Choose the vLLM local model connection.
3. Select or type the served model name, for example:

   ```text
   my-model
   ```

4. Choose a permission mode.
5. Enter the coding task and start the session.

AFK will route the session's model requests through vLLM while the browser UI shows progress, tool usage, diffs, approvals, and session history.

## Troubleshooting

**Connection refused**: Ensure vLLM is running and reachable from the daemon machine. Check that the Base URL includes the `/v1` path suffix.

**Tool calls not working**: Verify that the model supports tool calling and that vLLM was started with the correct `--enable-auto-tool-choice` and `--tool-call-parser` options. See [Tool Calling](../../features/tool_calling.md).

**Model not found**: Ensure the model name selected or typed in AFK matches the `--served-model-name` passed to vLLM.

**`localhost` confusion**: In AFK, local model requests are made from the daemon/session environment. `localhost` refers to that machine, not the hosted AFK web app.
