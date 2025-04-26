(deployment-anything-llm)=

# Anything LLM

[Anything LLM](https://github.com/Mintplex-Labs/anything-llm) is a full-stack application that enables you to turn any document, resource, or piece of content into context that any LLM can use as references during chatting.

It allows you to deploy a large language model (LLM) server with vLLM as the backend, which exposes OpenAI-compatible endpoints.

## Prerequisites

- Setup vLLm environment

## Deploy

1. Start the vLLM server with the supported chat completion model, e.g.

```console
vllm serve Qwen/Qwen1.5-32B-Chat-AWQ --max-model-len 4096
```

1. Download and install [Anything LLM desktop](https://anythingllm.com/desktop).

2. On the bottom left of open settings, AI Prooviders --> LLM:
   - LLM Provider: Generic OpenAI
   - Base URL: http://{vllm server host}:{vllm server port}/v1
   - Chat Model Name: `Qwen/Qwen1.5-32B-Chat-AWQ`

:::{image} /assets/deployment/anything-llm-provider.png
:::

1. Back to home page, New Workspace --> create `vllm` workspace, and start to chat:

:::{image} /assets/deployment/anything-llm-chat-without-doc.png
:::

1. Click the upload button:
   - upload the doc
   - select the doc and move to the workspace
   - save and embed

:::{image} /assets/deployment/anything-llm-upload-doc.png
:::

1. Chat again:

:::{image} /assets/deployment/anything-llm-chat-with-doc.png
:::
