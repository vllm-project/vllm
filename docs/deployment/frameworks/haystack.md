# Haystack

[Haystack](https://github.com/deepset-ai/haystack) is an end-to-end LLM framework that allows you to build applications powered by LLMs, Transformer models, vector search and more. Whether you want to perform retrieval-augmented generation (RAG), document search, question answering or answer generation, Haystack can orchestrate state-of-the-art embedding models and LLMs into pipelines to build end-to-end NLP applications and solve your use case.

It allows you to deploy a large language model (LLM) server with vLLM as the backend, which exposes OpenAI-compatible endpoints.

## Prerequisites

Set up the vLLM and Haystack environment:

```bash
pip install vllm haystack-ai
```

## Deploy

1. Start the vLLM server with the supported chat completion model, e.g.

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.1
    ```

1. Use the `OpenAIGenerator` and `OpenAIChatGenerator` components in Haystack to query the vLLM server.

??? code

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    generator = OpenAIChatGenerator(
        # for compatibility with the OpenAI API, a placeholder api_key is needed
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model="mistralai/Mistral-7B-Instruct-v0.1",
        api_base_url="http://{your-vLLM-host-ip}:{your-vLLM-host-port}/v1",
        generation_kwargs={"max_tokens": 512},
    )

    response = generator.run(
      messages=[ChatMessage.from_user("Hi. Can you help me plan my next trip to Italy?")]
    )

    print("-"*30)
    print(response)
    print("-"*30)
    ```

```console
------------------------------
{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=' Of course! Where in Italy would you like to go and what type of trip are you looking to plan?')], _name=None, _meta={'model': 'mistralai/Mistral-7B-Instruct-v0.1', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 23, 'prompt_tokens': 21, 'total_tokens': 44, 'completion_tokens_details': None, 'prompt_tokens_details': None}})]}
------------------------------
```

For details, see the tutorial [Using vLLM in Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/vllm.md).
