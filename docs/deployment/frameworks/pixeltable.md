# Pixeltable

[Pixeltable](https://github.com/pixeltable/pixeltable) is declarative, incremental multimodal data infrastructure for AI apps. Tables store inputs and model outputs together, and computed columns update incrementally when the data changes. See the [Pixeltable documentation](https://docs.pixeltable.com).

## Prerequisites

Set up the Pixeltable and vLLM environment. `openai` is only needed for the OpenAI-compatible server below:

```bash
pip install pixeltable vllm openai
```

## Deploy

### In-process chat completions

`chat_completions` in `pixeltable.functions.vllm` runs a Hugging Face model with vLLM and stores the request output on the table. The generated text is `outputs[0].text`.

1. Add the model as a computed column:

??? code

    ```python
    import pixeltable as pxt
    from pixeltable.functions import vllm

    pxt.create_dir("vllm_demo")
    t = pxt.create_table("vllm_demo/chat", {"input": pxt.String})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t.input},
    ]

    t.add_computed_column(
        result=vllm.chat_completions(
            messages,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            sampling_params={"max_tokens": 128},
        )
    )
    t.add_computed_column(output=t.result.outputs[0].text)

    t.insert([{"input": "What is the capital of France?"}])
    print(t.select(t.input, t.output).collect())
    ```

### OpenAI-compatible server

`pixeltable.functions.openai` builds its client from `api_key`, `base_url`, and `api_version`. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL`, or the same keys in the `openai` section of the Pixeltable config, to the vLLM server.

1. Start the vLLM server with a chat model, e.g.

    ```bash
    vllm serve Qwen/Qwen2.5-0.5B-Instruct
    ```

1. Point the Pixeltable OpenAI client at that server. `api_key` is required; use a placeholder unless the server was started with `--api-key`:

    ```bash
    export OPENAI_API_KEY=EMPTY
    export OPENAI_BASE_URL=http://localhost:8000/v1
    ```

1. Add a computed column that calls `chat_completions`:

??? code

    ```python
    import pixeltable as pxt
    from pixeltable.functions.openai import chat_completions

    pxt.create_dir("vllm_server_demo")
    t = pxt.create_table("vllm_server_demo/chat", {"prompt": pxt.String})

    messages = [{"role": "user", "content": t.prompt}]

    t.add_computed_column(
        response=chat_completions(
            messages,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            model_kwargs={"max_tokens": 128},
        )
    )
    ```

For details, see the tutorial [Working with vLLM](https://docs.pixeltable.com/howto/providers/working-with-vllm).
