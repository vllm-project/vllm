# Pixeltable

vLLM is also available via [Pixeltable](https://github.com/pixeltable/pixeltable).

To install Pixeltable with vLLM support, run

```bash
pip install pixeltable vllm -q
```

To run inference on a single or multiple GPUs, use the `vllm` functions from `pixeltable.functions` in computed columns.

??? code

    ```python
    import pixeltable as pxt
    import pixeltable.functions as pxtf

    pxt.create_dir("vllm_demo")
    t = pxt.create_table("vllm_demo/chat", {"input": pxt.String})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t.input},
    ]

    t.add_computed_column(
        result=pxtf.vllm.chat_completions(
            messages, model="Qwen/Qwen2.5-0.5B-Instruct"
        )
    )
    t.add_computed_column(output=t.result.outputs[0].text)

    t.insert([{"input": "What is the capital of France?"}])
    t.select(t.input, t.output).collect()
    ```

Please refer to this [Tutorial](https://docs.pixeltable.com/howto/providers/working-with-vllm) for more details.
