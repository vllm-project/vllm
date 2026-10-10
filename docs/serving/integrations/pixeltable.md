# Pixeltable

[Pixeltable](https://github.com/pixeltable/pixeltable) is an open source multimodal backend for building AI applications in Python. The vLLM integration lets you add model inference as a computed column, so inputs and outputs stay together in one queryable table.

To install Pixeltable with vLLM support, run

```bash
pip install pixeltable vllm -q
```

This example uses a vision language model to describe an image. Pixeltable stores the image and model output together in a table.

??? code

    ```python
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    import pixeltable as pxt
    import pixeltable.functions as pxtf

    t = pxt.create_table(
        "vision", {"image": pxt.Image}, if_exists="replace_force"
    )

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/"
        "Machu_Picchu%2C_Peru_%282018%29.jpg/"
        "1280px-Machu_Picchu%2C_Peru_%282018%29.jpg"
    )
    t.insert([{"image": image_url}])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": t.image},
                {"type": "text", "text": "Describe this image in two sentences."},
            ],
        }
    ]

    t.add_computed_column(
        result=pxtf.vllm.chat_completions(
            messages,
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            engine_args={"max_model_len": 4096, "dtype": "half"},
            sampling_params={"max_tokens": 128},
        )
    )
    t.add_computed_column(output=t.result.outputs[0].text)

    t.select(t.image, t.output).collect()
    ```

Please refer to this [Tutorial](https://docs.pixeltable.com/howto/providers/working-with-vllm) for more details.
