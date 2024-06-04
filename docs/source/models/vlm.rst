.. _vlm:

Using VLMs
==========

This document shows you how to run and serve Vision Language Models (VLMs) using vLLM.

Engine Arguments
----------------

The following :ref:`engine arguments <engine_args>` are specific to VLMs:

.. argparse::
    :module: vllm.engine.arg_utils
    :func: _vlm_engine_args_parser
    :prog: -m vllm.entrypoints.openai.api_server
    :nodefaultconst:

Offline Batched Inference
-------------------------

To initialize a VLM, the aforementioned arguments must be passed to the ``LLM`` class for instantiating the engine.

.. code-block:: python

    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
    )

For now, we only support a single image per text prompt. To pass an image to the model, note the following in :class:`vllm.inputs.PromptStrictInputs`:

* ``prompt``: The prompt should have a number of ``<image>`` tokens equal to ``image_feature_size``.
* ``multi_modal_data``: This should be an instance of :class:`~vllm.multimodal.image.ImagePixelData` or :class:`~vllm.multimodal.image.ImageFeatureData`.

.. code-block:: python

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # Load the image using PIL.Image
    image = ...

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": ImagePixelData(image),
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/llava_example.py <https://github.com/vllm-project/vllm/blob/main/examples/llava_example.py>`_.

OpenAI-Compatible Server
------------------------

We support image inputs to the OpenAI Chat API, as described in `GPT-4 with Vision <https://platform.openai.com/docs/guides/vision>`_.

Here is a simple example using the :code:`openai` package:

.. code-block:: python

    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Note that this model expects the image to come before the main text
    chat_response = client.chat.completions.create(
        model="llava-hf/llava-1.5-7b-hf",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
                {"type": "text", "text": "What's in this image?"},
            ],
        }],
    )
    print("Chat response:", chat_response)

.. note::

    For now, we only support a single image per API call. Also, the ``detail`` parameter is ignored since it may not be applicable to other models.
