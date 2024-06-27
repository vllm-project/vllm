.. _vlm:

Using VLMs
==========

vLLM provides experimental support for Vision Language Models (VLMs). This document shows you how to run and serve these models using vLLM.

.. important::
    We are actively iterating on VLM support. Expect breaking changes to VLM usage and development in upcoming releases without prior deprecation.

Engine Arguments
----------------

The following :ref:`engine arguments <engine_args>` are specific to VLMs:

.. argparse::
    :module: vllm.engine.arg_utils
    :func: _vlm_engine_args_parser
    :prog: -m vllm.entrypoints.openai.api_server
    :nodefaultconst:

.. important::
    Currently, the support for vision language models on vLLM has the following limitations:

    * Only single image input is supported per text prompt.
    * Dynamic ``image_input_shape`` is not supported: the input image will be resized to the static ``image_input_shape``. This means our LLaVA-NeXT output may not exactly match the huggingface implementation.

    We are continuously improving user & developer experience for VLMs. Please `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_ if you have any feedback or feature requests.

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

.. important::
    We will remove most of the vision-specific arguments in a future release as they can be inferred from the HuggingFace configuration.


To pass an image to the model, note the following in :class:`vllm.inputs.PromptStrictInputs`:

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

.. important::
    We will remove the need to format image tokens in a future release. Afterwards, the input text will follow the same format as that for the original HuggingFace model.

Online OpenAI Vision API Compatible Inference
----------------------------------------------

You can serve vision language models with vLLM's HTTP server that is compatible with `OpenAI Vision API <https://platform.openai.com/docs/guides/vision>`_.

.. note::
    Currently, vLLM supports only **single** ``image_url`` input per ``messages``. Support for multi-image inputs will be
    added in the future.

Below is an example on how to launch the same ``llava-hf/llava-1.5-7b-hf`` with vLLM API server.

.. important::
    Since OpenAI Vision API is based on `Chat <https://platform.openai.com/docs/api-reference/chat>`_ API, a chat template 
    is **required** to launch the API server if the model's tokenizer does not come with one. In this example, we use the 
    HuggingFace Llava chat template that you can find in the example folder `here <https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja>`_.

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model llava-hf/llava-1.5-7b-hf \
        --image-input-type pixel_values \
        --image-token-id 32000 \
        --image-input-shape 1,3,336,336 \
        --image-feature-size 576 \
        --chat-template template_llava.jinja

.. important::
    We will remove most of the vision-specific arguments in a future release as they can be inferred from the HuggingFace configuration.

To consume the server, you can use the OpenAI client like in the example below:

.. code-block:: python

    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="llava-hf/llava-1.5-7b-hf",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }],
    )
    print("Chat response:", chat_response)

A full code example can be found in `examples/openai_vision_api_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_vision_api_client.py>`_.

.. note::

    By default, the timeout for fetching images through http url is ``5`` seconds. You can override this by setting the environment variable:

    .. code-block:: shell

        export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>

.. note::
    The prompt formatting with the image token ``<image>`` is not needed when serving VLMs with the API server since the prompt will be 
    processed automatically by the server.
