.. _vlm:

Using VLMs
==========

This document shows you how to run and serve Vision Language Models (VLMs) using vLLM.

Additional Engine Arguments
---------------------------

Apart from the :ref:`basic engine arguments <engine_args>`, VLMs additionally require the following engine arguments for vLLM.

.. option:: --image-input-type {pixel_values,image_features}

    The image input type passed into vLLM. Should be one of "pixel_values" or "image_features".

.. option:: --image-token-id <id>

    Input ID for image token.

.. option:: --image-input-shape <tuple>

    The biggest image input shape (worst for memory footprint) given an input type. Only used for vLLM's profile_run.

    For example, if the image tensor has shape :code:`(1, 3, 336, 336)`, then you should pass :code:`--image-input-shape 1,3,336,336`.

.. option:: --image-feature-size <size>

    The image feature size along the context dimension.

.. option:: --image-processor <size>

    Name or path of the huggingface image processor to use.

.. option:: --image-processor-revision <revision>

    The specific image processor version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

.. option:: --no-image-processor

    Disables the use of image processor, even if one is defined for the model on huggingface.

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

For now, we only support a single image per text prompt when calling ``llm.generate``. To pass an image to the model, note the following parameters:

* ``prompt``: The prompt should have a number of ``<image>`` tokens equal to ``image_feature_size``.
* ``multi_modal_data``: This should be an instance of ``MultiModalData`` with type ``MultiModalData.Type.IMAGE`` and its data set to a single image tensor with the shape ``image_input_shape``.

.. code-block:: python

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # Load the image and reshape to (1, 3, 336, 336)
    image = ...

    outputs = llm.generate(prompt,
                           multi_modal_datas=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=image))

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
