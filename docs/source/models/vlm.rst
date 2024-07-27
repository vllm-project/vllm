.. _vlm:

Using VLMs
==========

vLLM provides experimental support for Vision Language Models (VLMs). See the :ref:`list of supported VLMs here <supported_vlms>`.
This document shows you how to run and serve these models using vLLM.

.. important::
    We are actively iterating on VLM support. Expect breaking changes to VLM usage and development in upcoming releases without prior deprecation.

    Currently, the support for vision language models on vLLM has the following limitations:

    * Only single image input is supported per text prompt.

    We are continuously improving user & developer experience for VLMs. Please `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_ if you have any feedback or feature requests.

Offline Batched Inference
-------------------------

To initialize a VLM, the aforementioned arguments must be passed to the ``LLM`` class for instantiating the engine.

.. code-block:: python

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

.. important::
    We have removed all vision language related CLI args in the ``0.5.1`` release. **This is a breaking change**, so please update your code to follow
    the above snippet. Specifically, ``image_feature_size`` is no longer required to be specified as we now calculate that
    internally for each model.


To pass an image to the model, note the following in :class:`vllm.inputs.PromptInputs`:

* ``prompt``: The prompt should follow the format that is documented on HuggingFace.
* ``multi_modal_data``: This is a dictionary that follows the schema defined in :class:`vllm.multimodal.MultiModalDataDict`. 

.. code-block:: python

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Load the image using PIL.Image
    image = PIL.Image.open(...)
    
    # Single prompt inference
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    
    # Batch inference
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/offline_inference_vision_language.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py>`_.


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

    vllm serve llava-hf/llava-1.5-7b-hf --chat-template template_llava.jinja

.. important::
    We have removed all vision language related CLI args in the ``0.5.1`` release. **This is a breaking change**, so please update your code to follow
    the above snippet. Specifically, ``image_feature_size`` is no longer required to be specified as we now calculate that
    internally for each model.

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
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
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
    There is no need to format the prompt in the API request since it will be handled by the server.
