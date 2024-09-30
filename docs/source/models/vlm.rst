.. _vlm:

Using VLMs
==========

vLLM provides experimental support for Vision Language Models (VLMs). See the :ref:`list of supported VLMs here <supported_vlms>`.
This document shows you how to run and serve these models using vLLM.

.. important::
    We are actively iterating on VLM support. Expect breaking changes to VLM usage and development in upcoming releases without prior deprecation.

    We are continuously improving user & developer experience for VLMs. Please `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_ if you have any feedback or feature requests.

Offline Inference
-----------------

Single-image input
^^^^^^^^^^^^^^^^^^

The :class:`~vllm.LLM` class can be instantiated in much the same way as language-only models.

.. code-block:: python

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

.. note::
    We have removed all vision language related CLI args in the ``0.5.1`` release. **This is a breaking change**, so please update your code to follow
    the above snippet. Specifically, ``image_feature_size`` can no longer be specified as we now calculate that internally for each model.

To pass an image to the model, note the following in :class:`vllm.inputs.PromptType`:

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

    # Inference with image embeddings as input
    image_embeds = torch.load(...) # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # Inference with image embeddings as input with additional parameters
    # Specifically, we are conducting a trial run of Qwen2VL with the new input format, as the model utilizes additional parameters for calculating positional encoding.
    image_embeds = torch.load(...) # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
    image_grid_thw = torch.load(...) # torch.Tensor of shape (1, 3)
    mm_data['image'] = {
        "image_embeds": image_embeds,
        "image_grid_thw":  image_grid_thw,
    }
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": mm_data,
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

Multi-image input
^^^^^^^^^^^^^^^^^

Multi-image input is only supported for a subset of VLMs, as shown :ref:`here <supported_vlms>`.

To enable multiple multi-modal items per text prompt, you have to set ``limit_mm_per_prompt`` for the :class:`~vllm.LLM` class.

.. code-block:: python

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    )

Instead of passing in a single image, you can pass in a list of images.

.. code-block:: python

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "<|user|>\n<image_1>\n<image_2>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # Load the images using PIL.Image
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": [image1, image2]
        },
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/offline_inference_vision_language_multi_image.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language_multi_image.py>`_.

Online Inference
----------------

OpenAI Vision API
^^^^^^^^^^^^^^^^^

You can serve vision language models with vLLM's HTTP server that is compatible with `OpenAI Vision API <https://platform.openai.com/docs/guides/vision>`_.

Below is an example on how to launch the same ``microsoft/Phi-3.5-vision-instruct`` with vLLM's OpenAI-compatible API server.

.. code-block:: bash

    vllm serve microsoft/Phi-3.5-vision-instruct --max-model-len 4096 \
      --trust-remote-code --limit-mm-per-prompt image=2

.. important::
    Since OpenAI Vision API is based on `Chat Completions <https://platform.openai.com/docs/api-reference/chat>`_ API,
    a chat template is **required** to launch the API server.

    Although Phi-3.5-Vision comes with a chat template, for other models you may have to provide one if the model's tokenizer does not come with it.
    The chat template can be inferred based on the documentation on the model's HuggingFace repo.
    For example, LLaVA-1.5 (``llava-hf/llava-1.5-7b-hf``) requires a chat template that can be found `here <https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja>`_.

To consume the server, you can use the OpenAI client like in the example below:

.. code-block:: python

    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Single-image input inference
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

    # Multi-image input inference
    image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in these images?"},
                {"type": "image_url", "image_url": {"url": image_url_duck}},
                {"type": "image_url", "image_url": {"url": image_url_lion}},
            ],
        }],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)


A full code example can be found in `examples/openai_vision_api_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_vision_api_client.py>`_.

.. note::

    By default, the timeout for fetching images through http url is ``5`` seconds. You can override this by setting the environment variable:

    .. code-block:: shell

        export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>

.. note::
    There is no need to format the prompt in the API request since it will be handled by the server.
